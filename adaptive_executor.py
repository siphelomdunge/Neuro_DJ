import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class AdaptationLog:
    actions_taken: List[str] = field(default_factory=list)
    handoff_forced: bool = False


class AdaptiveExecutor:
    def __init__(self, mixer, ears, plan: dict, spb: float, telemetry=None):
        self._mixer = mixer
        self._ears = ears
        self._plan = plan
        self._spb = spb
        self._telemetry = telemetry
        self._log = AdaptationLog()

        self._bass_suppressed_until = 0.0
        self._drift_masked = False
        self._drift_unmask_time = 0.0

        self._trans_dur = plan.get('trans_dur', 30.0)
        self._start_time = time.time()

    def run(self) -> AdaptationLog:
        print("   🤖 Adaptive Executor online — monitoring mix bus...")

        while self._mixer.is_transitioning():
            state = self._ears.get_state()
            if state is None:
                time.sleep(0.1)
                continue

            try:
                a_pos = self._mixer.get_position("A")
                b_pos = self._mixer.get_position("B")
            except Exception:
                break

            elapsed_wall_time = time.time() - self._start_time
            beat_elapsed = elapsed_wall_time / self._spb

            if self._telemetry:
                self._telemetry.log_tick(beat_elapsed, state)

            # ── INTERVENTION 1: Heavy Bass Suppression ──
            if (state.combined_low_congestion > 0.65
                    and b_pos > self._bass_suppressed_until):
                self._mixer.add_eq_event(1, b_pos, 0.35, 1.0, 1.0)
                restore_time = b_pos + (8.0 * self._spb)
                self._mixer.add_eq_event(1, restore_time, 1.0, 1.0, 1.0)
                self._bass_suppressed_until = restore_time + (4.0 * self._spb)

                msg = (f"HEAVY_BASS_HOLD_cong="
                       f"{state.combined_low_congestion:.2f}")
                self._log.actions_taken.append(msg)
                if self._telemetry:
                    self._telemetry.log_action(beat_elapsed, msg, state)
                print(f"   🎛️  ADAPT: {msg} — Ducking B bass for 8 beats")

            # ── INTERVENTION 2: Phase Drift Masking (with restore) ──
            if self._drift_masked and time.time() > self._drift_unmask_time:
                self._drift_masked = False

            if abs(state.beat_offset) > 0.15 and not self._drift_masked:
                self._mixer.add_eq_event(0, a_pos, 1.0, 0.7, 0.2)
                restore_a = a_pos + (4.0 * self._spb)
                self._mixer.add_eq_event(0, restore_a, 1.0, 1.0, 1.0)
                self._drift_masked = True
                self._drift_unmask_time = time.time() + (8.0 * self._spb)

                msg = f"DRIFT_MASKING_offset={state.beat_offset:.2f}"
                self._log.actions_taken.append(msg)
                if self._telemetry:
                    self._telemetry.log_action(beat_elapsed, msg, state)
                print(f"   🎛️  ADAPT: {msg} — High-passing A for 4 beats")

            # ── INTERVENTION 3: Early Handoff ──
            if elapsed_wall_time > (self._trans_dur * 0.30):
                if (state.readiness_score > 0.85
                        and state.a_space_opened
                        and state.b_groove_established):
                    self._log.handoff_forced = True
                    msg = "EARLY_HANDOFF_TRIGGERED"
                    self._log.actions_taken.append(msg)
                    if self._telemetry:
                        self._telemetry.log_action(beat_elapsed, msg, state)
                    print("   🎛️  ADAPT: Perfect conditions — "
                          "forcing handoff!")
                    break

            time.sleep(0.1)

        return self._log