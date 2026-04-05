"""
adaptive_executor.py — Neuro-DJ Real-Time Bounded Adaptation
════════════════════════════════════════════════════════════
Stages 7 & 8 of the roadmap: The "Hands on the Mixer".

While `neuro_gui.py` is the brain that plans the transition, this class 
acts as the DJ's hands during the actual blend. It constantly polls 
`live_ears.py` (the DJ's ears) and injects real-time surgical EQ commands 
into the C++ audio engine if things start to go wrong.

Key Interventions:
  1. Sonic Mud Prevention: If the combined bass of both tracks exceeds 
     safe thresholds, it drops Deck B's bass to 35% for 8 beats.
  2. Drift Masking: If the tracks fall out of phase (beat offset), it 
     heavily cuts the outgoing track's high frequencies to hide the 
     clashing hi-hats from the crowd.
  3. Early Handoff: If the new track's groove is fully established and 
     the old track unexpectedly dies (loses density), it forces an early 
     swap to save the dancefloor energy.
"""

import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class AdaptationLog:
    """Records real-time interventions to be appended to the transition dataset."""
    actions_taken: List[str] = field(default_factory=list)
    handoff_forced: bool = False


class AdaptiveExecutor:
    def __init__(self, mixer, ears, plan: dict, spb: float):
        self._mixer = mixer
        self._ears = ears
        self._plan = plan
        self._spb = spb
        self._log = AdaptationLog()
        
        # State tracking to prevent spamming the C++ engine
        self._bass_suppressed_until = 0.0
        self._drift_masked = False
        
        # Determine duration boundaries
        self._trans_dur = plan.get('trans_dur', 30.0)
        self._start_time = time.time()

    def run(self) -> AdaptationLog:
        """
        Blocks and monitors the transition while it is active.
        Returns the adaptation log once the transition completes or is forced.
        """
        print("   🤖 Adaptive Executor online — monitoring mix bus...")
        
        while self._mixer.is_transitioning():
            # 1. Fetch live perception data
            state = self._ears.get_state()
            
            # 2. Get accurate file positions for scheduling EQ events
            try:
                a_pos = self._mixer.get_position("A")
                b_pos = self._mixer.get_position("B")
            except Exception:
                break  # Mixer closed or crashed

            elapsed_wall_time = time.time() - self._start_time
            
            # ── INTERVENTION 1: Heavy Bass Suppression (Sonic Mud) ──
            # If the master bus low-end is highly congested, we duck B's bass
            if state.combined_low_congestion > 0.65 and b_pos > self._bass_suppressed_until:
                # Instantly drop B's bass to 35%
                self._mixer.add_eq_event(1, b_pos, 0.35, 1.0, 1.0)
                
                # Schedule it to restore after 8 beats
                restore_time = b_pos + (8.0 * self._spb)
                self._mixer.add_eq_event(1, restore_time, 1.0, 1.0, 1.0)
                
                self._bass_suppressed_until = restore_time + (4.0 * self._spb) # Cooldown
                
                msg = f"HEAVY_BASS_HOLD_cong={state.combined_low_congestion:.2f}"
                self._log.actions_taken.append(msg)
                print(f"   🎛️  ADAPT: {msg} — Ducking B's bass for 8 beats")

            # ── INTERVENTION 2: Phase Drift Masking (Trainwreck Guard) ──
            # If the beats drift apart by > 15% of a beat, hi-hats will flam/clash.
            # Real DJs mask this by killing the outgoing track's highs.
            if abs(state.beat_offset) > 0.15 and not self._drift_masked:
                # Muffle A's highs (0.2x) and mids (0.7x) to hide the drift
                self._mixer.add_eq_event(0, a_pos, 1.0, 0.7, 0.2)
                
                msg = f"DRIFT_MASKING_offset={state.beat_offset:.2f}"
                self._log.actions_taken.append(msg)
                self._drift_masked = True
                print(f"   🎛️  ADAPT: {msg} — High-passing Deck A to hide flamming")

            # ── INTERVENTION 3: Early Handoff (The Bailout) ──
            # If B is fully locked in, A is dead, and we are at least 30% into the mix, bail out.
            if elapsed_wall_time > (self._trans_dur * 0.30):
                if state.readiness_score > 0.85 and state.a_space_opened and state.b_groove_established:
                    self._log.handoff_forced = True
                    self._log.actions_taken.append("EARLY_HANDOFF_TRIGGERED")
                    print(f"   🎛️  ADAPT: Perfect conditions met early — forcing handoff!")
                    break  # Break loop to return to NeuroDJ which executes the force

            # Sleep for 100ms (10 Hz polling rate)
            time.sleep(0.1)

        return self._log