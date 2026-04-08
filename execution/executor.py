"""
Transition execution and real-time monitoring.
"""
from __future__ import annotations
import time
import random

from core.constants import (
    PRE_TRIGGER_BUFFER_SEC, EMERGENCY_HANDOFF_THRESHOLD, IMMEDIATE_TECHNIQUES
)

try:
    from adaptive_executor import AdaptiveExecutor
    ADAPTIVE_EXECUTOR_AVAILABLE = True
except ImportError:
    ADAPTIVE_EXECUTOR_AVAILABLE = False

try:
    from live_ears import LiveEars
    LIVE_EARS_AVAILABLE = True
except ImportError:
    LIVE_EARS_AVAILABLE = False


class TransitionExecutor:
    """Executes transitions with real-time adaptation."""
    
    def __init__(self, mixer, master_bpm_fn, is_paused_fn, sleep_pausable_fn):
        """
        Initialize executor.
        
        Args:
            mixer: NeuroMixer instance
            master_bpm_fn: Callable returning current master BPM
            is_paused_fn: Callable returning pause state
            sleep_pausable_fn: Pausable sleep function
        """
        self.mixer = mixer
        self.master_bpm_fn = master_bpm_fn
        self.is_paused_fn = is_paused_fn
        self.sleep_pausable = sleep_pausable_fn
        self._ears = None
    
    def wait_for_trigger(self, plan, track_a_dur: float):
        """
        Wait until pre-trigger point, handling pauses and stalls.
        
        Args:
            plan: TransitionPlan
            track_a_dur: Physical duration of track A
        """
        stall_start = None
        last_pos = self.mixer.get_position("A")
        
        while self.mixer.get_position("A") < (plan.mix_trigger - PRE_TRIGGER_BUFFER_SEC):
            if self.is_paused_fn():
                time.sleep(0.1)
                stall_start = None
                last_pos = self.mixer.get_position("A")
                continue
            
            pos = self.mixer.get_position("A")
            if pos <= last_pos:
                if stall_start is None:
                    stall_start = time.time()
                elif time.time() - stall_start > 15.0:
                    print("   ⚠️ Audio stalled during pre-trigger — aborting wait")
                    break
            else:
                stall_start = None
            
            last_pos = pos
            time.sleep(0.5)
    
    def schedule_eq_events(self, ta: dict, tb: dict, plan, spb: float):
        """
        Schedule EQ automation events for transition.
        
        Args:
            ta: Track A metadata
            tb: Track B metadata
            plan: TransitionPlan
            spb: Seconds per beat
        """
        self.mixer.clear_eq_events()
        
        mix_trigger = plan.mix_trigger
        trans_dur = plan.trans_dur
        recipe = plan.recipe
        tech_name = plan.tech_name
        b_start_w = plan.b_start_w
        b_ratio = plan.b_ratio
        
        # Acapella mashup special handling
        if tech_name == "ACAPELLA_MASHUP":
            swap_elapsed = trans_dur * recipe['bass']
            swap_time = mix_trigger + swap_elapsed
            
            self.mixer.add_eq_event(1, b_start_w, 0.0, 1.0, 1.0)
            self.mixer.add_eq_event(1, swap_time, 1.0, 1.0, 1.0)
            
            kill_time = swap_time + 4.0 * spb
            self.mixer.add_eq_event(0, kill_time, 1.0, 0.0, 1.0)
            
            print(f"   🎤 ACAPELLA EQ: B bass opens @ {swap_time:.1f}s, "
                  f"A mids killed @ {kill_time:.1f}s")
            return
        
        # Standard transition
        swap_elapsed = trans_dur * recipe['bass']
        swap_time_a = mix_trigger + swap_elapsed
        
        stems_a = ta.get('stems', {})
        stems_b = tb.get('stems', {})
        vocal_regions_a = stems_a.get('vocal_regions', [])
        vocal_regions_b = stems_b.get('vocal_regions', [])
        log_drum_hits_b = stems_b.get('log_drum_hits', [])
        
        # Vocal clash detection
        a_has_vocal = any(start <= mix_trigger + trans_dur and end >= mix_trigger
                         for start, end in vocal_regions_a)
        
        scaled_vocal_b = ([(s / b_ratio, e / b_ratio) for s, e in vocal_regions_b]
                         if abs(b_ratio - 1.0) > 0.005 else vocal_regions_b)
        b_has_vocal = any(start < b_start_w + trans_dur and end > b_start_w
                         for start, end in scaled_vocal_b)
        
        if a_has_vocal and b_has_vocal:
            kill_time = swap_time_a - 2.0 * spb
            self.mixer.add_eq_event(
                0,
                kill_time if kill_time > mix_trigger else mix_trigger,
                1.0, 0.0, 1.0
            )
            print(f"   🎤 VOCAL CLASH GUARD: mid-kill on A @ {kill_time:.1f}s")
        elif a_has_vocal:
            if swap_time_a - spb > mix_trigger:
                self.mixer.add_eq_event(0, swap_time_a - spb, 1.0, 0.0, 1.0)
                print(f"   🎙️  PRE-SWAP MID-KILL on A @ {swap_time_a - spb:.1f}s")
        
        # Log drum sync
        scaled_drum_hits = ([t / b_ratio for t in log_drum_hits_b]
                           if abs(b_ratio - 1.0) > 0.005 else log_drum_hits_b)
        first_drum = next((t for t in scaled_drum_hits
                          if b_start_w <= t <= b_start_w + trans_dur), None)
        
        if first_drum is not None:
            self.mixer.add_eq_event(1, b_start_w, 0.0, 1.0, 1.0)
            self.mixer.add_eq_event(1, first_drum, 1.0, 1.0, 1.0)
            print(f"   🥁 LOG DRUM SYNC: B bass snaps open @ {first_drum:.1f}s")
    
    def trigger_transition(self, plan, telemetry=None):
        """
        Trigger and monitor transition execution.
        
        Args:
            plan: TransitionPlan
            telemetry: Optional TransitionTelemetry instance
        
        Returns:
            True if handoff was forced early
        """
        recipe = plan.recipe
        master_bpm = self.master_bpm_fn()
        
        # Set acapella mode if needed
        if plan.acapella_method == 'mid_side_realtime':
            try:
                self.mixer.set_acapella_mode("A", float(recipe.get('acapella', 1.0)))
            except Exception:
                pass
        
        # Trigger transition
        self.mixer.trigger_hybrid_transition(
            plan.trans_dur,
            recipe['beats'],
            recipe['bass'],
            recipe['echo'],
            0.0,
            recipe['wash'],
            float(master_bpm),
            float(recipe.get('piano_hold', 0.0)),
            recipe['technique_id'],
            random.uniform(1.8, 3.2),
            random.uniform(4.5, 7.5),
            random.uniform(0.0, 6.28),
            random.uniform(0.0, 6.28),
            max(0.005, min(0.020, random.gauss(0.010, 0.003))),
        )
        
        # Start live ears if available
        if LIVE_EARS_AVAILABLE:
            self._ears = LiveEars(self.mixer, float(master_bpm))
            self._ears.start()
            self._ears.notify_b_started()
        
        # Wait for transition to start
        pickup_deadline = time.time() + 3.0
        transition_started = False
        
        while time.time() < pickup_deadline:
            if self.mixer.is_transitioning():
                transition_started = True
                break
            time.sleep(0.01)
        
        if not transition_started:
            print("   🚨 ERROR: Audio engine failed to start transition! Forcing sync.")
            self.mixer.play("B")
            self.mixer.set_volume("B", 1.0)
            time.sleep(0.5)
            self.mixer.pause("A")
            return True
        
        # Monitor transition
        handoff_forced = self._monitor_transition(plan, telemetry)
        
        # Cleanup
        if self._ears is not None:
            self._ears.stop()
            self._ears = None
        
        return handoff_forced
    
    def _monitor_transition(self, plan, telemetry) -> bool:
        """
        Monitor transition with adaptive executor or fallback.
        
        Returns:
            True if handoff was forced
        """
        spb = 60.0 / max(self.master_bpm_fn(), 60.0)
        
        try:
            if ADAPTIVE_EXECUTOR_AVAILABLE and self._ears is not None:
                adapt_log = AdaptiveExecutor(
                    self.mixer, self._ears, plan, spb, telemetry
                ).run()
                
                if getattr(adapt_log, 'handoff_forced', False):
                    self.mixer.pause("A")
                    hd = time.time() + 2.0
                    while self.mixer.is_transitioning() and time.time() < hd:
                        time.sleep(0.02)
                    return True
                else:
                    td = time.time() + 15.0
                    while self.mixer.is_transitioning() and time.time() < td:
                        time.sleep(0.02)
                    return False
            else:
                # Fallback monitoring
                if not self.mixer.is_transitioning():
                    return False
                
                track_a_dur = plan.track_b_dur  # After swap, this will be updated
                td = time.time() + plan.trans_dur + 15.0
                
                while self.mixer.is_transitioning() and time.time() < td:
                    if self.is_paused_fn():
                        time.sleep(0.02)
                        continue
                    
                    pos_a = self.mixer.get_position("A")
                    # Note: track_a_dur not available here in original, using plan estimate
                    remaining_a = plan.trans_dur - (pos_a - plan.mix_trigger)
                    
                    # ✅ FIX: Emergency handoff threshold now 5.0s (Issue #9)
                    if remaining_a < EMERGENCY_HANDOFF_THRESHOLD:
                        print(f"   🚨 EMERGENCY HANDOFF: Track A has {remaining_a:.1f}s left!")
                        self.mixer.set_volume("A", 0.0)
                        self.mixer.set_volume("B", 1.0)
                        self.mixer.pause("A")
                        return True
                    
                    time.sleep(0.02)
                
                return False
        
        finally:
            # Always clear acapella mode
            try:
                self.mixer.set_acapella_mode("A", 0.0)
                self.mixer.set_acapella_mode("B", 0.0)
            except Exception:
                pass
    
    def post_transition_sleep(self, tech_name: str):
        """
        Brief pause after immediate techniques.
        
        Args:
            tech_name: Name of technique just executed
        """
        if tech_name in IMMEDIATE_TECHNIQUES:
            self.sleep_pausable(4.0)