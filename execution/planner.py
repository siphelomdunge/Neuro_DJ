"""
Transition plan finalization and validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import os
import librosa

from core.constants import (
    TOTAL_SAFETY_MARGIN, TECHNIQUE_LIBRARY, TECHNIQUE_SAFE_MIN_BEATS,
    MINIMUM_EXIT_ROOM_SEC
)
from .warp import RunwayExhausted


@dataclass
class TransitionPlan:
    """Finalized transition plan ready for execution."""
    fnb: str
    b_ratio: float
    b_start_w: float
    mix_trigger: float
    trans_dur: float
    tech_name: str
    mem_key: str
    recipe: Dict
    track_b_dur: float
    overlap_score: Dict
    ctx: Dict
    assumptions: Dict
    master_bpm: float
    acapella_method: str
    
    def update_duration(self, new_dur: float, spb: float):
        """
        Atomically update transition duration and recipe beats.
        
        Args:
            new_dur: New transition duration (seconds)
            spb: Seconds per beat
        """
        self.trans_dur = new_dur
        self.recipe['beats'] = new_dur / spb
    
    def to_dict(self) -> dict:
        """
        Convert TransitionPlan to dictionary for backward compatibility.
        
        Returns:
            Dict representation of plan
        """
        return {
            'fnb': self.fnb,
            'b_ratio': self.b_ratio,
            'b_start_w': self.b_start_w,
            'mix_trigger': self.mix_trigger,
            'trans_dur': self.trans_dur,
            'tech_name': self.tech_name,
            'mem_key': self.mem_key,
            'recipe': self.recipe,
            'track_b_dur': self.track_b_dur,
            'overlap_score': self.overlap_score,
            'ctx': self.ctx,
            'assumptions': self.assumptions,
            'master_bpm': self.master_bpm,  # ✅ Use attribute name, not _master_bpm
            'acapella_method': self.acapella_method,
        }

class TransitionPlanner:
    """Finalizes transition plans with EOF validation."""
    
    def __init__(self, mixer, warper, brain, searcher, context_builder):
        """
        Initialize transition planner.
        
        Args:
            mixer: NeuroMixer instance
            warper: TrackWarper instance
            brain: DJBrain instance
            searcher: PhraseCandidateSearch instance
            context_builder: build_transition_context function
        """
        self.mixer = mixer
        self.warper = warper
        self.brain = brain
        self.searcher = searcher
        self.build_transition_context = context_builder
    
    def finalize_plan(self, decision, ta: dict, spb: float, track_a_dur: float,
                     master_bpm: float) -> TransitionPlan:
        """
        Finalize decision into executable transition plan.
        
        Args:
            decision: EvaluatedOption from DecisionCore
            ta: Track A metadata
            spb: Seconds per beat
            track_a_dur: Physical duration of track A
            master_bpm: Current master BPM
        
        Returns:
            TransitionPlan ready for execution
        
        Raises:
            RunwayExhausted: If insufficient time remaining
        """
        draft = decision.draft_plan
        
        if not draft or 'track_b' not in draft:
            return self._make_fallback_plan(ta, decision.option.track_b or ta, spb,
                                           track_a_dur, master_bpm)
        
        tb = draft['track_b']
        
        # Warp track B
        try:
            fnb, actual_b_ratio = self.warper.warp_track(tb, master_bpm)
            physical_b_dur = librosa.get_duration(path=fnb)
        except Exception as e:
            print(f"   ⚠️ Warp failed: {e}, using fallback")
            return self._make_fallback_plan(ta, tb, spb, track_a_dur, master_bpm)
        
        # Rescale track B metadata
        tb = self.warper.rescale_track_timestamps(tb, actual_b_ratio)
        
        # Extract plan parameters
        mix_trigger = draft['mix_trigger']
        recipe = draft['recipe']
        tech_name = draft['tech_name']
        b_start_w = draft['b_start_w']
        final_mem_key = draft['mem_key']
        b_ratio = actual_b_ratio
        
        # Adjust b_start_w if ratio changed significantly
        if abs(b_ratio - draft['b_ratio']) > 0.01:
            b_start_w = (draft['b_start_w'] * (draft['b_ratio'] / b_ratio)
                        if b_ratio > 0 else draft['b_start_w'])
        
        # Validate trigger against current position
        pos_now = self.mixer.get_position("A")
        min_trigger = pos_now + 8.0
        max_trigger = max(track_a_dur - 25.0, pos_now + 8.0)
        mix_trigger = max(min_trigger, min(mix_trigger, max_trigger))
        
        # Calculate real runway
        real_runway = track_a_dur - mix_trigger - TOTAL_SAFETY_MARGIN
        
        # ✅ FIX #5: Validate minimum runway
        if real_runway < 8.0:
            print(f"   ⚠️ Only {real_runway:.1f}s runway — aborting transition")
            return self._make_fallback_plan(ta, tb, spb, track_a_dur, master_bpm)
        
        print(f"   📏 Real runway: {real_runway:.1f}s ({real_runway/spb:.0f} beats)")
        
        # Calculate desired duration
        desired_beats = recipe['beats']
        desired_dur = desired_beats * spb
        old_tech = tech_name
        old_beats = recipe['beats']
        
        max_usable_runway = real_runway * 0.90
        
        # Shrink if needed
        if desired_dur > max_usable_runway:
            available_beats = max_usable_runway / spb
            safe_min = TECHNIQUE_SAFE_MIN_BEATS.get(tech_name, 16.0)
            
            if available_beats >= safe_min:
                new_beats = max(safe_min, 8.0 * int(available_beats / 8.0))
                recipe['beats'] = new_beats
                print(f"   ⚠️ Shrunk {tech_name}: {desired_beats:.0f}→{new_beats:.0f} beats "
                      f"(runway={real_runway:.1f}s, using {new_beats*spb:.1f}s)")
            else:
                echo_min = TECHNIQUE_SAFE_MIN_BEATS.get("ECHO_THROW", 16.0)
                if available_beats >= echo_min:
                    tech_name = "ECHO_THROW"
                    recipe['beats'] = max(echo_min, 8.0 * int(available_beats / 8.0))
                    print(f"   ⚠️ Switched to ECHO_THROW ({recipe['beats']:.0f} beats, "
                          f"runway={real_runway:.1f}s)")
                else:
                    print(f"   ⚠️ Only {available_beats:.0f} beats available "
                          f"(need {echo_min:.0f} minimum)")
                    return self._make_fallback_plan(ta, tb, spb, track_a_dur, master_bpm)
        
        trans_dur = recipe['beats'] * spb
        replan_needed = (tech_name != old_tech or abs(recipe['beats'] - old_beats) >= 16.0)
        
        # Replan if technique changed
        if replan_needed:
            try:
                best_a_exit, best_b_entry, _, _ = self.searcher.search(
                    ta, tb, b_ratio, trans_dur, spb, quiet=True
                )
                mix_trigger = max(min_trigger,
                                 min(best_a_exit,
                                     track_a_dur - trans_dur - TOTAL_SAFETY_MARGIN))
                b_start_w = best_b_entry
                print(f"   🔁 Replanned after duration change: A={mix_trigger:.1f}s B={b_start_w:.1f}s")
            except Exception as e:
                print(f"   ⚠️ Replan failed: {e}")
        
        # Final EOF validation
        max_physical_dur = track_a_dur - mix_trigger - 0.5
        
        if trans_dur > max_physical_dur:
            print(f"   🚨 RUNWAY PANIC: Transition ({trans_dur:.1f}s) overshoots EOF! "
                  f"Clamping to {max_physical_dur:.1f}s")
            
            # ✅ FIX #5: Hard abort if can't fit minimum
            if max_physical_dur < 2.0:
                raise RunwayExhausted(
                    f"Cannot fit transition: need {trans_dur:.1f}s, have {max_physical_dur:.1f}s"
                )
            
            trans_dur = max(2.0, max_physical_dur)
            recipe['beats'] = trans_dur / spb
            
            if recipe['beats'] < 16.0 and tech_name not in ("ECHO_THROW", "FILTER_SWEEP"):
                print(f"   ⚠️ Runway critically short ({recipe['beats']:.1f} beats). Forcing ECHO_THROW.")
                tech_name = "ECHO_THROW"
                recipe['technique_id'] = 3
                recipe['echo'] = 1.0
                recipe['bass'] = 0.90
        else:
            if mix_trigger + trans_dur + TOTAL_SAFETY_MARGIN > track_a_dur:
                new_max_dur = track_a_dur - mix_trigger - TOTAL_SAFETY_MARGIN
                if new_max_dur < 8.0 * spb:
                    print(f"   💀 Impossible to fit transition (need {trans_dur:.1f}s, "
                          f"have {new_max_dur:.1f}s)")
                    raise RunwayExhausted(
                        f"Impossible fit: need {trans_dur:.1f}s, have {new_max_dur:.1f}s"
                    )
                
                trans_dur = max(8.0 * spb, new_max_dur)
                recipe['beats'] = max(8.0, 8.0 * round((trans_dur / spb) / 8.0))
                trans_dur = recipe['beats'] * spb
                print(f"   ⚠️ Final safety clamp: {recipe['beats']:.0f} beats / {trans_dur:.1f}s")
        
        # Regenerate recipe with final technique
        actual_final_beats = recipe['beats']
        final_mem_key, recipe = self.brain.ml.generate_recipe(
            tech_name, ta.get('energy', 'High'),
            tb.get('energy', 'High'),
            ta.get('genre') == 'amapiano', quiet=True
        )
        recipe['beats'] = actual_final_beats
        
        # Handle acapella mashup
        acapella_method = 'none'
        if tech_name == "ACAPELLA_MASHUP":
            acapella_method = self._handle_acapella_stem(
                ta, mix_trigger, trans_dur, spb, recipe
            )
        
        # Build final context
        ctx = self.build_transition_context(
            ta, tb, spb,
            mix_out_time=mix_trigger, b_entry_time=b_start_w,
            trans_dur=trans_dur, b_ratio=b_ratio
        )
        
        # Print summary
        buf = track_a_dur - mix_trigger - trans_dur
        print(f"\n   {'═'*48}")
        print(f"   ✅ PLAN FINALIZED")
        print(f"   ├─ Trigger:    {mix_trigger:.1f}s  ({mix_trigger/track_a_dur*100:.0f}%)")
        print(f"   ├─ Technique:  {TECHNIQUE_LIBRARY[tech_name]['label']}")
        print(f"   ├─ Beats:      {recipe['beats']:.0f}")
        print(f"   ├─ Duration:   {trans_dur:.1f}s")
        print(f"   ├─ Trans ends: {mix_trigger+trans_dur:.1f}s  (buffer={buf:.1f}s)")
        print(f"   ├─ B cue:      {b_start_w:.1f}s")
        print(f"   └─ Bass swap:  beat {recipe['bass']*recipe['beats']:.0f} of {recipe['beats']:.0f}")
        print(f"   {'═'*48}")
        
        return TransitionPlan(
            fnb=fnb,
            b_ratio=b_ratio,
            b_start_w=b_start_w,
            mix_trigger=mix_trigger,
            trans_dur=trans_dur,
            tech_name=tech_name,
            mem_key=final_mem_key,
            recipe=recipe,
            track_b_dur=physical_b_dur,
            overlap_score=draft['overlap_score'],
            ctx=ctx,
            assumptions=decision.assumptions,
            master_bpm=master_bpm,
            acapella_method=acapella_method
        )
    
    def _handle_acapella_stem(self, ta: dict, mix_trigger: float,
                             trans_dur: float, spb: float, recipe: dict) -> str:
        """
        Load acapella harmonic stem if available.
        
        Returns:
            'hpss_stem', 'mid_side_realtime', or 'none'
        """
        harmonic_path = ta.get('harmonic_stem')
        
        if harmonic_path and not os.path.isabs(harmonic_path):
            ta_dir = os.path.dirname(ta.get('filename', ''))
            harmonic_path = os.path.join(ta_dir, harmonic_path)
        
        if harmonic_path and os.path.exists(harmonic_path):
            try:
                current_a_pos = self.mixer.get_position("A")
                stem_dur = librosa.get_duration(path=harmonic_path)
                
                # ✅ FIX #19: Safe position calculation
                if stem_dur < current_a_pos:
                    print(f"   ⚠️ Stem too short ({stem_dur:.1f}s), cannot maintain position")
                    safe_pos = max(0.0, stem_dur - 5.0)
                else:
                    safe_pos = min(current_a_pos, stem_dur - 5.0)
                
                self.mixer.load_deck("A", harmonic_path)
                self.mixer.seek("A", max(0.0, safe_pos))
                self.mixer.play("A")
                
                # ✅ FIX #20: Dynamic runway recalculation + plan sync
                new_runway = stem_dur - mix_trigger - TOTAL_SAFETY_MARGIN
                if new_runway < trans_dur:
                    print(f"   ⚠️ Stem shorter than expected — recalculating transition")
                    trans_dur = max(8.0 * spb, new_runway * 0.9)
                    recipe['beats'] = trans_dur / spb
                    # Note: trans_dur will be synced by caller via TransitionPlan.update_duration
                    print(f"   ✅ Adjusted transition to {trans_dur:.1f}s")
                
                print(f"   🎤 ACAPELLA: Loaded HPSS harmonic stem ({stem_dur:.1f}s)")
                return 'hpss_stem'
            
            except Exception as e:
                print(f"   ⚠️ Harmonic stem load failed: {e}, using M/S fallback")
                return 'mid_side_realtime'
        else:
            print(f"   🎤 ACAPELLA: Using real-time M/S extraction (no stem file)")
            return 'mid_side_realtime'
    
    def _make_fallback_plan(self, ta: dict, tb: dict, spb: float,
                           track_a_dur: float, master_bpm: float) -> TransitionPlan:
        """
        Generate emergency fallback plan (ECHO_THROW).
        
        Raises:
            RunwayExhausted: If even fallback won't fit
        """
        pos_now = self.mixer.get_position("A")
        
        physical_remaining = track_a_dur - pos_now
        trig = pos_now + 1.0
        
        max_safe_dur = max(2.0, physical_remaining - 1.5)
        desired_dur = 16.0 * spb
        tdur = min(desired_dur, max_safe_dur)
        actual_beats = max(4.0, tdur / spb)
        
        defs = dict(TECHNIQUE_LIBRARY['ECHO_THROW']['defaults'])
        defs.update({
            'beats': actual_beats,
            'technique_name': 'ECHO_THROW',
            'technique_id': 3,
            'echo': 1.0,
            'bass': 0.85
        })
        
        try:
            fnb, b_ratio = self.warper.warp_track(tb, master_bpm)
        except Exception as e:
            print(f"   ⚠️ Fallback warp failed: {e}")
            raise RunwayExhausted("Fallback warp failed")
        
        return TransitionPlan(
            fnb=fnb,
            b_ratio=b_ratio,
            b_start_w=0.0,
            mix_trigger=trig,
            trans_dur=tdur,
            tech_name='ECHO_THROW',
            mem_key=f"ECHO_THROW|{ta.get('energy','High')}->{tb.get('energy','High')}",
            recipe=defs,
            track_b_dur=tb.get('_duration', 300.0),
            overlap_score={},
            ctx={},
            assumptions={},
            master_bpm=master_bpm,
            acapella_method='none'
        )