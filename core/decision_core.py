"""
Decision Core — Intent generation and option evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
import time
import copy

from .constants import (
    TECHNIQUE_LIBRARY, TECHNIQUE_MIN_BEATS, TECHNIQUE_SAFE_MIN_BEATS,
    TECHNIQUE_IDEAL_BEATS, DANCE_MOMENT_SCORES, IMMEDIATE_TECHNIQUES,
    VALID_ENERGIES, MIN_STRETCH_RATIO, MAX_STRETCH_RATIO, MIN_SAFE_BPM
)


@dataclass
class Intent:
    """Strategic goal for the next transition."""
    goal: str
    target_energy_delta: float
    target_transition_intensity: float
    target_novelty: float
    vocal_tolerance: str
    harmonic_risk_tolerance: str
    preferred_entry_styles: list
    preferred_exit_styles: list
    max_execution_risk: float


@dataclass
class CandidateOption:
    """A potential action (transition or hold)."""
    option_type: str  # "transition" or "hold"
    track_b: Optional[dict]
    delay_beats: float
    
    def label(self) -> str:
        """Human-readable label for this option."""
        if self.option_type == "hold":
            return f"HOLD +{int(self.delay_beats)} beats"
        
        import os
        name = self.track_b.get('filename', '?') if self.track_b else '?'
        return f"TRANSITION {os.path.basename(name)} @ +{int(self.delay_beats)}b"


@dataclass
class EvaluatedOption:
    """An option with computed scores."""
    option: CandidateOption
    final_score: float
    intent_fit: float
    transition_opportunity: float
    global_compat: float
    robustness: float
    execution_risk: float
    future_flex: float
    reasoning: list = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    draft_plan: Optional[dict] = None


@dataclass
class DecisionRecord:
    """Record of a decision with alternatives."""
    intent: Intent
    chosen: EvaluatedOption
    alternatives: List[EvaluatedOption]
    timestamp: float


class IntentEngine:
    """Generates strategic intent based on set state."""
    
    def build_intent(self, current_track: dict, set_snapshot,
                    energy_history: List[str] = None,
                    transition_history: List[dict] = None,
                    set_duration_remaining: float = None) -> Intent:
        """
        Analyze set state and generate transition intent.
        
        Args:
            current_track: Track A metadata
            set_snapshot: SetStateModel snapshot (or None)
            energy_history: Recent energy levels
            transition_history: Recent transition records
            set_duration_remaining: Time left in set
        
        Returns:
            Intent object defining strategic goal
        """
        # Closing sequence
        if set_duration_remaining is not None and set_duration_remaining < 600:
            return Intent(
                "closing_sequence", -0.05, 0.35, 0.20, "low", "low",
                ["clean_intro", "harmonic_bed"],
                ["dj_friendly_outro", "decompression"], 0.30
            )
        
        # Recovery from failed transition
        if transition_history:
            for rec in reversed(transition_history):
                if rec.get('rating') is not None or rec.get('failure_tags'):
                    last_rating = rec.get('rating')
                    last_tags = rec.get('failure_tags', [])
                    
                    if last_rating is not None and last_rating <= 4:
                        return Intent(
                            "recovery", 0.00, 0.30, 0.15, "low", "low",
                            ["clean_intro", "drum_foundation"],
                            ["dj_friendly_outro", "decompression"], 0.25
                        )
                    
                    if 'vocal_clash' in last_tags or 'bass_fight' in last_tags:
                        return Intent(
                            "recovery", 0.00, 0.30, 0.15, "low", "low",
                            ["clean_intro"], ["dj_friendly_outro"], 0.25
                        )
                    break
        
        # Fallback if no set state available
        if set_snapshot is None:
            if energy_history and len(energy_history) >= 3:
                recent = energy_history[-3:]
                high_count = sum(1 for e in recent if e == "High")
                
                if high_count >= 3:
                    return Intent(
                        "controlled_rebuild", -0.10, 0.35, 0.25, "medium", "medium",
                        ["clean_intro", "drum_foundation"],
                        ["dj_friendly_outro", "decompression"], 0.35
                    )
                
                low_count = sum(1 for e in recent if e in ("Low/Chill",))
                if low_count >= 3:
                    return Intent(
                        "peak_lift", 0.20, 0.65, 0.40, "medium", "medium",
                        ["drum_foundation", "clean_intro"],
                        ["decompression", "dj_friendly_outro"], 0.42
                    )
                
                if len(set(recent)) == 1 and len(energy_history) > 5:
                    return Intent(
                        "harmonic_refresh", 0.05, 0.45, 0.55, "medium", "high",
                        ["clean_intro", "melodic"],
                        ["decompression", "harmonic_bed", "dj_friendly_outro"], 0.40
                    )
            
            return Intent(
                "balanced_progression", 0.0, 0.45, 0.35, "medium", "medium",
                ["clean_intro", "drum_foundation"],
                ["dj_friendly_outro", "decompression"], 0.45
            )
        
        # Set state analysis
        energy = float(getattr(set_snapshot, "energy_rolling", 0.5))
        vocal_fatigue = float(getattr(set_snapshot, "vocal_density_fatigue", 0.0))
        time_since_breather = float(getattr(set_snapshot, "time_since_breather", 0.0))
        time_since_hard_drop = float(getattr(set_snapshot, "time_since_hard_drop", 0.0))
        novelty = float(getattr(set_snapshot, "novelty_rate", 0.5))
        mix_count = int(getattr(set_snapshot, "mix_count", 0))
        
        # Vocal rest needed
        if vocal_fatigue > 0.8:
            return Intent(
                "vocal_rest", 0.00, 0.40, 0.30, "low", "medium",
                ["clean_intro", "drum_foundation"],
                ["dj_friendly_outro", "decompression"], 0.38
            )
        
        # Tension build
        if 0.55 < energy < 0.70 and time_since_hard_drop > 120 and mix_count > 2:
            return Intent(
                "tension_build", 0.10, 0.55, 0.30, "medium", "medium",
                ["drum_foundation", "clean_intro"],
                ["tension_build", "decompression"], 0.40
            )
        
        # Breather needed
        if energy > 0.72 and time_since_breather > 120:
            return Intent(
                "controlled_rebuild", -0.10, 0.35, 0.25,
                "low" if vocal_fatigue > 0.6 else "medium", "medium",
                ["clean_intro", "harmonic_bed", "drum_foundation"],
                ["dj_friendly_outro", "decompression"], 0.35
            )
        
        # Peak lift
        if energy < 0.45 and time_since_hard_drop > 180:
            return Intent(
                "peak_lift", 0.20, 0.65, 0.40, "medium", "medium",
                ["drum_foundation", "clean_intro"],
                ["decompression", "dj_friendly_outro"], 0.42
            )
        
        # Novelty refresh
        if novelty < 0.25:
            return Intent(
                "harmonic_refresh", 0.05, 0.45, 0.55, "medium", "high",
                ["clean_intro", "melodic"],
                ["decompression", "harmonic_bed", "dj_friendly_outro"], 0.40
            )
        
        # Default: hold groove
        return Intent(
            "hold_groove", 0.00, 0.45, 0.30,
            "low" if vocal_fatigue > 0.7 else "medium", "medium",
            ["clean_intro", "drum_foundation"],
            ["dj_friendly_outro", "decompression"], 0.42
        )


class DecisionCore:
    """Core decision-making engine for transition planning."""
    
    def __init__(self, brain, searcher, technique_selector: Callable,
                 context_builder: Callable, set_state_model=None,
                 technique_min_beats: dict = None, now_fn: Callable = None):
        """
        Initialize decision core.
        
        Args:
            brain: DJBrain instance
            searcher: PhraseCandidateSearch instance
            technique_selector: select_technique function
            context_builder: build_transition_context function
            set_state_model: SetStateModel instance (optional)
            technique_min_beats: Override min beats per technique
            now_fn: Time function (for testing)
        """
        self.brain = brain
        self.searcher = searcher
        self.select_technique = technique_selector
        self.build_transition_context = context_builder
        self.set_state_model = set_state_model
        self.intent_engine = IntentEngine()
        self.technique_min_beats = technique_min_beats or TECHNIQUE_SAFE_MIN_BEATS
        self.now_fn = now_fn
        self.dj = None  # Set externally by orchestrator
    
    def _now(self) -> float:
        """Get current timestamp."""
        return self.now_fn() if self.now_fn else time.time()
    
    def decide_next_action(self, current_track: dict, candidate_tracks: List[dict],
                          current_pos_a: float, track_a_dur: float, master_bpm: float,
                          spb: float, mix_count: int, energy_history: List[str],
                          transition_history: List[dict],
                          set_duration_remaining: float = None) -> DecisionRecord:
        """
        Evaluate all options and choose the best action.
        
        Returns:
            DecisionRecord with chosen option and alternatives
        """
        # Generate intent
        set_snapshot = (self.set_state_model.get_snapshot()
                       if self.set_state_model else None)
        
        intent = self.intent_engine.build_intent(
            current_track, set_snapshot, energy_history,
            transition_history, set_duration_remaining
        )
        
        # Generate candidate options
        options = self._generate_options(candidate_tracks)
        
        # Evaluate each option
        evaluated = []
        for option in options:
            try:
                ev = self._evaluate_option(
                    option, intent, current_track, current_pos_a,
                    track_a_dur, master_bpm, spb, mix_count
                )
                if ev:
                    evaluated.append(ev)
            except Exception as e:
                print(f"⚠️ Option evaluation failed: {e}")
        
        # Sort by score
        evaluated.sort(key=lambda x: x.final_score, reverse=True)
        chosen = evaluated[0] if evaluated else None
        
        return DecisionRecord(
            intent=intent,
            chosen=chosen,
            alternatives=evaluated,
            timestamp=self._now()
        )
    
    def _generate_options(self, candidate_tracks: List[dict]) -> List[CandidateOption]:
        """Generate candidate options (transitions + holds)."""
        opts = []
        
        # Transition options at various delays
        for tb in candidate_tracks:
            opts.append(CandidateOption("transition", tb, 32.0))
            opts.append(CandidateOption("transition", tb, 64.0))
            opts.append(CandidateOption("transition", tb, 96.0))
        
        # Hold options
        opts.append(CandidateOption("hold", None, 16.0))
        opts.append(CandidateOption("hold", None, 32.0))
        
        return opts
    
    def _safe_bpm_ratio(self, master_bpm: float, track_bpm: float) -> float:
        """
        Compute safe time-stretch ratio with division-by-zero guard.
        
        Args:
            master_bpm: Current master BPM
            track_bpm: Track's native BPM
        
        Returns:
            Ratio (1.0 if out of range or unsafe)
        """
        track_bpm = max(MIN_SAFE_BPM, track_bpm)
        ratio = master_bpm / track_bpm
        
        if ratio < MIN_STRETCH_RATIO or ratio > MAX_STRETCH_RATIO:
            return 1.0
        
        return ratio
    
    def _evaluate_option(self, option: CandidateOption, intent: Intent,
                        current_track: dict, current_pos_a: float,
                        track_a_dur: float, master_bpm: float, spb: float,
                        mix_count: int) -> Optional[EvaluatedOption]:
        """
        Evaluate a single option against intent.
        
        Returns:
            EvaluatedOption or None if invalid
        """
        if option.option_type == "hold":
            return self._evaluate_hold_option(
                option, intent, current_track, current_pos_a, track_a_dur, spb
            )
        
        # Transition option
        tb = dict(option.track_b) if option.track_b else {}
        
        # Future position after delay
        future_a_pos = current_pos_a + option.delay_beats * spb
        
        # ✅ FIX #4: Prevent infinite loop
        if future_a_pos >= track_a_dur - 1.0:
            return None  # Invalid: would overshoot EOF
        
        future_a_pos = min(future_a_pos, track_a_dur - 1.0)
        
        # Key compatibility
        kc = self.brain.key_compat(current_track.get('key', 'C'), tb.get('key', 'C'))
        
        # ✅ FIX #2: Safe BPM ratio calculation
        effective_master_bpm = self.dj._master_bpm_precise if self.dj else master_bpm
        tb_bpm = max(MIN_SAFE_BPM, tb.get('bpm', effective_master_bpm))
        raw_b_ratio = effective_master_bpm / tb_bpm
        b_ratio = self._safe_bpm_ratio(effective_master_bpm, tb_bpm)
        
        post_stretch_bpm = tb_bpm * b_ratio
        effective_bpm_diff = abs(post_stretch_bpm - effective_master_bpm)
        
        is_amp = (current_track.get('genre') == 'amapiano' or
                 tb.get('genre') == 'amapiano')
        
        # Global compatibility
        global_compat_raw = (
            self.brain.bpm_score(current_track.get('bpm', 112), tb.get('bpm', 112), is_amp) +
            self.brain.camelot_score(current_track.get('key', 'C'), tb.get('key', 'C'))[0] +
            self.brain.vocal_score(current_track, tb) +
            self.brain.spectral_score(current_track, tb) +
            self.brain.genre_score(current_track, tb)
        )
        global_compat = max(0.0, min(1.0, (global_compat_raw + 100) / 150.0))
        
        # Intro length
        intro_beats = 30.0 / spb
        for sec in tb.get('structure_map', []):
            if sec.get('label') == 'intro':
                intro_beats = sec.get('end', 30.0) / spb
                break
        
        # Search for best overlap
        max_search_dur = track_a_dur - future_a_pos - 1.0
        search_dur = min(80.0 * spb, max(16.0 * spb, max_search_dur))
        
        try:
            best_a_exit, best_b_entry, overlap_score, all_candidates = \
                self.searcher.search(current_track, tb, b_ratio, search_dur, spb, quiet=True)
        except Exception as e:
            best_a_exit = track_a_dur - 15.0
            best_b_entry = 0.0
            overlap_score = {'total': 0.1, 'harmonic': 0.5, 'vocal': 0.5}
            all_candidates = []
        
        # Ensure exit is after future position
        if best_a_exit < future_a_pos - (8.0 * spb):
            best_a_exit = future_a_pos + (16.0 * spb)
        
        # Calculate runway
        runway_seconds = track_a_dur - best_a_exit - 1.0
        runway_beats = max(0, int(runway_seconds / spb))
        
        # Select technique with runway constraint
        runway_penalty = 0.0
        if runway_beats < 16:
            tech_name = "ECHO_THROW"
            runway_penalty = 0.30
            runway_beats = 16
        else:
            rough_ctx = self.build_transition_context(
                current_track, tb, spb, mix_out_time=best_a_exit,
                b_entry_time=best_b_entry, trans_dur=search_dur, b_ratio=b_ratio
            )
            
            tech_name = self.select_technique(
                current_track.get('energy', 'High'), tb.get('energy', 'High'),
                effective_bpm_diff, kc, is_amp, intro_beats, mix_count,
                ctx=rough_ctx,
                set_snapshot=(self.set_state_model.get_snapshot()
                            if self.set_state_model else None),
                runway_beats=runway_beats,
                quiet=True
            )
        
        # Generate recipe
        mem_key, recipe = self.brain.ml.generate_recipe(
            tech_name, current_track.get('energy', 'High'),
            tb.get('energy', 'High'), is_amp, quiet=True
        )
        
        # Fit technique to runway
        tech_min = float(TECHNIQUE_MIN_BEATS.get(tech_name, 32.0))
        tech_ideal = float(TECHNIQUE_IDEAL_BEATS.get(tech_name, 64.0))
        
        if runway_beats >= tech_ideal:
            actual_beats = tech_ideal
        elif runway_beats >= tech_min:
            actual_beats = float(runway_beats)
        else:
            actual_beats = max(16.0, float(runway_beats))
        
        actual_beats = max(16.0, 8.0 * round(actual_beats / 8.0))
        recipe['beats'] = actual_beats
        trans_dur_est = actual_beats * spb
        
        # Mixability scores
        a_phrases = (current_track.get('phrases', []) or
                    current_track.get('phrase_map', []) or
                    current_track.get('phrase_analysis', []) or
                    current_track.get('phrase_windows', []))
        
        a_exit_mix = 0.5
        if a_phrases:
            # ✅ FIX #2: Safe min() with empty check
            try:
                effective_ratio = current_track.get('_stretch_ratio', 1.0)
                closest_phrase = min(
                    a_phrases,
                    key=lambda p: abs(float(p.get('start', 0.0)) / effective_ratio - best_a_exit)
                )
                if abs(float(closest_phrase.get('start', 0.0)) / effective_ratio - best_a_exit) < 30.0:
                    a_exit_mix = float(closest_phrase.get('mixability', 0.5))
            except (ValueError, TypeError, KeyError):
                pass
        
        b_phrases = (tb.get('phrases', []) or tb.get('phrase_map', []) or
                    tb.get('phrase_analysis', []) or tb.get('phrase_windows', []))
        
        b_entry_mix = 0.5
        if b_phrases:
            try:
                closest_phrase = min(
                    b_phrases,
                    key=lambda p: abs(float(p.get('start', 0.0)) - (best_b_entry * b_ratio))
                )
                b_entry_mix = float(closest_phrase.get('mixability', 0.5))
            except (ValueError, TypeError, KeyError):
                pass
        
        # Robustness
        robustness = min(1.0, len([
            c for c in all_candidates
            if float(c['score'].get('total', 0.0)) >= 0.65
        ]) / 4.0)
        
        # Transition opportunity
        transition_opportunity = (
            0.45 * float(overlap_score.get('total', 0.0)) +
            0.25 * a_exit_mix +
            0.20 * b_entry_mix +
            0.10 * robustness
        )
        
        # Intent fit
        intent_fit = 0.0
        reasons = []
        eb = tb.get('energy', 'High')
        
        if intent.goal == "recovery":
            if float(overlap_score.get('total', 0.0)) > 0.8:
                intent_fit += 0.30
                reasons.append("highly_safe_overlap")
            if tech_name in ("SLOW_BURN", "FILTER_SWEEP"):
                intent_fit += 0.20
                reasons.append("safe_technique_for_recovery")
        
        elif intent.goal == "vocal_rest":
            if not tb.get('stems', {}).get('has_vocals', False):
                intent_fit += 0.40
                reasons.append("vocal_rest_achieved")
        
        elif intent.goal == "tension_build":
            if tech_name in ("FILTER_SWEEP", "BASS_SWAP"):
                intent_fit += 0.20
                reasons.append("builds_tension")
        
        elif intent.goal == "closing_sequence":
            if eb in ("Low/Chill",):
                intent_fit += 0.30
                reasons.append("winding_down_energy")
            if tech_name in ("SLOW_BURN", "ECHO_THROW"):
                intent_fit += 0.20
                reasons.append("closing_technique")
        
        elif intent.goal == "controlled_rebuild":
            if eb in ("Low/Chill", "Medium", "High"):
                intent_fit += 0.20
                reasons.append("supports_controlled_rebuild")
            if tech_name in ("FILTER_SWEEP", "SLOW_BURN"):
                intent_fit += 0.20
                reasons.append("technique_matches_controlled_rebuild")
        
        elif intent.goal == "peak_lift":
            if eb == "High":
                intent_fit += 0.25
                reasons.append("supports_peak_lift")
            if tech_name in IMMEDIATE_TECHNIQUES.union({"BASS_SWAP", "PIANO_HANDOFF", "ACAPELLA_MASHUP"}):
                intent_fit += 0.20
                reasons.append("technique_matches_peak_lift")
        
        elif intent.goal == "hold_groove":
            intent_fit += 0.15
            reasons.append("supports_hold_groove")
            if tech_name in ("BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"):
                intent_fit += 0.15
                reasons.append("stable_transition_family")
        
        elif intent.goal == "harmonic_refresh":
            if kc in ("exact", "compatible"):
                intent_fit += 0.25
                reasons.append("harmonic_compatibility_for_refresh")
            if tech_name in ("FILTER_SWEEP", "SLOW_BURN", "BASS_SWAP"):
                intent_fit += 0.15
                reasons.append("smooth_technique_for_refresh")
        
        elif intent.goal == "balanced_progression":
            intent_fit += 0.15
            reasons.append("balanced_progression_baseline")
            if tech_name in ("BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"):
                intent_fit += 0.10
                reasons.append("versatile_technique")
        
        # Vocal tolerance
        if intent.vocal_tolerance == "low":
            if not (current_track.get('stems', {}).get('has_vocals', False) and
                   tb.get('stems', {}).get('has_vocals', False)):
                intent_fit += 0.20
                reasons.append("low_vocal_overlap_risk")
        
        intent_fit = max(0.0, min(1.0,
            intent_fit + (0.20 * a_exit_mix) + (0.10 * b_entry_mix)
        ))
        
        # Execution risk
        risk = 0.0
        if robustness < 0.35:
            risk += 0.25
            reasons.append("few_good_candidate_pairs")
        if float(overlap_score.get('vocal', 0.5)) < 0.65:
            risk += 0.20
            reasons.append("vocal_overlap_risk")
        if float(overlap_score.get('harmonic', 0.5)) < 0.50:
            risk += 0.18
            reasons.append("harmonic_risk")
        if track_a_dur - best_a_exit < trans_dur_est + 20.0:
            risk += 0.20
            reasons.append("tight_exit_room")
        if b_entry_mix < 0.55:
            risk += 0.12
            reasons.append("dense_b_entry")
        
        # Early commitment penalty
        total_wait_time = best_a_exit - current_pos_a
        far_threshold = 128.0 * spb
        early_penalty = 0.0
        
        if total_wait_time > far_threshold:
            overshoot = (total_wait_time - far_threshold) / far_threshold
            early_penalty = min(0.40, 0.15 + 0.15 * overshoot)
            reasons.append(f"committing_too_early_penalty_(-{early_penalty:.2f})")
        
        execution_risk = max(0.0, min(1.0, risk))
        
        # Future flexibility
        future_flex = max(0.0, min(1.0,
            0.4 + min(len(tb.get('best_exit_phrases', [])), 4) * 0.1 -
            (0.1 if tb.get('stems', {}).get('has_vocals', False) else 0.0)
        ))
        
        reasons.extend([
            f"best_exit={best_a_exit:.1f}s",
            f"best_entry={best_b_entry:.1f}s",
            f"technique={tech_name}",
            f"beats={actual_beats:.0f}",
            f"runway={runway_beats}beats",
            f"overlap={overlap_score.get('total', 0.0):.2f}",
        ])
        
        assumptions = {
            "a_decay_expected": True,
            "b_groove_should_establish_by_20_beats": True,
            "vocal_overlap_tolerance": intent.vocal_tolerance,
            "a_should_not_resurge_after_progress": 0.50,
            "max_expected_execution_risk": intent.max_execution_risk,
        }
        
        # Final score
        final_score = (
            0.30 * intent_fit +
            0.30 * transition_opportunity +
            0.15 * global_compat +
            0.10 * robustness +
            0.10 * future_flex -
            0.20 * execution_risk -
            early_penalty -
            runway_penalty
        )
        
        return EvaluatedOption(
            option=option,
            final_score=float(final_score),
            intent_fit=float(intent_fit),
            transition_opportunity=float(transition_opportunity),
            global_compat=float(global_compat),
            robustness=float(robustness),
            execution_risk=float(execution_risk),
            future_flex=float(future_flex),
            reasoning=reasons,
            assumptions=assumptions,
            draft_plan={
                "track_b": tb,
                "b_ratio": b_ratio,
                "delay_beats": option.delay_beats,
                "mix_trigger": best_a_exit,
                "b_start_w": best_b_entry,
                "tech_name": tech_name,
                "mem_key": mem_key,
                "recipe": recipe,
                "overlap_score": overlap_score,
                "trans_dur_est": trans_dur_est,
                "runway_beats": runway_beats,
            }
        )
    
    def _evaluate_hold_option(self, option: CandidateOption, intent: Intent,
                             current_track: dict, current_pos_a: float,
                             track_a_dur: float, spb: float) -> EvaluatedOption:
        """Evaluate a hold option."""
        future_pos = current_pos_a + option.delay_beats * spb
        future_pos_ratio = future_pos / max(track_a_dur, 1.0)
        hold_value, reasons = 0.0, []
        
        # Position-based value
        if future_pos_ratio < 0.50:
            hold_value += 0.45
            reasons.append("early_track_strong_hold")
        elif future_pos_ratio < 0.60:
            hold_value += 0.30
            reasons.append("mid_track_hold")
        elif future_pos_ratio < 0.65:
            hold_value += 0.15
            reasons.append("approaching_decision_zone")
        elif future_pos_ratio < 0.70:
            hold_value += 0.00
            reasons.append("decision_zone_neutral")
        else:
            overshoot = (future_pos_ratio - 0.70) / 0.30
            penalty = 0.50 * overshoot
            hold_value -= penalty
            reasons.append(f"LATE_hold_penalty_(-{penalty:.2f})")
        
        # Intent bonuses
        if intent.goal in ("hold_groove", "controlled_rebuild", "tension_build", "recovery"):
            intent_bonus = 0.25 * max(0.0, 1.0 - future_pos_ratio)
            hold_value += intent_bonus
            reasons.append(f"intent_{intent.goal}_scaled_{intent_bonus:.2f}")
        
        # Delay bonus
        if option.delay_beats == 16:
            hold_value += 0.05
        elif option.delay_beats == 32:
            hold_value += 0.02
        
        execution_risk = 0.20 if option.delay_beats == 16 else 0.32
        track_momentum = 0.8 if intent.goal in ("peak_lift", "tension_build") else 0.5
        
        final_score = (
            0.55 * hold_value +
            0.20 * (1.0 - execution_risk) +
            0.15 * 0.30 +
            0.10 * track_momentum
        )
        
        return EvaluatedOption(
            option=option,
            final_score=float(final_score),
            intent_fit=float(hold_value),
            transition_opportunity=0.0,
            global_compat=0.0,
            robustness=0.0,
            execution_risk=float(execution_risk),
            future_flex=0.30,
            reasoning=reasons + [f"future_pos={future_pos:.1f}s ratio={future_pos_ratio:.2f}"],
            assumptions={"a_should_continue_delivering_value": True},
            draft_plan=None
        )