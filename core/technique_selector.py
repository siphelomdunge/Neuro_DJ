"""
Technique selection with dance moment scoring.
"""
from __future__ import annotations
from typing import Optional, Dict

from .constants import (
    TECHNIQUE_LIBRARY, TECHNIQUE_MIN_BEATS, TECHNIQUE_IDEAL_BEATS,
    DANCE_MOMENT_SCORES, VALID_ENERGIES
)


def select_technique(energy_a: str, energy_b: str, bpm_diff: float,
                    key_compat: str, is_amapiano: bool, intro_beats: float,
                    mix_count: int, ctx: Optional[Dict] = None,
                    set_snapshot=None, runway_beats: float = 9999,
                    quiet: bool = False) -> str:
    """
    Select optimal technique based on context and constraints.
    
    Args:
        energy_a: Energy level of track A
        energy_b: Energy level of track B
        bpm_diff: BPM difference
        key_compat: Key compatibility ('exact', 'compatible', 'clash')
        is_amapiano: Whether either track is amapiano
        intro_beats: Length of track B intro
        mix_count: Number of previous mixes
        ctx: Transition context (optional)
        set_snapshot: Set state snapshot (optional)
        runway_beats: Available runway in beats
        quiet: Suppress logging
    
    Returns:
        Technique name
    """
    # Validate energy levels
    if energy_a not in VALID_ENERGIES:
        print(f"⚠️ Invalid energy_a: {energy_a}, defaulting to 'High'")
        energy_a = "High"
    if energy_b not in VALID_ENERGIES:
        print(f"⚠️ Invalid energy_b: {energy_b}, defaulting to 'High'")
        energy_b = "High"
    
    scores = {}
    
    for name, tech in TECHNIQUE_LIBRARY.items():
        s = 0
        
        tech_min = TECHNIQUE_MIN_BEATS.get(name, 32)
        tech_ideal = TECHNIQUE_IDEAL_BEATS.get(name, 64)
        
        # ✅ FIX #6: Hard veto with logging
        if runway_beats < tech_min:
            if not quiet:
                print(f"   ⛔ {name} vetoed: needs {tech_min} beats, only {runway_beats} available")
            scores[name] = -9999  # Hard veto
            continue
        
        # Runway scoring
        if runway_beats >= tech_ideal:
            s += 15
        elif runway_beats >= tech_min:
            s += 5
        
        # Context-based scoring
        if ctx is not None:
            dm = ctx['dance_moment']
            density = ctx['a_exit_density']
            traj = ctx['a_energy_trajectory']
            b_char = ctx['b_mix_character']
            v_clash = ctx['vocal_clash']
            
            # Apply dance moment scores
            if dm in DANCE_MOMENT_SCORES:
                s += DANCE_MOMENT_SCORES[dm].get(name, 0)
            
            # Density adjustments
            if density < 0.35:
                s += 10
                if name == "SLOW_BURN":
                    s += 10
            elif density > 0.65:
                if name == "BASS_SWAP":
                    s += 20
                elif name in ("SLOW_BURN", "FILTER_SWEEP"):
                    s -= 15
                elif name in ("ECHO_FREEZE", "ECHO_THROW"):
                    s += 10
            
            # Mix character
            if b_char == 'clean_intro':
                if name == "SLOW_BURN":
                    s += 20
                elif name == "BASS_SWAP":
                    s += 10
            elif b_char == 'drum_heavy':
                if name == "BASS_SWAP":
                    s += 30
                elif name == "PIANO_HANDOFF":
                    s += 20 if is_amapiano else -10
                elif name == "ECHO_FREEZE":
                    s += 10
            elif b_char == 'vocal_heavy':
                if name in ("ECHO_FREEZE", "ECHO_THROW"):
                    s += 30
                elif name == "SLOW_BURN":
                    s -= 10
            elif b_char == 'melodic':
                if name == "PIANO_HANDOFF":
                    s += 30 if is_amapiano else 0
                elif name == "FILTER_SWEEP":
                    s += 20
                elif name == "BASS_SWAP":
                    s += 10
            
            # Vocal clash
            if v_clash:
                if name in ("ECHO_FREEZE", "ECHO_THROW"):
                    s += 40
                elif name == "ACAPELLA_MASHUP":
                    s += 50
                elif name == "BASS_SWAP":
                    s -= 30
                elif name == "SLOW_BURN":
                    s -= 20
            
            # Trajectory
            if traj == 'floor':
                if name in ("ECHO_FREEZE", "ECHO_THROW"):
                    s += 20
            elif traj == 'falling':
                if name == "FILTER_SWEEP":
                    s += 15
                elif name == "SLOW_BURN":
                    s += 10
            
            # Acapella-specific logic
            if name == "ACAPELLA_MASHUP":
                a_vocal = ctx.get('a_exit_vocal', 0)
                
                # ✅ FIX #14: Gradual penalty instead of harsh -200
                if a_vocal < 0.25:
                    s -= 100 * (0.25 - a_vocal)  # Scales from 0 to -25
                elif a_vocal > 0.50:
                    s += 30
                
                if b_char == 'drum_heavy':
                    s += 30
                elif b_char == 'clean_intro':
                    s += 15
                elif b_char == 'vocal_heavy':
                    s -= 50
                
                if key_compat == "exact":
                    s += 20
                elif key_compat == "compatible":
                    s += 10
                elif key_compat == "clash":
                    s -= 40
                
                if bpm_diff > 3.0:
                    s -= 100
                elif bpm_diff <= 1.0:
                    s += 15
        
        # BPM compatibility
        if name == "SLOW_BURN":
            s += 12
        elif name in ("ECHO_FREEZE", "ECHO_THROW"):
            s += 20 if bpm_diff > 5.0 else 8 if bpm_diff > 2.0 else 0
        else:
            s += 20 if bpm_diff <= 1.0 else 8 if bpm_diff <= 3.0 else -15
        
        # Key compatibility
        if key_compat == "exact":
            if name == "BASS_SWAP":
                s += 25
            elif name == "PIANO_HANDOFF":
                s += 20
            else:
                s += 10
        elif key_compat == "compatible":
            s += 15 if name == "BASS_SWAP" else 5
        else:  # clash
            if name in ("ECHO_FREEZE", "ECHO_THROW"):
                s += 25
            if name == "SLOW_BURN":
                s += 15
            if name == "BASS_SWAP":
                s -= 25
            if name == "PIANO_HANDOFF":
                s -= 30
        
        # Amapiano bonus
        if is_amapiano:
            if name == "PIANO_HANDOFF":
                s += 90
            elif name == "BASS_SWAP":
                s += 50
            elif name == "ACAPELLA_MASHUP":
                s += 70
            elif name in ("ECHO_FREEZE", "ECHO_THROW"):
                s -= 60
            elif name == "FILTER_SWEEP":
                s -= 40
            elif name == "SLOW_BURN":
                s -= 15
        
        # Fallback energy scoring (if no context)
        if ctx is None:
            if energy_a == "High" and energy_b == "High":
                if name == "BASS_SWAP":
                    s += 40
                elif name == "SLOW_BURN":
                    s -= 30
            elif energy_a == "High" and energy_b == "Low/Chill":
                if name in ("FILTER_SWEEP", "SLOW_BURN"):
                    s += 40
            elif energy_a == "Low/Chill" and energy_b == "High":
                if name in ("ECHO_FREEZE", "ECHO_THROW"):
                    s += 40
                elif name == "BASS_SWAP":
                    s += 25
        
        # Intro length check
        if name != "ECHO_FREEZE":
            min_beats_needed = TECHNIQUE_MIN_BEATS.get(name, 64) * 0.65
            if name in ("BASS_SWAP", "PIANO_HANDOFF", "FILTER_SWEEP"):
                if intro_beats < min_beats_needed:
                    s -= 60
        
        # Anti-repetition
        if name in ("ECHO_FREEZE", "ECHO_THROW"):
            s -= 15
        variety_idx = list(TECHNIQUE_LIBRARY.keys()).index(name)
        if variety_idx == (mix_count % len(TECHNIQUE_LIBRARY)):
            s -= 12
        
        scores[name] = s
    
    # Apply set state scoring if available
    if set_snapshot is not None:
        try:
            from set_state import SetStateModel
            _tmp = SetStateModel.__new__(SetStateModel)
            _tmp.events = []
            scores = _tmp.apply_to_scoring(scores, set_snapshot)
        except ImportError:
            pass
    
    best = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    
    if not quiet:
        print(f"   🎯 Technique scores: " + " | ".join(f"{n}={v}" for n, v in ranked))
        if runway_beats < 9999:
            print(f"   📏 Runway: {runway_beats} beats — "
                  f"vetoed: {sum(1 for v in scores.values() if v <= -400)}")
        if set_snapshot:
            snap = set_snapshot
            rb = snap.time_since_breather > 120 and snap.energy_rolling > 0.7
            ri = snap.time_since_hard_drop > 300 and snap.energy_rolling < 0.5 and snap.mix_count > 3
            print(f"   🎪 Set phase: {snap.set_phase}  energy={snap.energy_rolling:.2f}"
                  f"  {'⚠️ BREATHER ' if rb else ''}{'💥 INTENSITY ' if ri else ''}")
    
    return best