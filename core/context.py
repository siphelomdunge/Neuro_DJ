"""
Transition context builder — analyzes phrase-level characteristics.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from .constants import CAMELOT_WHEEL


def build_transition_context(ta: dict, tb: dict, spb: float,
                             mix_out_time: Optional[float] = None,
                             b_entry_time: Optional[float] = None,
                             trans_dur: float = 60.0,
                             b_ratio: float = 1.0) -> Dict:
    """
    Build comprehensive transition context from phrase analysis.
    
    Args:
        ta: Track A metadata
        tb: Track B metadata
        spb: Seconds per beat
        mix_out_time: Exit point on track A
        b_entry_time: Entry point on track B
        trans_dur: Transition duration
        b_ratio: Time-stretch ratio for track B
    
    Returns:
        Context dict with dance_moment, density scores, etc.
    """
    if mix_out_time is None:
        mix_out_time = ta.get('zones', {}).get('optimal_mix_out', 0.0)
    
    # Analyze exit phrase on track A
    a_structure = ta.get('structure_map', [])
    a_exit_section = 'outro'
    a_exit_density = 0.3
    a_exit_energy = 0.5
    
    effective_ratio = ta.get('_stretch_ratio', 1.0)
    for s in a_structure:
        adj_start = s.get('start', 0) / effective_ratio
        adj_end = s.get('end', 9999) / effective_ratio
        if adj_start <= mix_out_time <= adj_end:
            a_exit_section = s.get('label', 'outro')
            a_exit_density = s.get('texture_density', 0.3)
            a_exit_energy = s.get('energy_level', 0.5)
            break
    
    # Get exit phrase details
    phrases = (ta.get('phrases', []) or ta.get('phrase_map', []) or
              ta.get('phrase_analysis', []) or ta.get('phrase_windows', []))
    
    a_exit_phrase = None
    a_exit_phrase_func = None
    
    if phrases:
        # ✅ FIX #2: Safe min() with proper error handling
        try:
            closest_phrase = min(
                phrases,
                key=lambda p: abs(float(p.get('start', 0.0)) / effective_ratio - mix_out_time)
            )
            if abs(float(closest_phrase.get('start', 0.0)) / effective_ratio - mix_out_time) < 30.0:
                a_exit_phrase = closest_phrase
                a_exit_phrase_func = a_exit_phrase.get('function', 'unknown')
            else:
                a_exit_phrase_func = 'unknown'
        except (ValueError, TypeError, KeyError):
            a_exit_phrase_func = 'unknown'
    else:
        a_exit_phrase_func = 'unknown'
    
    # Blend phrase-level density with structure density
    if a_exit_phrase:
        a_exit_density = 0.5 * a_exit_density + 0.5 * (
            0.35 * float(a_exit_phrase.get('bass_density', 0.0)) +
            0.25 * float(a_exit_phrase.get('vocal_density', 0.0)) +
            0.20 * float(a_exit_phrase.get('harmonic_density', 0.0)) +
            0.20 * float(a_exit_phrase.get('percussive_density', 0.0))
        )
        a_exit_mixability = float(a_exit_phrase.get('mixability', 0.5))
        a_exit_bass = float(a_exit_phrase.get('bass_density', 0.0))
        a_exit_vocal = float(a_exit_phrase.get('vocal_density', 0.0))
        a_exit_tension = float(a_exit_phrase.get('tension_score', 0.0))
    else:
        a_exit_mixability = 0.5
        a_exit_bass = a_exit_vocal = a_exit_tension = 0.0
    
    # Texture profile refinement
    tp_a = ta.get('texture_profile', [])
    if tp_a:
        closest = min(tp_a, key=lambda x: abs(x.get('time', 0.0) / effective_ratio - mix_out_time))
        a_exit_density = 0.5 * a_exit_density + 0.5 * closest.get('density', 0.3)
    
    # Analyze entry section on track B
    b_structure = tb.get('structure_map', [])
    b_entry_section = 'intro'
    
    if b_structure:
        if b_entry_time is not None:
            for s in b_structure:
                if (s.get('start', 0) / b_ratio) <= b_entry_time <= (s.get('end', 9999) / b_ratio):
                    b_entry_section = s.get('label', 'intro')
                    break
        else:
            b_entry_section = b_structure[0].get('label', 'intro')
    
    b_mix_character = tb.get('mix_character', 'clean_intro')
    a_energy_trajectory = ta.get('energy_trajectory', 'falling')
    
    # Vocal clash detection
    a_vocals = ta.get('stems', {}).get('vocal_regions', [])
    b_vocals = tb.get('stems', {}).get('vocal_regions', [])
    
    physical_trans_dur = trans_dur / max(b_ratio, 0.01)
    trans_window = max(30.0, physical_trans_dur)
    
    a_exit_has_vocal = any(
        start / effective_ratio <= mix_out_time + trans_window and
        end / effective_ratio >= mix_out_time
        for start, end in a_vocals
    )
    
    if b_entry_time is None:
        b_entry_time = 0.0
    
    scaled_vocal_b = ([(s / b_ratio, e / b_ratio) for s, e in b_vocals]
                     if abs(b_ratio - 1.0) > 0.005 else b_vocals)
    b_entry_has_vocal = any(
        start <= b_entry_time + trans_window and end > b_entry_time
        for start, end in scaled_vocal_b
    )
    
    # Override vocal detection with phrase-level data
    if a_exit_phrase and float(a_exit_phrase.get('vocal_density', 0.0)) > 0.50:
        a_exit_has_vocal = True
    
    vocal_clash = a_exit_has_vocal and b_entry_has_vocal
    
    # Determine dance moment
    energy_a = ta.get('energy', 'High')
    energy_b = tb.get('energy', 'High')
    
    if a_energy_trajectory == 'floor' and a_exit_density < 0.25:
        dance_moment = 'hard_reset' if energy_b == 'High' else 'cool_down'
    elif a_exit_phrase_func == 'dj_friendly_outro' and a_exit_density < 0.35:
        dance_moment = 'controlled_rebuild' if energy_b == 'High' else 'cool_down'
    elif a_exit_section == 'breakdown' and a_exit_density < 0.45:
        dance_moment = 'build_release' if energy_b == 'High' else 'breather'
    elif a_exit_section == 'drop' and a_energy_trajectory == 'peak':
        dance_moment = 'peak_swap' if energy_b == 'High' else 'cool_down'
    elif a_exit_phrase_func in ('harmonic_bed', 'decompression') and a_exit_density < 0.5:
        dance_moment = 'harmonic_lift' if energy_b == 'High' else 'melodic_reset'
    elif a_exit_phrase_func == 'drum_foundation' and a_exit_density < 0.55:
        dance_moment = 'groove_extension'
    elif a_exit_phrase_func == 'vocal_spotlight' or vocal_clash:
        dance_moment = 'vocal_relief'
    elif a_energy_trajectory == 'rising' and a_exit_density > 0.6:
        dance_moment = 'pre_peak_build'
    elif a_energy_trajectory == 'falling' and energy_b == 'High':
        dance_moment = 'natural_exit'
    elif energy_a == 'Low/Chill' and energy_b == 'High':
        dance_moment = 'reboot'
    elif energy_a == 'High' and energy_b == 'Low/Chill':
        dance_moment = 'cool_down'
    else:
        dance_moment = 'neutral'
    
    return {
        'a_exit_section': a_exit_section,
        'a_exit_phrase_func': a_exit_phrase_func,
        'a_exit_density': float(a_exit_density),
        'a_exit_energy': a_exit_energy,
        'a_exit_mixability': a_exit_mixability,
        'a_exit_bass': a_exit_bass,
        'a_exit_vocal': a_exit_vocal,
        'a_exit_tension': a_exit_tension,
        'a_energy_trajectory': a_energy_trajectory,
        'b_entry_section': b_entry_section,
        'b_mix_character': b_mix_character,
        'vocal_clash': vocal_clash,
        'dance_moment': dance_moment,
    }