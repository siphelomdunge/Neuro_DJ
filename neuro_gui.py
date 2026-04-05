"""
neuro_gui.py  —  Neuro-DJ Master Engine (Dynamic Technique Intelligence)

FULLY AUDITED VERSION (V29):
  • C++ dr_wav Absolute Path & Float compatibility.
  • PREDICTIVE PHRASE SCORING: AI looks ahead to avoid "fake outros".
  • TWO-STAGE CONTEXT: Technique selection reacts to ACTUAL phrase exit.
  • SET ARC RESTORED: Energy history actively tracked to prevent crowd fatigue.
  • WORKSTREAM 1: CrateRanker natively integrated for intelligent candidate funnels.
  • WORKSTREAM 6: PersistentLearner V2 wired up for rapid human-feedback adaptation.
  • WORKSTREAM 5: Expanded Intent Vocabulary (Recovery, Vocal Rest, Tension Build).
  • V23 PSYCHOLOGY EXPANSION: 12-State Dance Moment Intents, Hard-Veto Pre-Filtering.
  • V24 MATHEMATICAL SYNC: Fixed lethal trans_dur vs beats desync, introduced IDEAL_BEATS.
  • V25 PHRASE RE-OPTIMIZATION: Global safety constants, dynamic phrase re-searching.
  • V26 ARCHITECTURAL HARDENING: Thread-safe dataset I/O, Pause-aware thread sleeps.
  • V27 BATTLE-TESTED FIXES: Negative sleep guard, post-overrun buffer guard, 80-beat search window.
  • V28 & V29 UI LOCK UPDATE: Fixed NameError crash in submit_rating; fully implemented GUI Feedback Expiry Timer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import copy
import uuid

import neuro_core
import time, json, librosa, soundfile as sf, numpy as np
import os, sys, threading, random, warnings, concurrent.futures

# ── Import the new Sieve (Hard Dependency) ──
from crate_ranker import CrateRanker

# Stage 6: Live perception
try:
    from live_ears import LiveEars
    LIVE_EARS_AVAILABLE = True
except ImportError:
    LIVE_EARS_AVAILABLE = False
    print("⚠️  live_ears.py not found — live perception disabled")

# Stage 7+8: Adaptive executor
try:
    from adaptive_executor import AdaptiveExecutor
    ADAPTIVE_EXECUTOR_AVAILABLE = True
except ImportError:
    ADAPTIVE_EXECUTOR_AVAILABLE = False
    print("⚠️  adaptive_executor.py not found — bounded adaptation disabled")

# Stage 10: Set state model
try:
    from set_state import SetStateModel
    SET_STATE_AVAILABLE = True
except ImportError:
    SET_STATE_AVAILABLE = False
    print("⚠️  set_state.py not found — set state tracking disabled")
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QProgressBar, QTextEdit,
                             QFrame, QPushButton, QSlider, QCheckBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QTextCursor

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS & TECHNIQUE LIBRARY
# ════════════════════════════════════════════════════════════════════════════════
MIN_EXIT_POSITION_RATIO = 0.70  
MAX_WAIT_SECONDS = 120.0
DEFAULT_TRACK_DURATION = 300.0

SEARCH_SAFETY_BUFFER_SEC = 10.0
FINALIZE_SAFETY_BUFFER_SEC = 10.0
EXECUTION_RESERVE_SEC = 2.0

def clamp01(v): return max(0.0, min(1.0, v))

CAMELOT_WHEEL = {
    'C':(8,'B'), 'G':(9,'B'), 'D':(10,'B'), 'A':(11,'B'), 'E':(12,'B'),'B':(1,'B'), 'F#':(2,'B'), 'C#':(3,'B'),
    'G#':(4,'B'), 'D#':(5,'B'), 'A#':(6,'B'), 'F':(7,'B'), 'Am':(8,'A'), 'Em':(9,'A'), 'Bm':(10,'A'), 'F#m':(11,'A'),
    'C#m':(12,'A'),'G#m':(1,'A'), 'D#m':(2,'A'), 'A#m':(3,'A'), 'Fm':(4,'A'), 'Cm':(5,'A'), 'Gm':(6,'A'), 'Dm':(7,'A'),
}

TECHNIQUE_MIN_BEATS = {
    "BASS_SWAP": 64.0, "FILTER_SWEEP": 48.0, "ECHO_THROW": 32.0,
    "ECHO_FREEZE": 32.0, "SLOW_BURN": 64.0, "PIANO_HANDOFF": 64.0,
}

TECHNIQUE_SAFE_MIN_BEATS = {
    "PIANO_HANDOFF": 64.0,
    "SLOW_BURN": 64.0,
    "BASS_SWAP": 64.0,
    "FILTER_SWEEP": 48.0,
    "ECHO_THROW": 16.0,
    "ECHO_FREEZE": 16.0,
}

TECHNIQUE_IDEAL_BEATS = {
    "BASS_SWAP":      96.0,
    "FILTER_SWEEP":   80.0,
    "ECHO_THROW":     32.0,
    "ECHO_FREEZE":    32.0,
    "SLOW_BURN":      96.0,
    "PIANO_HANDOFF":  80.0,
}

TECHNIQUE_LIBRARY = {
    "BASS_SWAP": {
        "id": 0, "label": "🔊 Bass Swap",
        "defaults": {"beats": 96.0, "bass": 0.75, "echo": 0.0, "wash": 0.2, "piano_hold": 0.0}
    },
    "FILTER_SWEEP": {
        "id": 2, "label": "🌊 Filter Sweep",
        "defaults": {"beats": 96.0, "bass": 0.55, "echo": 0.0, "wash": 1.0, "piano_hold": 0.0}
    },
    "ECHO_THROW": {
        "id": 3, "label": "🎯 Echo Out",
        "defaults": {"beats": 32.0, "bass": 0.75, "echo": 1.0, "wash": 0.0, "piano_hold": 0.0}
    },
    "ECHO_FREEZE": {
        "id": 7, "label": "🧊 Echo Freeze",
        "defaults": {"beats": 32.0, "bass": 0.90, "echo": 1.0, "wash": 0.0, "piano_hold": 0.0}
    },
    "SLOW_BURN": {
        "id": 4, "label": "🕯️ Slow Burn",
        "defaults": {"beats": 128.0, "bass": 0.5, "echo": 0.0, "wash": 0.0, "piano_hold": 0.0}
    },
    "PIANO_HANDOFF": {
        "id": 5, "label": "🎹 Piano Handoff",
        "defaults": {"beats": 128.0, "bass": 0.50, "echo": 0.0, "wash": 0.0, "piano_hold": 1.0}
    },
}

def build_transition_context(ta: dict, tb: dict, spb: float, mix_out_time: Optional[float]=None, b_entry_time: Optional[float]=None, trans_dur: float=60.0, b_ratio: float=1.0) -> dict:
    if mix_out_time is None:
        mix_out_time = ta.get('zones', {}).get('optimal_mix_out', 0.0)

    a_structure = ta.get('structure_map', [])
    a_exit_section = 'outro'
    a_exit_density = 0.3
    a_exit_energy = 0.5

    for s in a_structure:
        if s.get('start', 0) <= mix_out_time <= s.get('end', 9999):
            a_exit_section = s.get('label', 'outro')
            a_exit_density = s.get('texture_density', 0.3)
            a_exit_energy  = s.get('energy_level', 0.5)
            break

    phrases = (ta.get('phrases', []) or ta.get('phrase_map', []) or 
               ta.get('phrase_analysis', []) or ta.get('phrase_windows', []))
    
    a_exit_phrase = None
    a_exit_phrase_func = None
    if phrases:
        try:
            a_exit_phrase = min(phrases, key=lambda p: abs(float(p.get('start', 0.0)) - mix_out_time))
            if abs(float(a_exit_phrase.get('start', 0.0)) - mix_out_time) < 30.0:
                a_exit_phrase_func = a_exit_phrase.get('function')
        except Exception:
            a_exit_phrase = None

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
        a_exit_bass = 0.0
        a_exit_vocal = 0.0
        a_exit_tension = 0.0

    tp_a = ta.get('texture_profile', [])
    if tp_a:
        closest = min(tp_a, key=lambda x: abs(x.get('time', 0.0) - mix_out_time))
        a_exit_density = 0.5 * a_exit_density + 0.5 * closest.get('density', 0.3)

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

    a_vocals = ta.get('stems', {}).get('vocal_regions', [])
    b_vocals = tb.get('stems', {}).get('vocal_regions', [])

    trans_window = max(30.0, trans_dur)
    a_exit_has_vocal = any(start <= mix_out_time + trans_window and end >= mix_out_time for start, end in a_vocals)

    if b_entry_time is None: b_entry_time = 0.0
    
    scaled_vocal_b = ([(s/b_ratio, e/b_ratio) for s,e in b_vocals] if abs(b_ratio - 1.0) > 0.005 else b_vocals)
    b_entry_has_vocal = any(start <= b_entry_time + trans_window and end > b_entry_time for start, end in scaled_vocal_b)

    if a_exit_phrase and float(a_exit_phrase.get('vocal_density', 0.0)) > 0.50:
        a_exit_has_vocal = True

    vocal_clash = a_exit_has_vocal and b_entry_has_vocal
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
        'a_exit_section':       a_exit_section,
        'a_exit_density':       float(a_exit_density),
        'a_exit_energy':        a_exit_energy,
        'a_exit_mixability':    a_exit_mixability,
        'a_exit_bass':          a_exit_bass,
        'a_exit_vocal':         a_exit_vocal,
        'a_exit_tension':       a_exit_tension,
        'a_energy_trajectory':  a_energy_trajectory,
        'b_entry_section':      b_entry_section,
        'b_mix_character':      b_mix_character,
        'vocal_clash':          vocal_clash,
        'dance_moment':         dance_moment,
    }

def select_technique(energy_a, energy_b, bpm_diff, key_compat, is_amapiano, intro_beats, mix_count, ctx=None, set_snapshot=None, runway_beats=9999, quiet=False):
    scores = {}
    for name, tech in TECHNIQUE_LIBRARY.items():
        s = 0
        
        tech_min = TECHNIQUE_MIN_BEATS.get(name, 32)
        if runway_beats < tech_min:
            s -= 500
            scores[name] = s
            continue
            
        tech_ideal = TECHNIQUE_IDEAL_BEATS.get(name, 64)
        if runway_beats >= tech_ideal:
            s += 15
        elif runway_beats >= tech_min:
            s += 5
            
        if ctx is not None:
            dm = ctx['dance_moment']
            density = ctx['a_exit_density']
            traj = ctx['a_energy_trajectory']
            b_char = ctx['b_mix_character']
            v_clash = ctx['vocal_clash']

            if dm == 'hard_reset':
                if name == "ECHO_FREEZE":  s += 70
                if name == "ECHO_THROW":   s += 45
                if name == "BASS_SWAP":    s += 20
                if name == "SLOW_BURN":    s -= 40
            elif dm == 'reboot':
                if name == "ECHO_FREEZE":  s += 60
                if name == "ECHO_THROW":   s += 40
                if name == "BASS_SWAP":    s += 30
                if name == "SLOW_BURN":    s -= 20
            elif dm == 'controlled_rebuild':
                if name == "BASS_SWAP":    s += 50
                if name == "PIANO_HANDOFF":s += 45 if is_amapiano else 15
                if name == "SLOW_BURN":    s += 35
                if name == "ECHO_FREEZE":  s -= 15
            elif dm == 'build_release':
                if name == "ECHO_FREEZE":  s += 50
                if name == "BASS_SWAP":    s += 40
                if name == "PIANO_HANDOFF":s += 30 if is_amapiano else 0
                if name == "SLOW_BURN":    s -= 30
            elif dm == 'peak_swap':
                if name == "BASS_SWAP":    s += 50
                if name == "PIANO_HANDOFF":s += 30 if is_amapiano else 0
                if name == "ECHO_FREEZE":  s -= 10
                if name == "SLOW_BURN":    s -= 20
            elif dm == 'harmonic_lift':
                if name == "PIANO_HANDOFF":s += 55 if is_amapiano else 25
                if name == "FILTER_SWEEP": s += 40
                if name == "BASS_SWAP":    s += 30
                if name == "ECHO_FREEZE":  s -= 20
            elif dm == 'melodic_reset':
                if name == "PIANO_HANDOFF":s += 50 if is_amapiano else 20
                if name == "SLOW_BURN":    s += 45
                if name == "FILTER_SWEEP": s += 35
                if name in ("ECHO_FREEZE","ECHO_THROW"): s -= 35
            elif dm == 'groove_extension':
                if name == "BASS_SWAP":    s += 55
                if name == "PIANO_HANDOFF":s += 40 if is_amapiano else 10
                if name == "FILTER_SWEEP": s += 30
            elif dm == 'vocal_relief':
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 55
                if name == "BASS_SWAP":    s -= 35
                if name == "SLOW_BURN":    s -= 25
            elif dm == 'pre_peak_build':
                if name == "ECHO_FREEZE":  s += 45
                if name == "BASS_SWAP":    s += 25
                if name == "SLOW_BURN":    s -= 20
            elif dm == 'natural_exit':
                if name == "SLOW_BURN":    s += 50
                if name == "FILTER_SWEEP": s += 40
                if name == "BASS_SWAP":    s += 30
                if name == "ECHO_FREEZE":  s -= 20
            elif dm == 'breather':
                if name == "SLOW_BURN":    s += 55
                if name == "FILTER_SWEEP": s += 45
                if name in ("ECHO_FREEZE","ECHO_THROW"): s -= 40
            elif dm == 'cool_down':
                if name == "SLOW_BURN":    s += 50
                if name == "FILTER_SWEEP": s += 40
                if name in ("ECHO_FREEZE","ECHO_THROW"): s -= 30

            if density < 0.35:
                s += 10
                if name == "SLOW_BURN": s += 10
            elif density > 0.65:
                if name == "BASS_SWAP": s += 20
                elif name in ("SLOW_BURN", "FILTER_SWEEP"): s -= 15
                elif name in ("ECHO_FREEZE","ECHO_THROW"): s += 10

            if b_char == 'clean_intro':
                if name == "SLOW_BURN": s += 20
                elif name == "BASS_SWAP": s += 10
            elif b_char == 'drum_heavy':
                if name == "BASS_SWAP": s += 30
                elif name == "PIANO_HANDOFF": s += 20 if is_amapiano else -10
                elif name == "ECHO_FREEZE": s += 10
            elif b_char == 'vocal_heavy':
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 30
                elif name == "SLOW_BURN": s -= 10
            elif b_char == 'melodic':
                if name == "PIANO_HANDOFF": s += 30 if is_amapiano else 0
                elif name == "FILTER_SWEEP": s += 20
                elif name == "BASS_SWAP": s += 10

            if v_clash:
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 40
                elif name == "BASS_SWAP": s -= 30
                elif name == "SLOW_BURN": s -= 20

            if traj == 'floor':
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 20
            elif traj == 'falling':
                if name == "FILTER_SWEEP": s += 15
                elif name == "SLOW_BURN": s += 10

        if name == "SLOW_BURN": s += 12
        elif name in ("ECHO_FREEZE", "ECHO_THROW"): s += 20 if bpm_diff > 5.0 else 8 if bpm_diff > 2.0 else 0
        else: s += 20 if bpm_diff <= 1.0 else 8 if bpm_diff <= 3.0 else -15

        if key_compat == "exact":
            if name == "BASS_SWAP": s += 25
            elif name == "PIANO_HANDOFF": s += 20
            else: s += 10
        elif key_compat == "compatible": s += 15 if name == "BASS_SWAP" else 5
        else:
            if name in ("ECHO_FREEZE","ECHO_THROW"): s += 25
            if name == "SLOW_BURN": s += 15
            if name == "BASS_SWAP": s -= 25
            if name == "PIANO_HANDOFF": s -= 30

        if is_amapiano:
            if name == "PIANO_HANDOFF": s += 90
            elif name == "BASS_SWAP": s += 50
            elif name in ("ECHO_FREEZE","ECHO_THROW"): s -= 60
            elif name == "FILTER_SWEEP": s -= 40
            elif name == "SLOW_BURN": s -= 15

        if ctx is None:
            if energy_a == "High" and energy_b == "High":
                if name == "BASS_SWAP": s += 40
                elif name == "SLOW_BURN": s -= 30
            elif energy_a == "High" and energy_b == "Low/Chill":
                if name in ("FILTER_SWEEP","SLOW_BURN"): s += 40
            elif energy_a == "Low/Chill" and energy_b == "High":
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 40
                elif name == "BASS_SWAP": s += 25

        if name != "ECHO_FREEZE":
            min_beats_needed = TECHNIQUE_MIN_BEATS.get(name, 64) * 0.65
            if intro_beats < min_beats_needed: s -= 60

        if name in ("ECHO_FREEZE","ECHO_THROW"): s -= 15
        variety_idx = list(TECHNIQUE_LIBRARY.keys()).index(name)
        if variety_idx == (mix_count % len(TECHNIQUE_LIBRARY)): s -= 12
        scores[name] = s

    if set_snapshot is not None and SET_STATE_AVAILABLE:
        from set_state import SetStateModel
        _tmp = SetStateModel.__new__(SetStateModel)
        _tmp.events = [] 
        scores = _tmp.apply_to_scoring(scores, set_snapshot)

    best = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    
    if not quiet:
        print(f"   🎯 Technique scores: " + " | ".join(f"{n}={v}" for n,v in ranked))
        if runway_beats < 9999:
            print(f"   📏 Runway: {runway_beats} beats — vetoed: {sum(1 for v in scores.values() if v <= -400)}")
        if set_snapshot:
            req_breather = set_snapshot.time_since_breather > 120 and set_snapshot.energy_rolling > 0.7
            req_intensity = set_snapshot.time_since_hard_drop > 300 and set_snapshot.energy_rolling < 0.5 and set_snapshot.mix_count > 3
            print(f"   🎪 Set phase: {set_snapshot.set_phase}  energy={set_snapshot.energy_rolling:.2f}  {'⚠️ BREATHER ' if req_breather else ''}{'💥 INTENSITY ' if req_intensity else ''}")
    return best

# ════════════════════════════════════════════════════════════════════════════════
# DECISION CORE ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class Intent:
    goal: str
    target_energy_delta: float
    target_transition_intensity: float
    target_novelty: float
    vocal_tolerance: str
    harmonic_risk_tolerance: str
    preferred_entry_styles: list[str]
    preferred_exit_styles: list[str]
    max_execution_risk: float

@dataclass
class CandidateOption:
    option_type: str
    track_b: Optional[dict]
    delay_beats: float

    def label(self) -> str:
        if self.option_type == "hold":
            return f"HOLD +{int(self.delay_beats)} beats"
        name = self.track_b.get('filename', '?') if self.track_b else '?'
        return f"TRANSITION {os.path.basename(name)} @ +{int(self.delay_beats)} beats"

@dataclass
class EvaluatedOption:
    option: CandidateOption
    final_score: float
    intent_fit: float
    transition_opportunity: float
    global_compat: float
    robustness: float
    execution_risk: float
    future_flex: float
    reasoning: list[str] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    draft_plan: Optional[dict] = None

@dataclass
class DecisionRecord:
    intent: Intent
    chosen: EvaluatedOption
    alternatives: list[EvaluatedOption]
    timestamp: float

class IntentEngine:
    def build_intent(
        self, 
        current_track: dict, 
        set_snapshot: Any | None, 
        energy_history: list[str] | None = None,
        transition_history: list[dict] | None = None,
        set_duration_remaining: float | None = None
    ) -> Intent:
        
        if set_duration_remaining is not None and set_duration_remaining < 600:
            return Intent("closing_sequence", -0.05, 0.35, 0.20, "low", "low", ["clean_intro", "harmonic_bed"], ["dj_friendly_outro", "decompression"], 0.30)

        if transition_history:
            for rec in reversed(transition_history):
                if rec.get('rating') is not None or rec.get('failure_tags'):
                    last_rating = rec.get('rating')
                    last_tags = rec.get('failure_tags', [])
                    if last_rating is not None and last_rating <= 4:
                        return Intent("recovery", 0.00, 0.30, 0.15, "low", "low", ["clean_intro", "drum_foundation"], ["dj_friendly_outro", "decompression"], 0.25)
                    if 'vocal_clash' in last_tags or 'bass_fight' in last_tags:
                        return Intent("recovery", 0.00, 0.30, 0.15, "low", "low", ["clean_intro"], ["dj_friendly_outro"], 0.25)
                    break  

        if set_snapshot is None:
            if energy_history and len(energy_history) >= 3:
                recent = energy_history[-3:]
                high_count = sum(1 for e in recent if e == "High")

                if high_count >= 3:
                    return Intent("controlled_rebuild", -0.10, 0.35, 0.25, "medium", "medium", ["clean_intro", "drum_foundation"], ["dj_friendly_outro", "decompression"], 0.35)

                low_count = sum(1 for e in recent if e in ("Low/Chill",))
                if low_count >= 3:
                    return Intent("peak_lift", 0.20, 0.65, 0.40, "medium", "medium", ["drum_foundation", "clean_intro"], ["decompression", "dj_friendly_outro"], 0.42)

                if len(set(recent)) == 1 and len(energy_history) > 5:
                    return Intent("harmonic_refresh", 0.05, 0.45, 0.55, "medium", "high", ["clean_intro", "melodic"], ["decompression", "harmonic_bed", "dj_friendly_outro"], 0.40)

            return Intent("balanced_progression", 0.0, 0.45, 0.35, "medium", "medium", ["clean_intro", "drum_foundation"], ["dj_friendly_outro", "decompression"], 0.45)

        energy = float(getattr(set_snapshot, "energy_rolling", 0.5))
        vocal_fatigue = float(getattr(set_snapshot, "vocal_density_fatigue", 0.0))
        time_since_breather = float(getattr(set_snapshot, "time_since_breather", 0.0))
        time_since_hard_drop = float(getattr(set_snapshot, "time_since_hard_drop", 0.0))
        novelty = float(getattr(set_snapshot, "novelty_rate", 0.5))
        mix_count = int(getattr(set_snapshot, "mix_count", 0))

        if vocal_fatigue > 0.8:
            return Intent("vocal_rest", 0.00, 0.40, 0.30, "low", "medium", ["clean_intro", "drum_foundation"], ["dj_friendly_outro", "decompression"], 0.38)

        if 0.55 < energy < 0.70 and time_since_hard_drop > 120 and mix_count > 2:
            return Intent("tension_build", 0.10, 0.55, 0.30, "medium", "medium", ["drum_foundation", "clean_intro"], ["tension_build", "decompression"], 0.40)

        if energy > 0.72 and time_since_breather > 120:
            return Intent("controlled_rebuild", -0.10, 0.35, 0.25, "low" if vocal_fatigue > 0.6 else "medium", "medium", ["clean_intro", "harmonic_bed", "drum_foundation"], ["dj_friendly_outro", "decompression"], 0.35)

        if energy < 0.45 and time_since_hard_drop > 180:
            return Intent("peak_lift", 0.20, 0.65, 0.40, "medium", "medium", ["drum_foundation", "clean_intro"], ["decompression", "dj_friendly_outro"], 0.42)

        if novelty < 0.25:
            return Intent("harmonic_refresh", 0.05, 0.45, 0.55, "medium", "high", ["clean_intro", "melodic"], ["decompression", "harmonic_bed", "dj_friendly_outro"], 0.40)

        return Intent("hold_groove", 0.00, 0.45, 0.30, "low" if vocal_fatigue > 0.7 else "medium", "medium", ["clean_intro", "drum_foundation"], ["dj_friendly_outro", "decompression"], 0.42)

class DecisionCore:
    def __init__(self, brain, searcher, technique_selector, context_builder, set_state_model=None, technique_min_beats=None, now_fn=None):
        self.brain = brain
        self.searcher = searcher
        self.select_technique = technique_selector
        self.build_transition_context = context_builder
        self.set_state_model = set_state_model
        self.intent_engine = IntentEngine()
        self.technique_min_beats = technique_min_beats or {}
        self.now_fn = now_fn

    def _now(self):
        return self.now_fn() if self.now_fn else time.time()

    def decide_next_action(self, current_track: dict, candidate_tracks: list[dict], current_pos_a: float, track_a_dur: float, master_bpm: float, spb: float, mix_count: int, energy_history: list[str], transition_history: list[dict], set_duration_remaining: float | None = None) -> DecisionRecord:
        set_snapshot = self.set_state_model.get_snapshot() if self.set_state_model else None
        intent = self.intent_engine.build_intent(current_track, set_snapshot, energy_history, transition_history, set_duration_remaining)

        options = self._generate_options(candidate_tracks)

        evaluated: list[EvaluatedOption] = []
        for option in options:
            try:
                ev = self._evaluate_option(option, intent, current_track, current_pos_a, track_a_dur, master_bpm, spb, mix_count)
                if ev: evaluated.append(ev)
            except Exception as e:
                print(f"⚠️ Option evaluation failed: {e}")

        evaluated.sort(key=lambda x: x.final_score, reverse=True)
        chosen = evaluated[0] if evaluated else None

        return DecisionRecord(intent=intent, chosen=chosen, alternatives=evaluated, timestamp=self._now())

    def _generate_options(self, candidate_tracks: list[dict]) -> list[CandidateOption]:
        opts: list[CandidateOption] = []
        for tb in candidate_tracks:
            opts.append(CandidateOption("transition", tb, 0.0))
            opts.append(CandidateOption("transition", tb, 16.0))
            opts.append(CandidateOption("transition", tb, 32.0))
        opts.append(CandidateOption("hold", None, 16.0))
        opts.append(CandidateOption("hold", None, 32.0))
        return opts

    def _evaluate_option(self, option: CandidateOption, intent: Intent, current_track: dict, current_pos_a: float, track_a_dur: float, master_bpm: float, spb: float, mix_count: int) -> EvaluatedOption | None:
        if option.option_type == "hold":
            return self._evaluate_hold_option(option, intent, current_track, current_pos_a, track_a_dur, spb)

        tb = dict(option.track_b) if option.track_b else {}
        future_a_pos = min(current_pos_a + option.delay_beats * spb, max(track_a_dur - 1.0, current_pos_a))

        kc = self.brain.key_compat(current_track.get('key', 'C'), tb.get('key', 'C'))
        
        original_bpm_diff = abs(tb.get('bpm', master_bpm) - master_bpm)
        raw_b_ratio = master_bpm / tb.get('bpm', master_bpm) if tb.get('bpm', master_bpm) > 0 else 1.0
        b_ratio = 1.0 if (raw_b_ratio < 0.92 or raw_b_ratio > 1.08) else raw_b_ratio
        
        post_stretch_bpm = tb.get('bpm', master_bpm) * b_ratio
        effective_bpm_diff = abs(post_stretch_bpm - master_bpm)
        
        is_amp = (current_track.get('genre') == 'amapiano' or tb.get('genre') == 'amapiano')

        global_compat_raw = self.brain.bpm_score(current_track.get('bpm', 112), tb.get('bpm', 112), is_amp) + self.brain.camelot_score(current_track.get('key', 'C'), tb.get('key', 'C'))[0] + self.brain.vocal_score(current_track, tb) + self.brain.spectral_score(current_track, tb) + self.brain.genre_score(current_track, tb)
        global_compat = max(0.0, min(1.0, (global_compat_raw + 100) / 150.0))

        intro_beats = 30.0 / spb
        for sec in tb.get('structure_map', []):
            if sec.get('label') == 'intro':
                intro_beats = sec.get('end', 30.0) / spb
                break

        search_dur = 80.0 * spb 

        try:
            best_a_exit, best_b_entry, overlap_score, all_candidates = \
                self.searcher.search(current_track, tb, b_ratio, search_dur, spb, quiet=True)
        except Exception as e:
            print(f"⚠️ Searcher failed for {tb.get('filename', '?')}: {e}")
            return None

        if best_a_exit < future_a_pos - (8.0 * spb):
            return None

        runway_seconds = track_a_dur - best_a_exit - SEARCH_SAFETY_BUFFER_SEC
        runway_beats = max(0, int(runway_seconds / spb))

        if runway_beats < 16:
            return None 

        rough_ctx = self.build_transition_context(
            current_track, tb, spb, mix_out_time=best_a_exit,
            b_entry_time=best_b_entry, trans_dur=search_dur, b_ratio=b_ratio)

        tech_name = self.select_technique(
            current_track.get('energy', 'High'), tb.get('energy', 'High'),
            effective_bpm_diff, kc, is_amp, intro_beats, mix_count,
            ctx=rough_ctx,
            set_snapshot=self.set_state_model.get_snapshot() if self.set_state_model else None,
            runway_beats=runway_beats,   
            quiet=True)

        mem_key, recipe = self.brain.ml.generate_recipe(
            tech_name, current_track.get('energy', 'High'),
            tb.get('energy', 'High'), is_amp, quiet=True)

        tech_min  = float(TECHNIQUE_MIN_BEATS.get(tech_name, 32.0))
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

        a_phrases = current_track.get('phrases', []) or current_track.get('phrase_map', []) or current_track.get('phrase_analysis', []) or current_track.get('phrase_windows', [])
        a_exit_mix = 0.5
        if a_phrases:
            try: a_exit_mix = float(min(a_phrases, key=lambda p: abs(float(p.get('start', 0.0)) - best_a_exit)).get('mixability', 0.5))
            except Exception: pass

        b_phrases = tb.get('phrases', []) or tb.get('phrase_map', []) or tb.get('phrase_analysis', []) or tb.get('phrase_windows', [])
        b_entry_mix = 0.5
        if b_phrases:
            try: b_entry_mix = float(min(b_phrases, key=lambda p: abs(float(p.get('start', 0.0)) - (best_b_entry * b_ratio))).get('mixability', 0.5))
            except Exception: pass

        robustness = min(1.0, len([c for c in all_candidates if float(c['score'].get('total', 0.0)) >= 0.65]) / 4.0)
        transition_opportunity = (0.45 * float(overlap_score.get('total', 0.0)) + 0.25 * a_exit_mix + 0.20 * b_entry_mix + 0.10 * robustness)

        intent_fit = 0.0
        reasons = []
        eb = tb.get('energy', 'High')
        
        if intent.goal == "recovery":
            if float(overlap_score.get('total', 0.0)) > 0.8: intent_fit += 0.30; reasons.append("highly_safe_overlap")
            if tech_name in ("SLOW_BURN", "FILTER_SWEEP"): intent_fit += 0.20; reasons.append("safe_technique_for_recovery")
        elif intent.goal == "vocal_rest":
            if not tb.get('stems', {}).get('has_vocals', False): intent_fit += 0.40; reasons.append("vocal_rest_achieved")
        elif intent.goal == "tension_build":
            if tech_name in ("FILTER_SWEEP", "BASS_SWAP"): intent_fit += 0.20; reasons.append("builds_tension")
        elif intent.goal == "closing_sequence":
            if eb in ("Low/Chill",): intent_fit += 0.30; reasons.append("winding_down_energy")
            if tech_name in ("SLOW_BURN", "ECHO_THROW"): intent_fit += 0.20; reasons.append("closing_technique")
        elif intent.goal == "controlled_rebuild":
            if eb in ("Low/Chill", "Medium", "High"): intent_fit += 0.20; reasons.append("supports_controlled_rebuild")
            if tech_name in ("FILTER_SWEEP", "SLOW_BURN"): intent_fit += 0.20; reasons.append("technique_matches_controlled_rebuild")
        elif intent.goal == "peak_lift":
            if eb == "High": intent_fit += 0.25; reasons.append("supports_peak_lift")
            if tech_name in ("BASS_SWAP", "PIANO_HANDOFF", "ECHO_THROW", "ECHO_FREEZE"): intent_fit += 0.20; reasons.append("technique_matches_peak_lift")
        elif intent.goal == "hold_groove":
            intent_fit += 0.15; reasons.append("supports_hold_groove")
            if tech_name in ("BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"): intent_fit += 0.15; reasons.append("stable_transition_family")
        elif intent.goal == "harmonic_refresh":
            if kc in ("exact", "compatible"): intent_fit += 0.25; reasons.append("harmonic_compatibility_for_refresh")
            if tech_name in ("FILTER_SWEEP", "SLOW_BURN", "BASS_SWAP"): intent_fit += 0.15; reasons.append("smooth_technique_for_refresh")
        elif intent.goal == "balanced_progression":
            intent_fit += 0.15; reasons.append("balanced_progression_baseline")
            if tech_name in ("BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"): intent_fit += 0.10; reasons.append("versatile_technique")
        
        if intent.vocal_tolerance == "low":
            if not (current_track.get('stems', {}).get('has_vocals', False) and tb.get('stems', {}).get('has_vocals', False)):
                intent_fit += 0.20; reasons.append("low_vocal_overlap_risk")
        intent_fit = max(0.0, min(1.0, intent_fit + (0.20 * a_exit_mix) + (0.10 * b_entry_mix)))

        risk = 0.0
        if robustness < 0.35: risk += 0.25; reasons.append("few_good_candidate_pairs")
        if float(overlap_score.get('vocal', 0.5)) < 0.65: risk += 0.20; reasons.append("vocal_overlap_risk")
        if float(overlap_score.get('harmonic', 0.5)) < 0.50: risk += 0.18; reasons.append("harmonic_risk")
        if track_a_dur - best_a_exit < trans_dur_est + 20.0: risk += 0.20; reasons.append("tight_exit_room")
        if b_entry_mix < 0.55: risk += 0.12; reasons.append("dense_b_entry")
        
        time_to_exit = best_a_exit - future_a_pos 
        far_threshold = 96.0 * spb 
        if time_to_exit > far_threshold:
            overshoot = (time_to_exit - far_threshold) / far_threshold 
            risk += min(0.60, 0.15 + 0.15 * overshoot) 
            reasons.append(f"exit_too_far_away_({time_to_exit:.1f}s)")

        execution_risk = max(0.0, min(1.0, risk))

        future_flex = max(0.0, min(1.0, 0.4 + min(len(tb.get('best_exit_phrases', [])), 4) * 0.1 - (0.1 if tb.get('stems', {}).get('has_vocals', False) else 0.0)))

        reasons.extend([
            f"best_exit={best_a_exit:.1f}s",
            f"best_entry={best_b_entry:.1f}s",
            f"technique={tech_name}",
            f"beats={actual_beats:.0f}({['EMERGENCY','SHORTENED','IDEAL'][0 if actual_beats < tech_min else 1 if actual_beats < tech_ideal else 2]})",
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

        final_score = (0.30 * intent_fit + 0.30 * transition_opportunity + 0.15 * global_compat + 0.10 * robustness + 0.10 * future_flex - 0.20 * execution_risk)

        return EvaluatedOption(
            option=option, final_score=float(final_score), intent_fit=float(intent_fit), transition_opportunity=float(transition_opportunity),
            global_compat=float(global_compat), robustness=float(robustness), execution_risk=float(execution_risk), future_flex=float(future_flex),
            reasoning=reasons, assumptions=assumptions,
            draft_plan={
                "track_b": tb, "b_ratio": b_ratio,
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

    def _evaluate_hold_option(self, option: CandidateOption, intent: Intent, current_track: dict, current_pos_a: float, track_a_dur: float, spb: float) -> EvaluatedOption:
        future_pos = current_pos_a + option.delay_beats * spb
        future_pos_ratio = future_pos / max(track_a_dur, 1.0)
        hold_value, reasons = 0.0, []

        if future_pos_ratio < 0.50:
            hold_value += 0.45; reasons.append("early_track_strong_hold")
        elif future_pos_ratio < 0.60:
            hold_value += 0.30; reasons.append("mid_track_hold")
        elif future_pos_ratio < 0.65:
            hold_value += 0.15; reasons.append("approaching_decision_zone")
        elif future_pos_ratio < 0.70:
            hold_value += 0.00; reasons.append("decision_zone_neutral")
        else:
            overshoot = (future_pos_ratio - 0.70) / 0.30  
            hold_value -= 0.20 * overshoot
            reasons.append(f"LATE_hold_penalty_{overshoot:.2f}")

        if intent.goal in ("hold_groove", "controlled_rebuild", "tension_build", "recovery"): 
            intent_bonus = 0.25 * max(0.0, 1.0 - future_pos_ratio)
            hold_value += intent_bonus
            reasons.append(f"intent_{intent.goal}_scaled_{intent_bonus:.2f}")
        
        if option.delay_beats == 16: hold_value += 0.05
        elif option.delay_beats == 32: hold_value += 0.02

        execution_risk = 0.20 if option.delay_beats == 16 else 0.32
        
        track_momentum = 0.8 if intent.goal in ("peak_lift", "tension_build") else 0.5
        final_score = (0.55 * hold_value + 0.20 * (1.0 - execution_risk) + 0.15 * 0.30 + 0.10 * track_momentum)

        return EvaluatedOption(
            option=option, final_score=float(final_score), intent_fit=float(hold_value), transition_opportunity=0.0, global_compat=0.0, robustness=0.0, execution_risk=float(execution_risk), future_flex=0.30,
            reasoning=reasons + [f"future_pos={future_pos:.1f}s ratio={future_pos_ratio:.2f}"],
            assumptions={"a_should_continue_delivering_value": True}, draft_plan=None
        )

# ════════════════════════════════════════════════════════════════════════════════
# PHASE B — PHRASE-LEVEL CANDIDATE SEARCH 
# ════════════════════════════════════════════════════════════════════════════════
class OverlapScorer:
    def score(self, ta, tb, a_exit_time, b_entry_time, trans_dur, spb):
        scores = {}
        scores['spectral'] = 1.0 if ta.get('energy','High') == tb.get('energy','High') else 0.5

        ca, cb = CAMELOT_WHEEL.get(ta.get('key','C')), CAMELOT_WHEEL.get(tb.get('key','C'))
        if ca is not None and cb is not None:
            diff = min(abs(ca[0]-cb[0]), 12-abs(ca[0]-cb[0]))
            scores['harmonic'] = 1.0 if diff==0 else 0.75 if diff==1 else 0.50 if diff==2 else 0.10
        else:
            scores['harmonic'] = 0.5

        def overlap_fraction(regions, window_start, window_end):
            window_len = window_end - window_start
            if window_len <= 0: return 0.0
            covered = sum(max(0.0, min(e,window_end)-max(s,window_start)) for s,e in regions)
            return min(1.0, covered/window_len)

        a_vf = overlap_fraction(ta.get('stems',{}).get('vocal_regions',[]), a_exit_time, a_exit_time+trans_dur)
        b_vf = overlap_fraction(tb.get('stems',{}).get('vocal_regions',[]), b_entry_time, b_entry_time+trans_dur)
        scores['vocal'] = 1.0 - (a_vf * b_vf)

        bpm_ratio = min(ta.get('bpm',112.0), tb.get('bpm',112.0)) / max(ta.get('bpm',112.0), tb.get('bpm',112.0), 0.01)
        a_dd = len([t for t in ta.get('stems',{}).get('log_drum_hits',[]) if a_exit_time<=t<=a_exit_time+trans_dur]) / max(trans_dur,1)
        b_dd = len([t for t in tb.get('stems',{}).get('log_drum_hits',[]) if b_entry_time<=t<=b_entry_time+trans_dur]) / max(trans_dur,1)
        dc = min(a_dd, b_dd) / max(a_dd, b_dd, 0.01)
        scores['rhythmic'] = max(0.0, min(1.0, bpm_ratio * (1.0 - dc * 0.5)))

        scores['total'] = max(0.0, min(1.0,
            scores['spectral']*0.20 + scores['harmonic']*0.30 +
            scores['vocal']*0.30   + scores['rhythmic']*0.20))
        return scores

class PhraseCandidateSearch:
    def __init__(self):
        self.scorer = OverlapScorer()

    def _is_safe_exit_phrase(self, p, dur):
        if dur <= 0: return False
        start = float(p.get('start', 0.0))
        pos   = start / dur
        fn    = p.get('function', '')
        mix   = float(p.get('mixability', 0.0))
        bass  = float(p.get('bass_density', 0.0))
        vocal = float(p.get('vocal_density', 0.0))
        tens  = float(p.get('tension_score', 0.0))

        if pos < 0.70: return False
        if fn in ('release_peak','bass_showcase','tension_build','fake_outro','vocal_spotlight'):
            return False

        if pos >= 0.80:
            return mix >= 0.40 and bass <= 0.75 and vocal <= 0.70 and tens <= 0.80
        else:
            return mix >= 0.55 and bass <= 0.60 and vocal <= 0.55 and tens <= 0.65

    def _future_rebound_penalty(self, next_phrases):
        penalty = 0.0
        for p in next_phrases[:2]:
            fn = p.get('function', '')
            if fn == 'release_peak':     penalty += 2.5
            elif fn == 'bass_showcase':  penalty += 2.0
            elif fn == 'tension_build':  penalty += 1.5
            elif fn == 'vocal_spotlight':penalty += 1.5
            elif fn == 'fake_outro':     penalty += 2.0
            penalty += (1.2*float(p.get('bass_density',0.0)) +
                        0.8*float(p.get('vocal_density',0.0)) +
                        1.0*float(p.get('tension_score',0.0)) +
                        0.6*float(p.get('percussive_density',0.0)))
        return penalty

    def _score_exit_phrase(self, phrase, next_phrases, track_dur):
        if track_dur <= 0: return -10.0
        pos   = float(phrase.get('start',0.0)) / track_dur
        mix   = float(phrase.get('mixability',0.5))
        bass  = float(phrase.get('bass_density',0.0))
        vocal = float(phrase.get('vocal_density',0.0))
        harm  = float(phrase.get('harmonic_density',0.0))
        perc  = float(phrase.get('percussive_density',0.0))
        tens  = float(phrase.get('tension_score',0.0))
        fn    = phrase.get('function','')

        score = 2.5 * mix

        if   pos > 0.85: score += 3.5
        elif pos > 0.80: score += 2.8
        elif pos > 0.75: score += 2.0
        elif pos > 0.70: score += 1.0
        else:            score -= 1.5

        FN_SCORES = {'dj_friendly_outro':2.5,'decompression':1.5,'drum_foundation':1.0,
                      'harmonic_bed':0.4,'release_peak':-3.5,'bass_showcase':-2.5,
                      'tension_build':-2.2,'fake_outro':-2.8,'vocal_spotlight':-2.8}
        score += FN_SCORES.get(fn, 0.0)

        score -= 1.7*bass + 1.5*vocal + 1.2*tens + 0.5*perc
        score += 0.8*(1.0-harm) + 0.6*(1.0-perc)

        if next_phrases:
            fm = float(sum(p.get('mixability',0.5) for p in next_phrases[:2]) / len(next_phrases[:2]))
            fb = float(sum(p.get('bass_density',0.0) for p in next_phrases[:2]) / len(next_phrases[:2]))
            fv = float(sum(p.get('vocal_density',0.0) for p in next_phrases[:2]) / len(next_phrases[:2]))
            ft = float(sum(p.get('tension_score',0.0) for p in next_phrases[:2]) / len(next_phrases[:2]))
            score += 0.8*fm - 1.0*fb - 0.8*fv - 1.0*ft
            score -= self._future_rebound_penalty(next_phrases)
        return score

    def search(self, ta, tb, b_ratio, trans_dur, spb, quiet=False):
        
        tb_stretched = dict(tb)
        if abs(b_ratio - 1.0) > 0.005:
            stems_s = dict(tb.get('stems', {}))
            if 'vocal_regions' in stems_s:
                stems_s['vocal_regions'] = [(s/b_ratio, e/b_ratio) for s,e in stems_s['vocal_regions']]
            if 'log_drum_hits' in stems_s:
                stems_s['log_drum_hits'] = [t/b_ratio for t in stems_s['log_drum_hits']]
            tb_stretched['stems'] = stems_s
            tb_stretched['bpm'] = tb.get('bpm', 112.0) * b_ratio

        a_exits   = self._candidate_exits(ta, trans_dur, quiet)
        b_entries = self._candidate_entries(tb, b_ratio, trans_dur)

        all_candidates = []
        best_score = -1.0
        best_pair  = (a_exits[0], b_entries[0])

        for a_exit in a_exits:
            for b_entry in b_entries:
                score = self.scorer.score(ta, tb_stretched, a_exit, b_entry, trans_dur, spb)
                all_candidates.append({'a_exit':round(a_exit,2),'b_entry':round(b_entry,2),'score':score})
                if score['total'] > best_score:
                    best_score = score['total']
                    best_pair  = (a_exit, b_entry)

        all_candidates.sort(key=lambda x: -x['score']['total'])
        best_a, best_b = best_pair
        if not quiet:
            print(f"   🔍 Candidate search: {len(a_exits)} exits × {len(b_entries)} entries = {len(all_candidates)} pairs")
            print(f"   🏆 Best pair: A={best_a:.1f}s  B={best_b:.1f}s  score={best_score:.3f}")
        return best_a, best_b, all_candidates[0]['score'], all_candidates

    def _candidate_exits(self, ta, trans_dur, quiet=False):
        dur = ta.get('_duration', DEFAULT_TRACK_DURATION)
        phrases = (ta.get('phrases',[]) or ta.get('phrase_map',[]) or
                   ta.get('phrase_analysis',[]) or ta.get('phrase_windows',[]))

        if phrases:
            scored = []
            for i, p in enumerate(phrases):
                start = float(p.get('start',0.0))
                if (start < dur * MIN_EXIT_POSITION_RATIO or
                    start + trans_dur + SEARCH_SAFETY_BUFFER_SEC >= dur or
                    not self._is_safe_exit_phrase(p, dur)):
                    continue
                s = self._score_exit_phrase(p, phrases[i+1:i+3], dur)
                scored.append((s, start, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                return [round(start,2) for _s, start, _p in scored[:6]]

            late = [(float(p.get('start',0.0)), p) for p in phrases
                    if float(p.get('start',0.0))/max(dur,1.0) >= MIN_EXIT_POSITION_RATIO
                    and float(p.get('start',0.0))+trans_dur+SEARCH_SAFETY_BUFFER_SEC < dur]
            if late:
                return [round(max(late, key=lambda x: x[0])[0], 2)]

        exits_set = set()
        for pt in ta.get('best_exit_phrases',[]): exits_set.add(round(float(pt),1))
        prep = ta.get('zones',{}).get('optimal_mix_out')
        if prep and prep > dur*MIN_EXIT_POSITION_RATIO: exits_set.add(round(float(prep),1))
        for sec in ta.get('structure_map',[]):
            if sec.get('label') in ('breakdown','outro') and sec.get('start',0)>dur*MIN_EXIT_POSITION_RATIO:
                exits_set.add(round(float(sec['start']),1))

        valid = sorted(e for e in exits_set if e>(dur*MIN_EXIT_POSITION_RATIO) and e+trans_dur+SEARCH_SAFETY_BUFFER_SEC<dur)
        fallback = max(30.0, min(dur * 0.80, dur - trans_dur - SEARCH_SAFETY_BUFFER_SEC))
        return valid[:6] if valid else [fallback]

    def _candidate_entries(self, tb, b_ratio, trans_dur):
        entries = set([round(max(0.0, tb.get('first_beat_time',0.0)/b_ratio),1)])
        for pt in tb.get('best_entry_phrases',[]): entries.add(round(pt/b_ratio,1))
        for sec in tb.get('structure_map',[]):
            if sec.get('label')=='intro': entries.add(round(sec.get('end',0)/b_ratio,1))
            if sec.get('label')=='breakdown' and sec.get('start',0)/b_ratio < trans_dur*0.4:
                entries.add(round(sec['start']/b_ratio,1))
        if tb.get('piano_entries',[]): entries.add(round(tb['piano_entries'][0]/b_ratio,1))
        b_dur = tb.get('_duration', DEFAULT_TRACK_DURATION) / b_ratio
        valid = sorted(e for e in entries if e>=0 and e+trans_dur+60.0<b_dur)
        return valid[:5] if valid else [round(max(0.0, tb.get('first_beat_time',0.0)/b_ratio),1)]

# ════════════════════════════════════════════════════════════════════════════════
# PERSISTENT LEARNER
# ════════════════════════════════════════════════════════════════════════════════
class PersistentLearner:
    CONFIDENCE_FILE = "neuro_confidence.json"

    def __init__(self):
        self.memory_file = "neuro_brain_memory.json"
        self.memory = {}
        self.confidence = {}
        self.load_brain()
        self._load_confidence()

    def _default_memory(self):
        mem = {}
        for tech_name, tech in TECHNIQUE_LIBRARY.items():
            for pair in ("High->High","High->Low/Chill","Low/Chill->High", "Low/Chill->Low/Chill","Amapiano->Amapiano"):
                key = f"{tech_name}|{pair}"
                mem[key] = dict(tech["defaults"])
        return mem

    def load_brain(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file,'r') as f: loaded = json.load(f)
            self.memory = self._default_memory()
            self.memory.update(loaded)
            retired = [k for k in self.memory if k.startswith("HARD_CUT|") or k.startswith("STUTTER_DROP|")]
            for k in retired: del self.memory[k]
            if retired: self.save_brain()
        else:
            self.memory = self._default_memory()
            self.save_brain()

    def save_brain(self):
        tmp = self.memory_file + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(self.memory, f, indent=2)
            os.replace(tmp, self.memory_file)
        except Exception:
            pass

    def _load_confidence(self):
        if os.path.exists(self.CONFIDENCE_FILE):
            try:
                with open(self.CONFIDENCE_FILE, 'r') as f:
                    self.confidence = json.load(f)
            except Exception:
                self.confidence = {}
        else:
            self.confidence = {}

    def _save_confidence(self):
        tmp = self.CONFIDENCE_FILE + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(self.confidence, f, indent=2)
            os.replace(tmp, self.CONFIDENCE_FILE)
        except Exception:
            pass

    def generate_recipe(self, technique_name, energy_a, energy_b, is_amapiano, quiet=False):
        pair = "Amapiano->Amapiano" if is_amapiano else f"{energy_a}->{energy_b}"
        key  = f"{technique_name}|{pair}"
        if key not in self.memory: key = f"{technique_name}|High->High"
        base = self.memory[key]
        tech_id = TECHNIQUE_LIBRARY[technique_name]["id"]

        recipe = {
            "technique_name": technique_name, "technique_id": tech_id,
            "beats": max(float(TECHNIQUE_MIN_BEATS.get(technique_name, 64.0)), base["beats"] + random.choice([-16, 0, 16])),
            "bass": clamp01(base["bass"] + random.uniform(-0.05, 0.05)),
            "echo": clamp01(base.get("echo", 0) + random.uniform(-0.15, 0.15)),
            "wash": clamp01(base.get("wash", 0) + random.uniform(-0.10, 0.10)),
            "piano_hold": clamp01(base.get("piano_hold",0) + random.uniform(-0.15, 0.15)),
        }

        if technique_name == "PIANO_HANDOFF": 
            recipe["echo"] = recipe["wash"] = 0.0
            
        if technique_name in ("BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"): 
            recipe["bass"] = min(recipe["bass"], 0.75)
        elif technique_name == "PIANO_HANDOFF": 
            recipe["bass"] = min(recipe["bass"], 0.55)
            
        if technique_name == "FILTER_SWEEP": recipe["wash"] = clamp01(recipe["wash"] + 0.3)
        if technique_name == "ECHO_THROW": recipe["echo"] = 1.0   

        if not quiet:
            print(f"\n🧪 [{TECHNIQUE_LIBRARY[technique_name]['label']}]  {pair}")
            print(f"   Beats:{recipe['beats']:.0f}  Bass@{recipe['bass']*100:.0f}%  "
                  f"Echo:{recipe['echo']:.2f}  Wash:{recipe['wash']:.2f}  "
                  f"PianoHold:{recipe['piano_hold']:.2f}")

        return key, recipe

    def learn_from_feedback(self, rating, mem_key, recipe, failure_tags=None):
        if mem_key not in self.memory: return

        if rating <= 2:    blend = -0.50
        elif rating <= 3:  blend = -0.30
        elif rating <= 4:  blend = -0.12
        elif rating <= 5:  blend = 0.0
        elif rating <= 6:  blend = 0.25
        elif rating <= 7:  blend = 0.40
        elif rating <= 8:  blend = 0.65
        elif rating <= 9:  blend = 0.80
        else:              blend = 1.0

        if blend == 0.0 and not failure_tags: 
            return

        conf = self.confidence.get(mem_key, 0)
        if conf > 10:
            dampening = max(0.3, 1.0 - (conf - 10) * 0.05)
            blend *= dampening

        baseline = self.memory[mem_key]
        updated = dict(baseline)

        if blend != 0.0:
            for field in ['beats', 'bass', 'echo', 'wash', 'piano_hold']:
                if field not in recipe: continue
                base_val = baseline.get(field, 0.0)
                new_val = base_val + blend * (recipe[field] - base_val)
                if field == 'beats': new_val = max(16.0, min(192.0, new_val))
                else: new_val = clamp01(new_val)
                updated[field] = round(new_val, 4)

        if failure_tags:
            tags = set(failure_tags)

            if 'bass_fight' in tags and 'energy_dip' not in tags:
                updated['bass'] = max(0.0, updated.get('bass', 0.5) - 0.15)
            elif 'energy_dip' in tags and 'bass_fight' not in tags:
                updated['bass'] = min(1.0, updated.get('bass', 0.5) + 0.10)

            if 'too_abrupt' in tags and 'outgoing_too_long' not in tags:
                updated['beats'] = min(192.0, updated.get('beats', 64.0) + 16.0)
            elif 'outgoing_too_long' in tags and 'too_abrupt' not in tags:
                updated['beats'] = max(32.0, updated.get('beats', 96.0) - 16.0)
            elif 'no_payoff' in tags and 'too_abrupt' not in tags:
                updated['beats'] = max(32.0, updated.get('beats', 96.0) - 16.0)

            if 'vocal_clash' in tags:
                updated['echo'] = min(1.0, updated.get('echo', 0.0) + 0.20)

        self.confidence[mem_key] = conf + 1

        self.memory[mem_key] = updated
        self.save_brain()
        self._save_confidence()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE A — TRANSITION DATASET  
# ════════════════════════════════════════════════════════════════════════════════
class TransitionDataset:
    LOG_FILE = "transition_log.json"
    FAILURE_TAGS = ["bass_fight", "vocal_clash", "energy_dip", "too_abrupt", "no_payoff", "phrase_mismatch", "tonal_mismatch", "groove_drift", "intro_too_empty", "outgoing_too_long"]
    MAX_RECORDS = 1000

    def __init__(self):
        self.records = []
        self._lock = threading.Lock()
        if os.path.exists(self.LOG_FILE):
            try:
                with open(self.LOG_FILE) as f: self.records = json.load(f)
            except Exception: pass

    def _save(self):
        with self._lock:
            snapshot = list(self.records)
            
        tmp = self.LOG_FILE + ".tmp"
        try:
            with open(tmp, 'w') as f: json.dump(snapshot, f, indent=2)
            os.replace(tmp, self.LOG_FILE)
        except Exception: pass

    def log_transition_async(self, rec_id, ta, tb, plan, ctx, recipe, overlap_score, assumptions=None):
        record = {
            "id": rec_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "track_a": os.path.basename(ta.get('filename', '?')), "track_b": os.path.basename(tb.get('filename', '?')),
            "bpm": plan.get('_master_bpm', 0), "key_a": ta.get('key', '?'), "key_b": tb.get('key', '?'),
            "technique": plan['tech_name'], "recipe": {k: v for k, v in recipe.items() if k not in ('technique_name', 'technique_id')},
            "mix_trigger": round(plan['mix_trigger'], 2), "b_start_w": round(plan['b_start_w'], 2), "trans_dur": round(plan['trans_dur'], 2),
            "a_exit_section": ctx.get('a_exit_section', '?'), "a_exit_function": ctx.get('a_exit_function', '?'),
            "a_exit_density": round(ctx.get('a_exit_density', 0), 3), "a_exit_mixability": round(ctx.get('a_exit_mixability', 0), 3),
            "a_exit_bass": round(ctx.get('a_exit_bass', 0), 3), "a_exit_vocal": round(ctx.get('a_exit_vocal', 0), 3),
            "a_exit_tension": round(ctx.get('a_exit_tension', 0), 3), "b_entry_section": ctx.get('b_entry_section', '?'),
            "vocal_clash": ctx.get('vocal_clash', False), "dance_moment": ctx.get('dance_moment', '?'),
            "overlap_score": round(overlap_score.get('total', 0), 3) if isinstance(overlap_score, dict) else 0.0,
            "assumptions": assumptions or {},
            "rating": None, "failure_tags": [], "adaptation_actions": []
        }
        with self._lock:
            self.records.append(record)
            if len(self.records) > self.MAX_RECORDS:
                self.records = self.records[-self.MAX_RECORDS:]
        self._save()

    def update_rating(self, rec_id, rating, failure_tags):
        with self._lock:
            for rec in self.records:
                if rec['id'] == rec_id:
                    rec['rating'] = rating
                    rec['failure_tags'] = failure_tags
                    break
        self._save()

    def log_adaptation(self, rec_id, actions):
        if not actions: return
        with self._lock:
            for rec in self.records:
                if rec['id'] == rec_id:
                    rec['adaptation_actions'] = actions
                    break
        self._save()

    def get_overlap_score(self, tx_id) -> float:
        with self._lock:
            for rec in self.records:
                if rec['id'] == tx_id:
                    return rec.get('overlap_score', 0.0)
        return 0.0

    def summary(self):
        with self._lock:
            rated = [r for r in self.records if r.get('rating') is not None]
            total = len(self.records)
        if not rated: return "No rated transitions yet."
        avg = sum(r['rating'] for r in rated) / len(rated)
        return f"📊 Transition Dataset: {total} total, {len(rated)} rated (Avg: {avg:.2f}/10)"

# ════════════════════════════════════════════════════════════════════════════════
# DJ BRAIN (Compatibility Scoring)
# ════════════════════════════════════════════════════════════════════════════════
class DJBrain:
    def __init__(self): self.ml = PersistentLearner()
    def camelot_score(self, key_a, key_b):
        ca, cb = CAMELOT_WHEEL.get(key_a), CAMELOT_WHEEL.get(key_b)
        if not ca or not cb: return 10, "unknown"
        if ca[0] == cb[0] and ca[1] == cb[1]: return 40, "exact"
        if ca[0] == cb[0] and ca[1] != cb[1]: return 20, "energy_boost"   
        diff = min(abs(ca[0] - cb[0]), 12 - abs(ca[0] - cb[0]))    
        if diff == 1: return 25, "adjacent"
        if diff == 2: return 10, "two_step"
        return -30, "clash"
    def key_compat(self, key_a, key_b):
        score, label = self.camelot_score(key_a, key_b)
        return "exact" if label in ("exact", "energy_boost") else "compatible" if label == "adjacent" else "clash"
    def bpm_score(self, bpm_a, bpm_b, is_amapiano):
        diff = abs(bpm_b - bpm_a)
        if is_amapiano: return 35 if diff <= 1.0 else 15 if diff <= 2.0 else 0 if diff <= 3.0 else -120  
        else: return 35 if diff <= 1.0 else 20 if diff <= 3.0 else 5 if diff <= 6.0 else -10 if diff <= 8.0 else -50
    def vocal_score(self, track_a, track_b):
        a_vocal = track_a.get('stems', {}).get('has_vocals', False)
        b_vocal = track_b.get('stems', {}).get('has_vocals', False)
        return -20 if a_vocal and b_vocal else 10 if not a_vocal and not b_vocal else 0        
    def spectral_score(self, track_a, track_b):
        return 10 if track_a.get('energy', 'High') == track_b.get('energy', 'High') else -5
    def genre_score(self, track_a, track_b):
        ga, gb = track_a.get('genre', 'open'), track_b.get('genre', 'open')
        return 30 if ga == gb == 'amapiano' else -25 if ga == 'amapiano' and gb != 'amapiano' else -15 if ga != 'amapiano' and gb == 'amapiano' else 0


# ════════════════════════════════════════════════════════════════════════════════
# NEURO-DJ CORE
# ════════════════════════════════════════════════════════════════════════════════
class NeuroDJ(QObject):
    request_rating = pyqtSignal(str, object)

    def __init__(self, library_json):
        super().__init__()
        self.mixer, self.brain, self.dataset, self.searcher = neuro_core.NeuroMixer(), DJBrain(), TransitionDataset(), PhraseCandidateSearch()
        self._analysis_pool, self._learn_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2), concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.track_a_dur = self.track_b_dur = DEFAULT_TRACK_DURATION
        self.track_a_name, self.track_b_name, self.master_bpm, self.current_mix_trigger, self.current_technique, self.mix_count = "Loading...", "—", 0, 0.0, "—", 0
        self.energy_history = []
        self._pending_tx_id = None
        self._tx_lock = threading.Lock()
        self._temp_files = []
        self.target_set_duration = 3600.0  
        self._load_failures = set()
        self._total_pause_time = 0.0
        self._pause_start_time = 0.0
        self.is_paused = False
        
        self._master_bpm_precise = 0.0
        
        with open(library_json,'r') as f: self.crate = json.load(f)
        for t in self.crate:
            if '_duration' not in t: t['_duration'] = librosa.get_duration(path=t['filename']) if os.path.exists(t.get('filename','')) else DEFAULT_TRACK_DURATION
                
        self._ears = None
        self.set_state = SetStateModel() if SET_STATE_AVAILABLE else None

        self.decision_core = DecisionCore(
            brain=self.brain,
            searcher=self.searcher,
            technique_selector=select_technique,
            context_builder=build_transition_context,
            set_state_model=self.set_state,
            technique_min_beats=TECHNIQUE_SAFE_MIN_BEATS
        )
        self.crate_ranker = CrateRanker(self.brain)

    def _sleep_pausable(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.is_paused:
                time.sleep(0.1)
                end_time += 0.1
                continue
            time.sleep(max(0.0, min(0.1, end_time - time.time())))

    def warp_track(self, track, bpm):
        filename = track.get('filename', '')
        if not os.path.exists(filename): raise FileNotFoundError(f"Track file not found: {filename}")
        track_bpm = track.get('bpm', bpm)
        if track_bpm <= 0: track_bpm = bpm
        ratio = bpm / track_bpm
        if ratio < 0.92 or ratio > 1.08: ratio = 1.0

        if 0.99 <= ratio <= 1.01:
            if filename.lower().endswith('.wav'): return os.path.abspath(filename), 1.0  
            else:
                tmp = os.path.abspath(f"temp_sync_deck_b_{uuid.uuid4().hex[:8]}.wav")
                self._temp_files.append(tmp)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, sr = librosa.load(filename, sr=44100, mono=False)
                if y.ndim == 1: y = np.array([y, y])
                sf.write(tmp, y.T, 44100, subtype='FLOAT')
                return tmp, 1.0
        
        tmp = os.path.abspath(f"temp_sync_deck_b_{uuid.uuid4().hex[:8]}.wav")
        self._temp_files.append(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(filename, sr=44100, mono=False)
        if y.ndim == 1: y = np.array([y, y])
        yw = np.array([librosa.effects.time_stretch(c, rate=ratio) for c in y])
        sf.write(tmp, yw.T, 44100, subtype='FLOAT')
        return tmp, ratio

    def _rescale_track_timestamps(self, track: dict, ratio: float):
        if abs(ratio - 1.0) < 0.005:
            return
        inv = 1.0 / ratio  
        
        for key in ('phrases', 'phrase_map', 'phrase_analysis', 'phrase_windows'):
            for p in track.get(key, []):
                for field in ('start', 'end'):
                    if field in p:
                        p[field] = float(p[field]) * inv
        
        for s in track.get('structure_map', []):
            for field in ('start', 'end'):
                if field in s:
                    s[field] = float(s[field]) * inv
        
        stems = track.get('stems', {})
        if 'vocal_regions' in stems:
            stems['vocal_regions'] = [(s*inv, e*inv) for s, e in stems['vocal_regions']]
        if 'log_drum_hits' in stems:
            stems['log_drum_hits'] = [t*inv for t in stems['log_drum_hits']]
        
        for key in ('best_exit_phrases', 'best_entry_phrases', 'piano_entries'):
            if key in track:
                track[key] = [float(t)*inv for t in track[key]]
        
        zones = track.get('zones', {})
        for zk in ('optimal_mix_out', 'optimal_mix_in'):
            if zk in zones:
                zones[zk] = float(zones[zk]) * inv
        
        if 'first_beat_time' in track:
            track['first_beat_time'] = float(track['first_beat_time']) * inv
        
        track['_duration'] = track.get('_duration', DEFAULT_TRACK_DURATION) * inv
        track['bpm'] = track.get('bpm', 112.0) * ratio
        track['_stretch_ratio'] = ratio

    def _schedule_eq_events(self, ta, tb, plan, spb):
        self.mixer.clear_eq_events()
        mix_trigger = plan['mix_trigger']
        trans_dur = plan['trans_dur']
        recipe = plan['recipe']
        b_start_w = plan['b_start_w']
        swap_elapsed = trans_dur * recipe['bass']
        swap_time_a = mix_trigger + swap_elapsed

        stems_a = ta.get('stems', {})
        stems_b = tb.get('stems', {})
        vocal_regions_a = stems_a.get('vocal_regions', [])
        vocal_regions_b = stems_b.get('vocal_regions', [])
        log_drum_hits_b = stems_b.get('log_drum_hits', [])

        a_has_vocal = any(start <= mix_trigger + trans_dur and end >= mix_trigger for start, end in vocal_regions_a)
        
        b_ratio = plan.get('b_ratio', 1.0)
        scaled_vocal_b = ([(s/b_ratio, e/b_ratio) for s,e in vocal_regions_b] if abs(b_ratio - 1.0) > 0.005 else vocal_regions_b)
        b_has_vocal = any(start < b_start_w + trans_dur and end > b_start_w for start, end in scaled_vocal_b)

        if a_has_vocal and b_has_vocal:
            kill_time = swap_time_a - 2.0 * spb
            self.mixer.add_eq_event(0, kill_time if kill_time > mix_trigger else mix_trigger, 1.0, 0.0, 1.0)
            print(f"   🎤 VOCAL CLASH GUARD: mid-kill on A @ {kill_time:.1f}s")
        elif a_has_vocal:
            if swap_time_a - spb > mix_trigger:
                self.mixer.add_eq_event(0, swap_time_a - spb, 1.0, 0.0, 1.0)
                print(f"   🎙️  PRE-SWAP MID-KILL on A @ {swap_time_a - spb:.1f}s")

        scaled_drum_hits = [t / b_ratio for t in log_drum_hits_b] if abs(b_ratio - 1.0) > 0.005 else log_drum_hits_b
        first_drum = next((t for t in scaled_drum_hits if b_start_w <= t <= b_start_w + trans_dur), None)
        if first_drum is not None:
            self.mixer.add_eq_event(1, first_drum, 1.0, 1.0, 1.0)
            self.mixer.add_eq_event(1, b_start_w, 0.0, 1.0, 1.0)
            print(f"   🥁 LOG DRUM SYNC: B bass snaps open @ {first_drum:.1f}s")

    def _is_viable(self, curr, cand):
        is_amp = (curr.get('genre') == 'amapiano' or cand.get('genre') == 'amapiano')
        bpm_limit = 4.0 if is_amp else 10.0
        
        if abs(cand.get('bpm', 112.0) - curr.get('bpm', 112.0)) > bpm_limit:
            return False

        ca = CAMELOT_WHEEL.get(curr.get('key', 'C'))
        cb = CAMELOT_WHEEL.get(cand.get('key', 'C'))
        if ca and cb:
            diff = abs(ca[0] - cb[0])
            if min(diff, 12 - diff) >= 4:
                return False

        if not cand.get('best_entry_phrases'):
            has_intro = any(s.get('label') in ('intro', 'breakdown') for s in cand.get('structure_map', []))
            if not has_intro:
                spb = 60.0 / max(cand.get('bpm', 112.0), 60.0)
                rough_intro_beats = (cand.get('_duration', 300.0) * 0.15) / spb
                if rough_intro_beats < 16:
                    return False
                    
        return True

    def _load_opener(self):
        while self.crate:
            valid = [t for t in self.crate if t.get('filename') not in self._load_failures]
            if not valid:
                print("🛑 No loadable tracks in crate.")
                return None
            opener = min(valid, key=lambda t: t.get('bpm', 112.0) + (5 if t.get('stems', {}).get('has_vocals', False) else 0))
            self.crate.remove(opener)
            ta = opener
            ta['stretch_ratio'] = 1.0
            filename = ta.get('filename', '')

            if not os.path.exists(filename): continue
            try: self.track_a_dur = librosa.get_duration(path=filename)
            except Exception: self.track_a_dur = DEFAULT_TRACK_DURATION

            try:
                self.track_a_name = os.path.basename(filename)
                load_path = os.path.abspath(filename)
                if not filename.lower().endswith('.wav'):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y, sr = librosa.load(filename, sr=44100, mono=False)
                    tmp_a = os.path.abspath(f"temp_deck_a_{uuid.uuid4().hex[:8]}.wav")
                    self._temp_files.append(tmp_a)
                    if y.ndim == 1: y = np.array([y, y])
                    sf.write(tmp_a, y.T, 44100, subtype='FLOAT')
                    load_path = tmp_a

                self.mixer.load_deck("A", load_path)
                self.mixer.play("A")
                
                self.master_bpm = round(ta.get('bpm', 112.0)) or 112.0 
                self._master_bpm_precise = float(self.master_bpm)
                
                print(f"\n🎵 NOW PLAYING: {self.track_a_name} ({self.master_bpm} BPM, {self.track_a_dur:.0f}s)")
                self.energy_history.append(ta.get('energy', 'High'))
                return ta
            except Exception as e:
                print(f"⚠️  Failed to load opener: {e}")
                self._load_failures.add(filename)
                self.crate.append(ta)
                continue
        return None

    def _make_fallback_plan(self, ta: dict, tb: dict, spb: float):
        pos_now = self.mixer.get_position("A")
        max_tdur = max(16.0 * spb, self.track_a_dur - pos_now - 15.0)
        tdur = min(64.0 * spb, max_tdur)
        trig = max(pos_now + 5.0, min(self.track_a_dur * MIN_EXIT_POSITION_RATIO, self.track_a_dur - tdur - 10.0))
        defs = dict(TECHNIQUE_LIBRARY['SLOW_BURN']['defaults'])
        defs.update({'beats': 64.0, 'technique_name': 'SLOW_BURN', 'technique_id': 4})
        
        try:
            fnb, b_ratio = self.warp_track(tb, self.master_bpm)
        except Exception as e:
            print(f"   ⚠️ Fallback warp failed: {e}")
            return None
            
        return {
            'fnb': fnb, 'b_ratio': b_ratio, 'b_start_w': 0.0, 
            'mix_trigger': trig, 'trans_dur': tdur, 
            'tech_name': 'SLOW_BURN', 'mem_key': f"SLOW_BURN|{ta.get('energy', 'High')}->{tb.get('energy', 'High')}", 
            'recipe': defs, 'track_b_dur': tb.get('_duration', DEFAULT_TRACK_DURATION), 
            'ctx': {}, 'overlap_score': {}, 'assumptions': {},
            '_master_bpm': self.master_bpm
        }

    def _finalize_mix_plan(self, decision: EvaluatedOption, ta: dict, spb: float) -> dict:
        draft = decision.draft_plan
        if not draft or 'track_b' not in draft:
            fb = self._make_fallback_plan(ta, decision.option.track_b if decision.option.track_b else ta, spb)
            if fb is None: raise RuntimeError("Fallback plan failed")
            return fb

        tb = draft['track_b']
        
        try:
            fnb, actual_b_ratio = self.warp_track(tb, self.master_bpm)
        except Exception:
            fb = self._make_fallback_plan(ta, tb, spb)
            if fb is None: raise RuntimeError("Fallback plan failed")
            return fb

        track_a_dur = self.track_a_dur
        mix_trigger = draft['mix_trigger']
        recipe = draft['recipe']
        tech_name = draft['tech_name']
        b_start_w = draft['b_start_w']
        final_mem_key = draft['mem_key']
        b_ratio = actual_b_ratio

        if abs(b_ratio - draft['b_ratio']) > 0.01:
            b_start_w = draft['b_start_w'] * (draft['b_ratio'] / b_ratio) if b_ratio > 0 else draft['b_start_w']

        pos_now = self.mixer.get_position("A")
        min_trigger = pos_now + 8.0
        max_trigger = track_a_dur - 25.0
        if max_trigger < min_trigger:
            max_trigger = track_a_dur - 10.0

        mix_trigger = max(min_trigger, min(mix_trigger, max_trigger))

        real_runway = track_a_dur - mix_trigger - FINALIZE_SAFETY_BUFFER_SEC

        if real_runway < 8.0:
            print(f"   ⚠️  Only {real_runway:.1f}s runway — aborting transition")
            fb = self._make_fallback_plan(ta, tb, spb)
            if fb is None: raise RuntimeError("Fallback plan failed")
            return fb

        desired_beats = recipe['beats']
        desired_dur = desired_beats * spb
        
        old_tech = tech_name
        old_beats = recipe['beats']

        if desired_dur > real_runway:
            available_beats = real_runway / spb
            safe_min = TECHNIQUE_SAFE_MIN_BEATS.get(tech_name, 16.0)

            if available_beats >= safe_min:
                new_beats = max(safe_min, 8.0 * int(available_beats / 8.0))
                recipe['beats'] = new_beats
                print(f"   ⚠️  Shrunk {tech_name}: {desired_beats:.0f}→{new_beats:.0f} beats "
                      f"(runway={real_runway:.1f}s)")
            else:
                echo_min = TECHNIQUE_SAFE_MIN_BEATS.get("ECHO_THROW", 16.0)
                if available_beats >= echo_min:
                    tech_name = "ECHO_THROW"
                    final_mem_key, recipe = self.brain.ml.generate_recipe(
                        tech_name, ta.get('energy', 'High'),
                        tb.get('energy', 'High'),
                        ta.get('genre') == 'amapiano')
                    recipe['beats'] = max(echo_min, 8.0 * int(available_beats / 8.0))
                    print(f"   ⚠️  Switched to ECHO_THROW ({recipe['beats']:.0f} beats)")
                else:
                    print(f"   ⚠️  Only {available_beats:.0f} beats — insufficient")
                    fb = self._make_fallback_plan(ta, tb, spb)
                    if fb is None: raise RuntimeError("Fallback plan failed")
                    return fb

        trans_dur = recipe['beats'] * spb

        replan_needed = False
        if tech_name != old_tech or abs(recipe['beats'] - old_beats) >= 16.0:
            replan_needed = True

        if replan_needed:
            try:
                best_a_exit, best_b_entry, overlap_score, _ = self.searcher.search(
                    ta, tb, b_ratio, trans_dur, spb, quiet=True
                )
                mix_trigger = max(min_trigger, min(best_a_exit, track_a_dur - trans_dur - FINALIZE_SAFETY_BUFFER_SEC))
                b_start_w = best_b_entry
                print(f"   🔁 Replanned phrase pair after duration change: "
                      f"A={mix_trigger:.1f}s B={b_start_w:.1f}s")
            except Exception as e:
                print(f"   ⚠️  Replan after shrink failed: {e}")

        if mix_trigger + trans_dur + EXECUTION_RESERVE_SEC > track_a_dur:
            trans_dur = max(8.0 * spb, track_a_dur - mix_trigger - EXECUTION_RESERVE_SEC)
            recipe['beats'] = max(8.0, 8.0 * round((trans_dur / spb) / 8.0))
            trans_dur = recipe['beats'] * spb 
            print(f"   ⚠️  Final clamp: {recipe['beats']:.0f} beats / {trans_dur:.1f}s")

        ctx = build_transition_context(
            ta, tb, spb, mix_out_time=mix_trigger,
            b_entry_time=b_start_w, trans_dur=trans_dur, b_ratio=b_ratio)

        buffer_secs = track_a_dur - mix_trigger - trans_dur
        print(f"\n   {'═'*48}")
        print(f"   ✅ PLAN FINALIZED")
        print(f"   ├─ Trigger:    {mix_trigger:.1f}s  ({mix_trigger/track_a_dur*100:.0f}%)")
        print(f"   ├─ Technique:  {TECHNIQUE_LIBRARY[tech_name]['label']}")
        print(f"   ├─ Beats:      {recipe['beats']:.0f}")
        print(f"   ├─ Duration:   {trans_dur:.1f}s")
        print(f"   ├─ Trans ends: {mix_trigger + trans_dur:.1f}s  (buffer={buffer_secs:.1f}s)")
        print(f"   ├─ B cue:      {b_start_w:.1f}s")
        print(f"   └─ Bass swap:  beat {recipe['bass']*recipe['beats']:.0f} of {recipe['beats']:.0f}")
        print(f"   {'═'*48}")

        return {
            'fnb': fnb, 'b_ratio': b_ratio, 'b_start_w': b_start_w,
            'mix_trigger': mix_trigger, 'trans_dur': trans_dur,
            'tech_name': tech_name, 'mem_key': final_mem_key,
            'recipe': recipe, 'track_b_dur': tb['_duration'],
            'overlap_score': draft['overlap_score'], 'ctx': ctx,
            'assumptions': decision.assumptions,
            '_master_bpm': self.master_bpm,
        }

    def cleanup_temp_files(self, keep_last=True):
        if not self._temp_files: return
        safe_count = 2 if keep_last else 0
        files_to_delete = self._temp_files[:-safe_count] if safe_count else self._temp_files
        for f in files_to_delete:
            try: os.remove(f)
            except Exception: pass
        self._temp_files = self._temp_files[-safe_count:] if safe_count else []

    def start_set(self):
        if not self.crate: return
        ta = self._load_opener()
        if ta is None: return

        set_start_time = time.time()

        while self.crate:
            self.crate = [t for t in self.crate if t.get('filename') not in self._load_failures]
            if not self.crate: 
                print("🛑 Crate exhausted or all remaining tracks corrupted.")
                break

            spb = 60.0 / self.master_bpm
            elapsed = (time.time() - set_start_time) - self._total_pause_time
            set_duration_remaining = max(0.0, self.target_set_duration - elapsed)
            
            print(f"\n⚙️  DecisionCore evaluating options...")
            
            viable_crate = [t for t in self.crate if self._is_viable(ta, t)]
            if not viable_crate:
                print("   ⚠️  No viable candidates — using least-bad fallback")
                viable_crate = sorted(self.crate, key=lambda c: abs(c.get('bpm', 112) - ta.get('bpm', 112)))[:3]
            else:
                vetoed = len(self.crate) - len(viable_crate)
                if vetoed: print(f"   🔍 Pre-filter: {vetoed} vetoed, {len(viable_crate)} viable")
            
            candidate_pool = self.crate_ranker.select_candidates(
                current_track=ta,
                crate=viable_crate,
                master_bpm=self.master_bpm,
                n=8, 
                energy_history=list(self.energy_history)
            )
            
            with self.dataset._lock:
                trans_history = copy.deepcopy(self.dataset.records[-5:])
                
            decision_future = self._analysis_pool.submit(
                self.decision_core.decide_next_action,
                current_track=ta,
                candidate_tracks=candidate_pool,
                current_pos_a=self.mixer.get_position("A"),
                track_a_dur=self.track_a_dur,
                master_bpm=self.master_bpm,
                spb=spb,
                mix_count=self.mix_count,
                energy_history=list(self.energy_history),
                transition_history=trans_history,
                set_duration_remaining=set_duration_remaining
            )

            decision = None
            wait_deadline = time.time() + MAX_WAIT_SECONDS
            while decision is None:
                if time.time() > wait_deadline: break
                
                if self.is_paused:
                    time.sleep(0.1)
                    wait_deadline += 0.1
                    continue
                    
                try: decision = decision_future.result(timeout=3.0)
                except concurrent.futures.TimeoutError: pass
                except Exception as e: print(f"❌ Decision failed: {e}"); break

            if decision is None or decision.chosen is None or decision.chosen.option.option_type == "hold":
                time_left = self.track_a_dur - self.mixer.get_position("A")
                
                pipeline_budget = 12.0 + 3.0 + 32.0 * spb + 15.0
                force_threshold = max(90.0, pipeline_budget + 30.0)

                if time_left < force_threshold:
                    print(f"   ⚠️  Only {time_left:.0f}s left (need {force_threshold:.0f}s). Forcing transition!")
                    if decision and decision.alternatives:
                        forced_opt = next((alt for alt in decision.alternatives if alt.option.option_type == "transition" and alt.draft_plan is not None), None)
                        if forced_opt:
                            decision = DecisionRecord(
                                intent=decision.intent, chosen=forced_opt,
                                alternatives=decision.alternatives,
                                timestamp=decision.timestamp)
                        else:
                            print("   💀 No viable transition. Ending set.")
                            break
                    else:
                        print("   💀 No alternatives available. Ending set.")
                        break
                else:
                    delay = (decision.chosen.option.delay_beats if (decision and decision.chosen) else 32.0)
                    max_sleep = max(2.0, (time_left - force_threshold) * 0.3)
                    actual_sleep = min(delay * spb * 0.5, max_sleep)
                    print(f"   ⏳ Hold: sleeping {actual_sleep:.1f}s ({time_left:.0f}s remaining)")
                    self._sleep_pausable(actual_sleep)
                    continue

            print("🧠 DECISION LOCKED:")
            print(f"   Goal: {decision.intent.goal}")
            print(f"   Action: {decision.chosen.option.label()}")
            for r in decision.chosen.reasoning[:5]: print(f"   - {r}")

            tb = decision.chosen.option.track_b
            if tb in self.crate: self.crate.remove(tb)
            self.track_b_name = os.path.basename(tb.get('filename','?'))

            try:
                plan = self._finalize_mix_plan(decision.chosen, ta, spb)
                self.track_b_dur = plan['track_b_dur']
            except RuntimeError as e:
                print(f"   ⚠️ Finalize skipped track: {e}")
                self._load_failures.add(tb.get('filename'))
                continue

            pos_after_warp = self.mixer.get_position("A")
            if pos_after_warp >= plan['mix_trigger']:
                gap = pos_after_warp - plan['mix_trigger']
                if gap > plan['trans_dur'] * 0.5:
                    print(f"   ⚠️  Overran trigger by {gap:.1f}s — too late for blend, re-planning")
                    self.crate.append(tb)
                    continue
                plan['mix_trigger'] = pos_after_warp + 1.0
                remaining = self.track_a_dur - plan['mix_trigger'] - EXECUTION_RESERVE_SEC
                
                if remaining < 8.0 * spb:
                    print(f"   ⚠️ Track nearly over ({remaining:.1f}s) — re-queueing")
                    self.crate.append(tb)
                    continue
                
                if plan['trans_dur'] > remaining:
                    plan['trans_dur'] = max(8.0 * spb, remaining)
                    plan['recipe']['beats'] = max(8.0, 8.0 * round((plan['trans_dur'] / spb) / 8.0))
                    plan['trans_dur'] = plan['recipe']['beats'] * spb
                print(f"   ⚡ Immediate trigger at {plan['mix_trigger']:.1f}s (trans={plan['trans_dur']:.1f}s)")

            self.current_mix_trigger = plan['mix_trigger']
            self.current_technique = TECHNIQUE_LIBRARY[plan['tech_name']]['label']

            wait_limit = time.time() + max(300.0, plan['mix_trigger'] - self.mixer.get_position("A") + 30.0)
            while self.mixer.get_position("A") < (plan['mix_trigger'] - 12.0):
                if time.time() > wait_limit:
                    print("   ⚠️ Pre-trigger wait timeout")
                    break
                if self.is_paused:
                    time.sleep(0.1)
                    wait_limit += 0.1
                    continue
                time.sleep(0.5)

            try:
                self.mixer.load_deck("B", plan['fnb'])
            except Exception as e:
                print(f"⚠️ Failed to load Deck B ({plan['fnb']}): {e}")
                self.current_mix_trigger = 0.0
                self.current_technique = "—"
                self.track_b_name = "—"
                self._load_failures.add(tb.get('filename'))
                self.crate.append(tb)
                continue

            self.crate_ranker.record_selection(tb)
            self.mixer.seek("B", plan['b_start_w'])

            wait_start = time.time()
            max_wait = max(plan['trans_dur'] + 30.0, plan['mix_trigger'] - self.mixer.get_position("A") + 15.0)

            while True:
                tl = plan['mix_trigger'] - self.mixer.get_position("A")
                if tl <= 0: 
                    break
                if time.time() - wait_start > max_wait:
                    print(f"   ⚠️  Wait timeout — forcing trigger now")
                    break
                
                if self.is_paused:
                    time.sleep(0.1)
                    wait_start += 0.1
                    continue
                    
                if tl > 0.05:
                    time.sleep(min(tl * 0.5, 0.1))
                elif tl > 0.005:
                    time.sleep(0.001)
                else:
                    deadline = time.perf_counter() + tl
                    while time.perf_counter() < deadline: pass

            recipe = plan['recipe']
            tx_id = f"tx_{int(time.time())}_{self.mix_count}"
            with self._tx_lock:
                self._pending_tx_id = tx_id
            
            self._learn_pool.submit(self.dataset.log_transition_async, tx_id, ta, tb, plan, plan.get('ctx', {}), recipe, plan.get('overlap_score', {}), plan.get('assumptions', {}))

            self._schedule_eq_events(ta, tb, plan, spb)

            self.mixer.trigger_hybrid_transition(plan['trans_dur'], recipe['beats'], recipe['bass'], recipe['echo'], 0.0, recipe['wash'], float(self.master_bpm), float(recipe.get('piano_hold', 0.0)), recipe['technique_id'], random.uniform(1.8, 3.2), random.uniform(4.5, 7.5), random.uniform(0.0, 6.28), random.uniform(0.0, 6.28), max(0.005, min(0.020, random.gauss(0.010, 0.003))))

            if LIVE_EARS_AVAILABLE: self._ears = LiveEars(self.mixer, float(self.master_bpm)); self._ears.start(); self._ears.notify_b_started()

            pickup_deadline = time.time() + 0.2
            while not self.mixer.is_transitioning():
                if time.time() > pickup_deadline: break
                time.sleep(0.001)

            if ADAPTIVE_EXECUTOR_AVAILABLE and self._ears is not None:
                adapt_log = AdaptiveExecutor(self.mixer, self._ears, plan, spb).run()
                if tx_id and getattr(adapt_log, 'actions_taken', None):
                    self._learn_pool.submit(self.dataset.log_adaptation, tx_id, adapt_log.actions_taken)
                
                if getattr(adapt_log, 'handoff_forced', False):
                    self.mixer.pause("A") 
                    handoff_deadline = time.time() + 2.0
                    while self.mixer.is_transitioning() and time.time() < handoff_deadline:
                        time.sleep(0.02)
                else:
                    t_deadline = time.time() + 15.0
                    while self.mixer.is_transitioning() and time.time() < t_deadline: 
                        time.sleep(0.02)
            else:
                t_deadline = time.time() + plan['trans_dur'] + 15.0
                while self.mixer.is_transitioning() and time.time() < t_deadline: time.sleep(0.02)

            if self._ears is not None: self._ears.stop(); self._ears = None
            
            if plan['recipe'].get('technique_id', 0) in (3, 7): 
                self._sleep_pausable(4.0)

            self.mixer.swap_decks()
            
            if abs(plan['b_ratio'] - 1.0) > 0.005:
                self._rescale_track_timestamps(tb, plan['b_ratio'])

            new_bpm = float(tb.get('bpm', self.master_bpm))
            if new_bpm > 0:
                self._master_bpm_precise = 0.7 * self._master_bpm_precise + 0.3 * new_bpm
                self.master_bpm = round(self._master_bpm_precise)

            self.track_a_dur, self.track_a_name, self.track_b_name, self.current_mix_trigger, self.current_technique, self.mix_count = tb.get('_duration', DEFAULT_TRACK_DURATION), self.track_b_name, "—", 0.0, "—", self.mix_count + 1

            self.energy_history.append(tb.get('energy', 'High'))
            if len(self.energy_history) > 100: self.energy_history = self.energy_history[-50:]

            if self.set_state: self._learn_pool.submit(self.set_state.update_transition, ta, tb, plan['tech_name'])
            self.request_rating.emit(plan['mem_key'], recipe); ta = tb
            self.cleanup_temp_files(keep_last=True)

        print(f"\n🏁 Setlist complete. '{self.track_a_name}' playing out.")
        self.brain.ml.save_brain()
        self.cleanup_temp_files(keep_last=False)

# ════════════════════════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════════════════════════
class Stream(QObject):
    newText = pyqtSignal(str)
    def write(self, t): self.newText.emit(str(t))
    def flush(self): pass

class NeuroDJWindow(QMainWindow):
    def __init__(self, dj):
        super().__init__()
        self.dj, self._old_stdout = dj, sys.stdout
        self.setWindowTitle("Neuro-DJ — Dynamic Technique Intelligence")
        self.setGeometry(100,100,1080,880); self.setStyleSheet("background:#121212;color:#FFF;")
        root = QVBoxLayout(); root.setSpacing(8); root.setContentsMargins(12,12,12,12)
        self.setCentralWidget(QWidget()); self.centralWidget().setLayout(root)
        
        self._msg_lock_expiry = 0.0

        h = QLabel("🧠 NEURO-DJ MAINSTAGE"); h.setFont(QFont("Arial",22,QFont.Bold)); h.setStyleSheet("color:#00FFCC;padding:4px 0;"); root.addWidget(h)

        dr = QHBoxLayout()
        for deck, label_text, color in [("A", "DECK A", "#FF0055"), ("B", "DECK B", "#0088FF")]:
            frame = QFrame(); frame.setStyleSheet("background:#1E1E1E;border-radius:8px;padding:8px;"); layout = QVBoxLayout(frame)
            lbl_d, lbl_n = QLabel(label_text), QLabel("—")
            lbl_d.setFont(QFont("Arial",13,QFont.Bold)); lbl_n.setStyleSheet("color:#777;font-size:11px;"); lbl_n.setWordWrap(True)
            plot = pg.PlotWidget(); plot.setBackground('#1E1E1E'); plot.setYRange(-1,1); plot.hideAxis('left'); plot.hideAxis('bottom'); plot.setFixedHeight(110)
            curve = plot.plot(pen=pg.mkPen(color,width=2))
            prog = QProgressBar(); prog.setStyleSheet(f"QProgressBar::chunk{{background:{color};}}"); prog.setTextVisible(False); prog.setFixedHeight(5)
            for w in (lbl_d, lbl_n, plot, prog): layout.addWidget(w)
            dr.addWidget(frame)
            if deck == "A": self.lbl_da, self.lbl_na, self.plot_a, self.curve_a, self.prog_a = lbl_d, lbl_n, plot, curve, prog
            else: self.lbl_db, self.lbl_nb, self.plot_b, self.curve_b, self.prog_b = lbl_d, lbl_n, plot, curve, prog
        root.addLayout(dr)

        inf = QFrame(); inf.setStyleSheet("background:#1A1A1A;border:1px solid #2A2A2A;border-radius:6px;padding:6px;"); ir = QHBoxLayout(inf); ir.setSpacing(14)
        self.lbl_bpm = QLabel("♩ — BPM"); self.lbl_bpm.setFont(QFont("Consolas",11,QFont.Bold)); self.lbl_bpm.setStyleSheet("color:#FFAA00;"); ir.addWidget(self.lbl_bpm)
        sep = QLabel("│"); sep.setStyleSheet("color:#333;"); ir.addWidget(sep)
        self.lbl_tech = QLabel("⬜ —"); self.lbl_tech.setFont(QFont("Consolas",11,QFont.Bold)); self.lbl_tech.setStyleSheet("color:#AA88FF;"); ir.addWidget(self.lbl_tech)
        sep2 = QLabel("│"); sep2.setStyleSheet("color:#333;"); ir.addWidget(sep2)
        self.lbl_cd = QLabel("⏱ Calculating..."); self.lbl_cd.setFont(QFont("Consolas",11,QFont.Bold)); self.lbl_cd.setStyleSheet("color:#00FFCC;"); ir.addWidget(self.lbl_cd)
        ir.addStretch()

        self.btn_play_pause = QPushButton("⏸ PAUSE")
        self.btn_play_pause.setFont(QFont("Arial",10,QFont.Bold))
        self.btn_play_pause.setStyleSheet("QPushButton{background:#333;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}QPushButton:hover{background:#555;}")
        self.btn_play_pause.clicked.connect(self.toggle_pause)
        ir.addWidget(self.btn_play_pause)

        self.btn_skip = QPushButton("⏩  SKIP TO MIX  (−30s)"); self.btn_skip.setFont(QFont("Arial",10,QFont.Bold)); self.btn_skip.setStyleSheet("QPushButton{background:#FF5500;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}QPushButton:hover{background:#FF7733;}QPushButton:disabled{background:#2A2A2A;color:#555;}"); self.btn_skip.setEnabled(False); self.btn_skip.clicked.connect(self.skip_to_mix); ir.addWidget(self.btn_skip)
        root.addWidget(inf)

        self.prog_mix = QProgressBar(); self.prog_mix.setRange(0,100); self.prog_mix.setValue(0); self.prog_mix.setTextVisible(False); self.prog_mix.setFixedHeight(6); self.prog_mix.setStyleSheet("QProgressBar{background:#1A1A1A;border-radius:3px;}QProgressBar::chunk{background:#FF5500;border-radius:3px;}"); root.addWidget(self.prog_mix)

        self.rlhf = QFrame(); self.rlhf.setStyleSheet("background:#1A1A2E;border:2px solid #00FFCC;border-radius:8px;padding:8px;"); rl_root = QVBoxLayout(self.rlhf)
        row1 = QHBoxLayout(); self.lbl_rate = QLabel("🤖 Rate transition (1–10):"); self.lbl_rate.setFont(QFont("Arial",11,QFont.Bold))
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(1,10); self.slider.setValue(5); self.slider.setFocusPolicy(Qt.NoFocus)
        self.lbl_sv = QLabel("5"); self.lbl_sv.setFont(QFont("Consolas",14,QFont.Bold)); self.lbl_sv.setStyleSheet("color:#00FFCC;min-width:22px;"); self.slider.valueChanged.connect(lambda v: self.lbl_sv.setText(str(v)))
        self.btn_train = QPushButton("TRAIN MODEL"); self.btn_train.setStyleSheet("background:#00FFCC;color:#000;font-weight:bold;padding:7px 14px;"); self.btn_train.clicked.connect(self.submit_rating)
        for w in (self.lbl_rate, self.slider, self.lbl_sv, self.btn_train): row1.addWidget(w)
        rl_root.addLayout(row1)

        tags_label = QLabel("Tag failure modes (optional):"); tags_label.setFont(QFont("Arial",9)); tags_label.setStyleSheet("color:#AAA;"); rl_root.addWidget(tags_label)
        tags_row1, tags_row2, self._tag_checks = QHBoxLayout(), QHBoxLayout(), {}
        for i, tag in enumerate(TransitionDataset.FAILURE_TAGS):
            cb = QCheckBox(tag.replace('_', ' ')); cb.setStyleSheet("color:#CCC;font-size:10px;"); self._tag_checks[tag] = cb
            (tags_row1 if i < 5 else tags_row2).addWidget(cb)
        rl_root.addLayout(tags_row1); rl_root.addLayout(tags_row2)
        root.addWidget(self.rlhf); self.rlhf.hide()

        self.console = QTextEdit(); self.console.setReadOnly(True); self.console.setFocusPolicy(Qt.NoFocus); self.console.setStyleSheet("background:#000;color:#00FF00;font-family:Consolas;font-size:11px;"); root.addWidget(self.console)

        sys.stdout = Stream(newText=self._append_console)
        self.dj.request_rating.connect(self._show_rating); self.pending_rating = None
        self.timer = QTimer(); self.timer.timeout.connect(self._tick); self.timer.start(50)

    def _append_console(self, t):
        c = self.console.textCursor(); c.movePosition(QTextCursor.End); c.insertText(t)
        if self.console.document().blockCount() > 2000:
            c.movePosition(QTextCursor.Start)
            c.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 500)
            c.removeSelectedText()
            c.movePosition(QTextCursor.End)
        self.console.setTextCursor(c); self.console.ensureCursorVisible()

    def toggle_pause(self):
        self.dj.is_paused = not self.dj.is_paused
        if self.dj.is_paused:
            self.dj._pause_start_time = time.time()
            self.dj.mixer.pause("A")
            self.dj.mixer.pause("B")
            self.btn_play_pause.setText("▶ RESUME")
            self.btn_play_pause.setStyleSheet("QPushButton{background:#00AA00;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}QPushButton:hover{background:#00CC00;}")
        else:
            if self.dj._pause_start_time > 0:
                self.dj._total_pause_time += (time.time() - self.dj._pause_start_time)
                self.dj._pause_start_time = 0.0
            self.dj.mixer.play("A")
            if self.dj.mixer.is_transitioning():
                self.dj.mixer.play("B")
            self.btn_play_pause.setText("⏸ PAUSE")
            self.btn_play_pause.setStyleSheet("QPushButton{background:#333;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}QPushButton:hover{background:#555;}")

    def skip_to_mix(self):
        mt = getattr(self.dj,'current_mix_trigger',0.0)
        if mt > 30: self.dj.mixer.seek("A", mt-30); print(f"\n⏩ SKIP → {mt-30:.1f}s")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Up: self.skip_to_mix()
        elif ev.key() == Qt.Key_Space: self.toggle_pause()
        else: super().keyPressEvent(ev)

    def closeEvent(self, ev):
        sys.stdout = self._old_stdout
        print("\n🛑 Saving brain and shutting down threads...")
        self.dj._analysis_pool.shutdown(wait=False)
        self.dj._learn_pool.shutdown(wait=True)
        self.dj.brain.ml.save_brain()
        self.dj.cleanup_temp_files(keep_last=False)
        ev.accept()

    def _tick(self):
        try: self.curve_a.setData(self.dj.mixer.get_visual_buffer("A")); self.curve_b.setData(self.dj.mixer.get_visual_buffer("B"))
        except Exception as e: pass

        pa, pb, da, db = self.dj.mixer.get_position("A"), self.dj.mixer.get_position("B"), max(1, self.dj.track_a_dur), max(1, self.dj.track_b_dur)
        self.prog_a.setValue(min(int(pa/da*100),100)); self.prog_b.setValue(min(int(pb/db*100),100))
        self.lbl_na.setText(self.dj.track_a_name); self.lbl_nb.setText(self.dj.track_b_name)
        if self.dj.master_bpm: self.lbl_bpm.setText(f"♩ {self.dj.master_bpm} BPM")
        self.lbl_tech.setText(getattr(self.dj,'current_technique','—'))

        is_trans = self.dj.mixer.is_transitioning()
        mt = getattr(self.dj,'current_mix_trigger',0.0)
        can_update_msg = time.time() > getattr(self, '_msg_lock_expiry', 0.0)
        
        if is_trans:
            self.lbl_da.setText(f"DECK A:  🌊 MIXING OUT"); self.lbl_db.setText(f"DECK B:  🌊 MIXING IN")
            self.lbl_da.setStyleSheet("color:#FFAA00;"); self.lbl_db.setStyleSheet("color:#00FFCC;")
            self.prog_mix.setValue(100); self.btn_skip.setEnabled(False)
            if can_update_msg:
                self.lbl_cd.setText("🔥 MIXING LIVE")
                self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
        elif pa > 0.1:
            self.lbl_da.setText(f"DECK A:  ▶ LIVE"); self.lbl_db.setText(f"DECK B:  Cued")
            self.lbl_da.setStyleSheet("color:#FF0055;"); self.lbl_db.setStyleSheet("color:#555;")

        if not is_trans and mt > 0.5:
            ttm = mt - pa
            if ttm > 0:
                m, s = int(ttm//60), int(ttm%60)
                self.btn_skip.setEnabled(ttm >= 30)
                if mt > 0.0: self.prog_mix.setValue(min(int(max(0, mt-ttm)/mt*100),99))
                if can_update_msg:
                    self.lbl_cd.setText(f"⏱ Next Mix: {m}m {s:02d}s" if m else f"⏱ Next Mix: {s}s")
                    self.lbl_cd.setStyleSheet("color:#FF0055;" if ttm < 30 else "color:#FFAA00;" if ttm < 90 else "color:#00FFCC;")
            else: 
                self.prog_mix.setValue(100); self.btn_skip.setEnabled(False)
                if can_update_msg:
                    self.lbl_cd.setText("🔥 Dropping Now!")
                    self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
        elif not is_trans and mt <= 0.5:
            self.prog_mix.setValue(0); self.btn_skip.setEnabled(False)
            if can_update_msg:
                self.lbl_cd.setText("⏱ Calculating next mix...")
                self.lbl_cd.setStyleSheet("color:#444;")

    def _show_rating(self, mem_key, recipe):
        with self.dj._tx_lock:
            tx_id = getattr(self.dj, '_pending_tx_id', None)
        self.pending_rating = (mem_key, recipe, tx_id)
        self.slider.setValue(5); self.lbl_sv.setText("5")
        score = self.dj.dataset.get_overlap_score(tx_id)
        score_str = f"  [quality={score:.2f}]" if score > 0 else ""
        self.lbl_rate.setText(f"🤖 Rate [{recipe.get('technique_name', '?')}]{score_str} (1–10):")
        for cb in self._tag_checks.values(): cb.setChecked(False)
        self.rlhf.show()

    def submit_rating(self):
        if self.pending_rating:
            mk, rec, tx_id = self.pending_rating
            rating = self.slider.value()
            tags = [tag for tag, cb in self._tag_checks.items() if cb.isChecked()]
            if tx_id:
                self.dj._learn_pool.submit(self.dj.dataset.update_rating, tx_id, rating, tags)
            self.dj._learn_pool.submit(self.dj.brain.ml.learn_from_feedback, rating, mk, rec, tags)
            
            tech = rec.get('technique_name', '?')
            if rating >= 8:
                msg = f"✅ Brain updated — {tech} improved"
            elif rating <= 4:
                msg = f"📉 Slight nudge away from {tech}"
            elif 6 <= rating <= 7:
                msg = f"👍 Small blend toward {tech}"
            else:
                msg = f"⚖️  No change for {tech} (mediocre)"

            if tags:
                msg += f"  | {len(tags)} tag(s)"

            self.lbl_cd.setText(msg)
            self.lbl_cd.setStyleSheet("color:#FFAA00;font-weight:bold;")
            
            self._msg_lock_expiry = time.time() + 4.0

            rated_count = sum(1 for r in self.dj.dataset.records
                              if r.get('rating') is not None) + 1
            if rated_count % 5 == 0:
                print(f"\n{self.dj.dataset.summary()}\n")

            self.pending_rating = None
        self.rlhf.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv); dj = NeuroDJ(sys.argv[1]); win = NeuroDJWindow(dj); win.show()
    threading.Thread(target=dj.start_set, daemon=True).start(); sys.exit(app.exec_())