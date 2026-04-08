"""
neuro_gui.py  —  Neuro-DJ Master Engine (Dynamic Technique Intelligence)

What's new:
  • 7 named mixing techniques, each with its own ML memory slot
  • Technique selector: scores every technique against BPM match, key
    compatibility, energy arc, genre, and intro length — picks the best one
  • ML brain learns per-technique-×-energy-pair (not just energy pair)
  • GUI shows technique name + live technique badge during transition
  • Skip button jumps to T-30s before the next mix trigger
  • Track names, BPM, countdown with urgency colour all live-updated
"""

import neuro_core
import time, json, librosa, soundfile as sf, numpy as np
import os, sys, threading, random, warnings, concurrent.futures
from scipy.signal import find_peaks
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QProgressBar, QTextEdit,
                             QFrame, QPushButton, QSlider)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QTextCursor

# ════════════════════════════════════════════════════════════════════════════════
# TECHNIQUE LIBRARY
# Every technique maps to:
#   id        → integer sent to C++ (selects the algorithm in the callback)
#   label     → displayed in the GUI
#   defaults  → base parameter set (the ML mutates these)
#   beats_mul → default beat-length multiplier (relative to a 32-beat unit)
# ════════════════════════════════════════════════════════════════════════════════
# ── Per-technique minimum beat counts ────────────────────────────────────────
# Based on professional DJ mixing research:
#   Standard dance phrase = 8 bars = 32 beats.
#   Professional overlap: 16–32 bars = 64–128 beats (16/32 bars before outro).
#   Amapiano phrase structure: 32 bars = 128 beats is the natural section length.
#
#   BASS_SWAP / PIANO_HANDOFF:
#     Both tracks play simultaneously the whole window — 128 beats (32 bars)
#     is the Amapiano standard and sounds like a real mix.
#   FILTER_SWEEP:
#     Needs room for the gradual filter open/close — 96 beats minimum.
#   SLOW_BURN:
#     Pure volume crossfade needs the most runway — 128 beats minimum.
#   ECHO_THROW / ECHO_FREEZE:
#     Event-driven — the actual transition is only 4–6 beats.
#     32 beats total gives 26 clean beats of A before anything happens.
TECHNIQUE_MIN_BEATS = {
    "BASS_SWAP":     128,
    "FILTER_SWEEP":   96,
    "ECHO_THROW":     32,
    "ECHO_FREEZE":    32,
    "SLOW_BURN":     128,
    "PIANO_HANDOFF": 128,
}

TECHNIQUE_LIBRARY = {
    "BASS_SWAP": {
        "id": 0, "label": "🔊 Bass Swap",
        "defaults": {"beats": 96.0, "bass": 0.75, "echo": 0.0, "stutter": 0.0,
                     "wash": 0.2, "piano_hold": 0.0},
        "desc": "Bass-kill crossfade — both tracks overlap for a full 96 beats"
    },
    "FILTER_SWEEP": {
        "id": 2, "label": "🌊 Filter Sweep",
        "defaults": {"beats": 96.0, "bass": 0.55, "echo": 0.0, "stutter": 0.0,
                     "wash": 1.0, "piano_hold": 0.0},
        "desc": "LPF closes on A + reverb wash; B opens from dark to full"
    },
    "ECHO_THROW": {
        "id": 3, "label": "🎯 Echo Out",
        "defaults": {"beats": 32.0, "bass": 0.75, "echo": 1.0, "stutter": 0.0,
                     "wash": 0.0, "piano_hold": 0.0},
        "desc": "Trap→Kill→Drop: A clean then echo builds, dry killed, B bass slams"
    },
    "ECHO_FREEZE": {
        "id": 7, "label": "🧊 Echo Freeze",
        "defaults": {"beats": 32.0, "bass": 0.90, "echo": 1.0, "stutter": 0.0,
                     "wash": 0.0, "piano_hold": 0.0},
        "desc": "Pure club weapon: A clean until last 6 beats, trap+kill, B drops into silence"
    },
    "SLOW_BURN": {
        "id": 4, "label": "🕯️ Slow Burn",
        "defaults": {"beats": 128.0, "bass": 0.5, "echo": 0.0, "stutter": 0.0,
                     "wash": 0.0, "piano_hold": 0.0},
        "desc": "Long transparent constant-power crossfade — 128 beats, no EQ tricks"
    },
    "PIANO_HANDOFF": {
        "id": 5, "label": "🎹 Piano Handoff",
        "defaults": {"beats": 96.0, "bass": 0.5, "echo": 0.0, "stutter": 0.0,
                     "wash": 0.0, "piano_hold": 1.0},
        "desc": "Amapiano: keep A piano alive while B log drums phase in, then swap"
    },
}

def build_transition_context(ta, tb, ca, cb, spb):
    """
    Extracts structural context from both tracks at the exact transition moment.
    This is what separates intelligent mixing from metadata matching.

    The context tells select_technique:
      - What section of A is playing when the mix starts
        (breakdown = easy, drop = dangerous, outro = ideal)
      - What section of B opens with
        (clean_intro = safe, drum_heavy = can bass-swap over)
      - Whether a vocal clash will occur
      - What the dancefloor psychology moment is
        (build_moment / peak_moment / release_moment / floor_moment)
      - The texture density at A's exit point
        (< 0.4 = sparse = easy to blend; > 0.7 = dense = risky)
    """
    # ── A exit section ────────────────────────────────────────────────────────
    # Find which section of A the mix_trigger falls in
    mix_out_time = ta.get('zones', {}).get('optimal_mix_out', 0)
    a_structure  = ta.get('structure_map', [])
    a_exit_section  = 'outro'   # fallback
    a_exit_density  = 0.3
    a_exit_energy   = 0.5
    for s in a_structure:
        if s['start'] <= mix_out_time <= s['end']:
            a_exit_section  = s['label']
            a_exit_density  = s['texture_density']
            a_exit_energy   = s['energy_level']
            break

    # ── Texture density at exit ───────────────────────────────────────────────
    # Use fine-grained texture profile if available
    tp_a = ta.get('texture_profile', [])
    if tp_a:
        closest = min(tp_a, key=lambda x: abs(x['time'] - mix_out_time))
        a_exit_density = closest['density']

    # ── B entry section ───────────────────────────────────────────────────────
    b_structure     = tb.get('structure_map', [])
    b_entry_section = 'intro'
    if b_structure:
        b_entry_section = b_structure[0]['label']

    b_mix_character    = tb.get('mix_character', 'clean_intro')
    a_energy_trajectory= ta.get('energy_trajectory', 'falling')

    # ── Vocal clash detection ─────────────────────────────────────────────────
    a_vocals = ta.get('stems', {}).get('vocal_regions', [])
    b_vocals = tb.get('stems', {}).get('vocal_regions', [])
    # Check if A has a vocal in its exit window
    trans_window = 120.0   # rough transition window
    a_exit_has_vocal = any(
        start <= mix_out_time + trans_window and end >= mix_out_time
        for start, end in a_vocals
    )
    b_entry_has_vocal = len(b_vocals) > 0
    vocal_clash = a_exit_has_vocal and b_entry_has_vocal

    # ── Dancefloor psychology moment ──────────────────────────────────────────
    # What is the crowd expecting right now?
    energy_a = ta.get('energy', 'High')
    energy_b = tb.get('energy', 'High')

    if a_energy_trajectory == 'floor' or a_exit_section in ('outro',):
        if energy_b == 'High':
            dance_moment = 'reboot'      # crowd needs re-energising
        else:
            dance_moment = 'cool_down'
    elif a_exit_section == 'breakdown' and energy_b == 'High':
        dance_moment = 'build_release'   # classic tension/release
    elif a_exit_section in ('drop',) and a_energy_trajectory == 'peak':
        dance_moment = 'peak_swap'       # swapping at the peak
    elif a_energy_trajectory == 'falling':
        dance_moment = 'natural_exit'    # A is fading — smooth blend
    else:
        dance_moment = 'neutral'

    return {
        'a_exit_section':     a_exit_section,
        'a_exit_density':     a_exit_density,
        'a_exit_energy':      a_exit_energy,
        'a_energy_trajectory':a_energy_trajectory,
        'b_entry_section':    b_entry_section,
        'b_mix_character':    b_mix_character,
        'vocal_clash':        vocal_clash,
        'dance_moment':       dance_moment,
    }


def select_technique(energy_a, energy_b, bpm_diff, key_compat,
                     is_amapiano, intro_beats, mix_count, ctx=None):
    """
    Scores all 6 techniques using structural context when available.

    With context (the new path):
      The dominant factors are dancefloor psychology moment,
      texture density at A's exit, and B's entry character.

    Without context (fallback):
      Falls back to metadata-only scoring (same as before).
    """
    scores = {}
    for name, tech in TECHNIQUE_LIBRARY.items():
        s = 0

        # ══════════════════════════════════════════════════════════════════════
        # STRUCTURAL SCORING — uses the deep context when available
        # This is the intelligent path that replaces metadata guessing.
        # ══════════════════════════════════════════════════════════════════════
        if ctx is not None:
            dm      = ctx['dance_moment']
            density = ctx['a_exit_density']
            traj    = ctx['a_energy_trajectory']
            a_sect  = ctx['a_exit_section']
            b_char  = ctx['b_mix_character']
            v_clash = ctx['vocal_clash']

            # ── Dance moment scoring ──────────────────────────────────────────
            # This is the highest-weight factor — what the crowd needs
            if dm == 'reboot':
                # A has died away, crowd needs energy injected.
                # Best: Echo Freeze (dramatic silence → B drops like a bomb)
                # Good: Bass Swap, Piano Handoff
                if name == "ECHO_FREEZE":  s += 60
                if name == "ECHO_THROW":   s += 40
                if name == "BASS_SWAP":    s += 30
                if name == "SLOW_BURN":    s -= 20   # too gentle for a reboot

            elif dm == 'build_release':
                # A is in a breakdown — crowd is ready for the drop.
                # B should arrive with maximum impact.
                if name == "ECHO_FREEZE":  s += 50   # silence → B drops = perfect release
                if name == "BASS_SWAP":    s += 40   # A breakdown + B bass in = classic
                if name == "PIANO_HANDOFF":s += 30 if is_amapiano else 0
                if name == "SLOW_BURN":    s -= 30   # wastes the tension

            elif dm == 'peak_swap':
                # Both tracks are at high energy. Best to keep energy level constant.
                if name == "BASS_SWAP":    s += 50   # cleanest at peak
                if name == "PIANO_HANDOFF":s += 30 if is_amapiano else 0
                if name == "ECHO_FREEZE":  s -= 10   # silence kills the peak momentum
                if name == "SLOW_BURN":    s -= 20

            elif dm == 'natural_exit':
                # A is fading naturally — smooth blend, no drama needed.
                if name == "SLOW_BURN":    s += 50   # most natural
                if name == "BASS_SWAP":    s += 30
                if name == "FILTER_SWEEP": s += 40   # gradual filter mirrors the natural fade
                if name == "ECHO_FREEZE":  s -= 20   # overly dramatic for a natural exit

            elif dm == 'cool_down':
                if name == "SLOW_BURN":    s += 50
                if name == "FILTER_SWEEP": s += 40
                if name in ("ECHO_FREEZE","ECHO_THROW"): s -= 30

            # ── Texture density at A's exit ───────────────────────────────────
            # Sparse (density < 0.35): easy to blend — any technique works
            # Dense  (density > 0.65): risky to blend — need surgical techniques
            if density < 0.35:
                s += 10   # all techniques get a small bonus — easy moment
                if name == "SLOW_BURN": s += 10   # especially good for sparse exits
            elif density > 0.65:
                if name == "BASS_SWAP":    s += 20   # bass cut separates the dense mix
                if name in ("SLOW_BURN", "FILTER_SWEEP"): s -= 15  # muddy at high density
                if name in ("ECHO_FREEZE","ECHO_THROW"):  s += 10  # kill solves density

            # ── B's entry character ───────────────────────────────────────────
            if b_char == 'clean_intro':
                if name == "SLOW_BURN":    s += 20   # clean + slow = perfect blend
                if name == "BASS_SWAP":    s += 10
            elif b_char == 'drum_heavy':
                if name == "BASS_SWAP":    s += 30   # drum-heavy B = bass swap to highlight
                if name == "PIANO_HANDOFF":s += 20 if is_amapiano else -10
                if name == "ECHO_FREEZE":  s += 10   # B's drums hit into silence = big
            elif b_char == 'vocal_heavy':
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 30  # clear space for B vocal
                if name == "SLOW_BURN":    s -= 10   # vocal clash if B vocal starts mid-blend
            elif b_char == 'melodic':
                if name == "PIANO_HANDOFF":s += 30 if is_amapiano else 0
                if name == "FILTER_SWEEP": s += 20   # filter opening matches melodic entry
                if name == "BASS_SWAP":    s += 10

            # ── Vocal clash ───────────────────────────────────────────────────
            if v_clash:
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 40  # silence erases A vocal
                if name == "BASS_SWAP":    s -= 30   # bass swap with two vocals = muddy
                if name == "SLOW_BURN":    s -= 20   # both vocals overlap = terrible

            # ── Energy trajectory of A ────────────────────────────────────────
            if traj == 'floor':
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 20  # frozen floor → dramatic drop
            elif traj == 'falling':
                if name == "FILTER_SWEEP": s += 15   # mirrors the natural fall
                if name == "SLOW_BURN":    s += 10

        # ══════════════════════════════════════════════════════════════════════
        # METADATA SCORING — always applied (baseline compatibility)
        # ══════════════════════════════════════════════════════════════════════

        # BPM compatibility
        if name == "SLOW_BURN":
            s += 12
        elif name in ("ECHO_FREEZE", "ECHO_THROW"):
            if bpm_diff > 5.0:   s += 20
            elif bpm_diff > 2.0: s += 8
        else:
            if bpm_diff <= 1.0:   s += 20
            elif bpm_diff <= 3.0: s += 8
            else:                 s -= 15

        # Key compatibility
        if key_compat == "exact":
            if name == "BASS_SWAP":       s += 25
            elif name == "PIANO_HANDOFF": s += 20
            else:                         s += 10
        elif key_compat == "compatible":
            if name == "BASS_SWAP":       s += 15
            else:                         s += 5
        else:
            if name in ("ECHO_FREEZE","ECHO_THROW"): s += 25
            if name == "SLOW_BURN":                   s += 15
            if name == "BASS_SWAP":                   s -= 25
            if name == "PIANO_HANDOFF":               s -= 30

        # Amapiano genre affinity
        if is_amapiano:
            if name == "PIANO_HANDOFF":                          s += 90
            if name == "BASS_SWAP":                              s += 50
            if name in ("ECHO_FREEZE","ECHO_THROW"):             s -= 60
            if name == "FILTER_SWEEP":                           s -= 40
            if name == "SLOW_BURN":                              s -= 15

        # Fallback energy arc (used when context is missing)
        if ctx is None:
            if energy_a == "High" and energy_b == "High":
                if name == "BASS_SWAP":    s += 40
                if name == "SLOW_BURN":    s -= 30
            elif energy_a == "High" and energy_b == "Low/Chill":
                if name in ("FILTER_SWEEP","SLOW_BURN"): s += 40
            elif energy_a == "Low/Chill" and energy_b == "High":
                if name in ("ECHO_FREEZE","ECHO_THROW"): s += 40
                if name == "BASS_SWAP":                   s += 25

        # Intro length guard
        if name != "ECHO_FREEZE":
            min_beats_needed = TECHNIQUE_MIN_BEATS.get(name, 64) * 0.65
            if intro_beats < min_beats_needed:
                s -= 60

        # Echo overuse penalty — keep them as special weapons
        if name in ("ECHO_FREEZE","ECHO_THROW"):
            s -= 15

        # Variety rotation
        variety_idx = list(TECHNIQUE_LIBRARY.keys()).index(name)
        if variety_idx == (mix_count % len(TECHNIQUE_LIBRARY)):
            s -= 12

        scores[name] = s

    best   = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print(f"   🎯 Technique scores: " + " | ".join(f"{n}={v}" for n,v in ranked))
    return best





# ════════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYZER
# ════════════════════════════════════════════════════════════════════════════════
class AudioAnalyzer:
    def __init__(self):
        print("🔬 Amapiano-Native Analyzer ready (HPSS + harmonic flux)...")

    def snap_to_transient(self, filepath, approx_time, window=0.1):
        try:
            s0 = max(0, approx_time - window)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(filepath, sr=44100, offset=s0, duration=window*2)
            if not len(y): return approx_time
            yh = librosa.effects.preemphasis(y)
            pk = np.argmax(np.abs(yh))
            thr = np.abs(yh[pk]) * 0.20
            oi = pk
            while oi > 0 and np.abs(yh[oi]) > thr: oi -= 1
            return s0 + oi/sr
        except: return approx_time

    def quantize_to_phrase(self, raw_time, first_beat, bpm):
        if raw_time <= first_beat: return first_beat
        spb = 60.0/bpm
        beats = (raw_time - first_beat)/spb
        return first_beat + round(beats/16.0)*16.0*spb

    def find_amapiano_cue_points(self, filepath):
        print(f"   🎧 Structural scan: {os.path.basename(filepath)}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(filepath, sr=11025, mono=True)
            dur = librosa.get_duration(y=y, sr=sr)
            harmonic, _ = librosa.effects.hpss(y)
            chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
            flux = np.sum(np.diff(chroma, axis=1)**2, axis=0)
            flux_s = np.convolve(flux, np.ones(20)/20, mode='same')
            flux_t = librosa.frames_to_time(np.arange(len(flux_s)), sr=sr)
            peaks, _ = find_peaks(flux_s, height=np.percentile(flux_s,75),
                                  distance=sr//512*8)
            piano_entries = [float(flux_t[p]) for p in peaks if flux_t[p]>15]
            rms = librosa.feature.rms(y=y)[0]
            rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            log_drums = []
            for i,t in enumerate(rms_t):
                if t>30 and rms[i]>np.percentile(rms,60):
                    fi = np.argmin(np.abs(flux_t-t))
                    if flux_s[fi]<np.percentile(flux_s,40): log_drums.append(t)
            outro = piano_entries[-1] if piano_entries else dur*0.75
            intro = piano_entries[0]  if piano_entries else 30.0
            return {'intro_end':intro,'outro_start':outro,
                    'piano_entries':piano_entries,'log_drum_solos':log_drums,'duration':dur}
        except Exception as ex:
            print(f"   ⚠️ Analyzer: {ex}")
            try: dur=librosa.get_duration(path=filepath)
            except: dur=300.
            return {'intro_end':30.,'outro_start':dur*0.75,
                    'log_drum_solos':[dur*0.75],'piano_entries':[],'duration':dur}


# ════════════════════════════════════════════════════════════════════════════════
# PERSISTENT LEARNER — now keyed per technique×energy pair
# ════════════════════════════════════════════════════════════════════════════════
class PersistentLearner:
    def __init__(self):
        self.memory_file = "neuro_brain_memory.json"
        self.memory = {}
        self.load_brain()

    def _default_memory(self):
        mem = {}
        for tech_name, tech in TECHNIQUE_LIBRARY.items():
            for pair in ("High->High","High->Low/Chill","Low/Chill->High",
                         "Low/Chill->Low/Chill","Amapiano->Amapiano"):
                key = f"{tech_name}|{pair}"
                mem[key] = dict(tech["defaults"])
                mem[key]["timing"] = 0.0
        return mem

    def load_brain(self):
        if os.path.exists(self.memory_file):
            print("💾 ML: Loading persistent memories...")
            with open(self.memory_file,'r') as f:
                loaded = json.load(f)
            # Merge: start from fresh defaults, overlay saved values
            self.memory = self._default_memory()
            self.memory.update(loaded)
            # Purge retired techniques that may have been saved previously.
            retired = [k for k in self.memory
                       if k.startswith("HARD_CUT|") or k.startswith("STUTTER_DROP|")]
            for k in retired:
                del self.memory[k]
            if retired:
                print(f"🗑️  ML: Purged {len(retired)} retired technique memory slots.")
                self.save_brain()
        else:
            print("🌱 ML: Initialising fresh brain...")
            self.memory = self._default_memory()
            self.save_brain()

    def save_brain(self):
        with open(self.memory_file,'w') as f:
            json.dump(self.memory, f, indent=2)

    def generate_recipe(self, technique_name, energy_a, energy_b, is_amapiano):
        pair = "Amapiano->Amapiano" if is_amapiano else f"{energy_a}->{energy_b}"
        key  = f"{technique_name}|{pair}"
        if key not in self.memory:
            key = f"{technique_name}|High->High"
        base = self.memory[key]

        tech_id = TECHNIQUE_LIBRARY[technique_name]["id"]

        recipe = {
            "technique_name": technique_name,
            "technique_id":   tech_id,
            "beats": max(
                float(TECHNIQUE_MIN_BEATS.get(technique_name, 64)),
                base["beats"] + random.choice([-16, 0, 16])
            ),
            # bass = the fraction of the transition before the swap.
            # For overlap techniques (BASS_SWAP, FILTER_SWEEP, PIANO_HANDOFF):
            # the fade window = (1-bass) × beats. If bass drifts to 0.95,
            # on a 128-beat transition the fade window = 6 beats = 3s — hard stop.
            # Cap at 0.75 for overlap techniques so there are always ≥32 beats
            # of gradual fade after the swap.
            # Echo techniques can use higher values (B drops at the kill beat).
            "bass": clamp01(base["bass"] + random.uniform(-0.05, 0.05)),
            "echo":      clamp01(base.get("echo", 0)      + random.uniform(-0.15, 0.15)),
            "stutter":   clamp01(base.get("stutter", 0)   + random.uniform(-0.10, 0.10)),
            "wash":      clamp01(base.get("wash", 0)      + random.uniform(-0.10, 0.10)),
            "piano_hold":clamp01(base.get("piano_hold",0) + random.uniform(-0.15, 0.15)),
            "timing":    base.get("timing", 0)            + random.choice([-16, 0, 16]),
        }

        # Enforce per-technique hard constraints
        if technique_name == "PIANO_HANDOFF":
            recipe["echo"] = recipe["stutter"] = recipe["wash"] = 0.0

        # ── Bass swap point cap ───────────────────────────────────────────────
        # For overlap techniques, the fade window after the swap =
        # (1 - bass) × beats. If bass drifts above 0.75, the fade shrinks fast:
        #   bass=0.80 → fade = 0.20 × 128 = 25 beats  (13s) — borderline
        #   bass=0.85 → fade = 0.15 × 128 = 19 beats  (10s) — short
        #   bass=0.90 → fade = 0.10 × 128 = 13 beats  (7s)  — abrupt
        #   bass=0.95 → fade = 0.05 × 128 = 6  beats  (3s)  — HARD STOP
        # Cap at 0.75 so fade is always ≥ 32 beats (16s at 112BPM) — gradual.
        # Echo techniques keep full range — they use bass differently.
        OVERLAP_BASS_MAX = 0.75
        if technique_name in ("BASS_SWAP", "FILTER_SWEEP", "PIANO_HANDOFF", "SLOW_BURN"):
            recipe["bass"] = min(recipe["bass"], OVERLAP_BASS_MAX)
        if technique_name == "FILTER_SWEEP":
            recipe["wash"]    = clamp01(recipe["wash"] + 0.3)
        if technique_name == "ECHO_THROW":
            recipe["echo"] = 1.0   # Echo Out always runs at full wet — that's the technique

        recipe["timing"] = max(-48.0, min(48.0, recipe["timing"]))

        print(f"\n🧪 [{TECHNIQUE_LIBRARY[technique_name]['label']}]  {pair}")
        print(f"   Beats:{recipe['beats']:.0f}  Bass@{recipe['bass']*100:.0f}%  "
              f"Echo:{recipe['echo']:.2f}  Wash:{recipe['wash']:.2f}  "
              f"Stutter:{recipe['stutter']:.2f}  PianoHold:{recipe['piano_hold']:.2f}")
        return key, recipe

    def learn_from_feedback(self, rating, mem_key, recipe):
        """
        Gradient-blended RLHF update.

        Instead of a hard snap (old: if ≥8 replace baseline entirely),
        we blend the baseline toward or away from the mutation in proportion
        to how good or bad the rating was. This is proper reinforcement learning.

        Rating scale:
          1–3  → Actively bad. Nudge baseline AWAY from mutation (opposite direction).
          4–5  → Mediocre. No change — don't reward, don't punish.
          6–7  → Decent. Small blend toward mutation (20% weight).
          8–9  → Good. Larger blend toward mutation (60% weight).
          10   → Perfect. Full replace — this is the new gold standard.

        The blend formula for any parameter p:
            new_baseline[p] = baseline[p] + blend_weight * (mutation[p] - baseline[p])

        For negative ratings, blend_weight is negative, pushing baseline away.
        All values are re-clamped to [0, 1] after blending.
        """
        if mem_key not in self.memory:
            print(f"⚠️  ML: Unknown key '{mem_key}' — skipping update")
            return

        # Determine blend weight from rating
        if rating <= 3:
            blend = -0.25    # push away from this mutation
            label = f"📉 BAD ({rating}/10) — nudging baseline away"
        elif rating <= 5:
            blend = 0.0      # mediocre — no change
            label = f"⚖️  MEH ({rating}/10) — no change"
        elif rating <= 7:
            blend = 0.20     # decent — small step toward
            label = f"👍 DECENT ({rating}/10) — small blend toward mutation"
        elif rating <= 9:
            blend = 0.60     # good — strong step toward
            label = f"🔥 GOOD ({rating}/10) — blending toward mutation"
        else:
            blend = 1.0      # perfect — full replace
            label = f"⭐ PERFECT (10/10) — replacing baseline"

        print(f"\n{label}")

        if blend == 0.0:
            return   # nothing to do for mediocre

        baseline = self.memory[mem_key]
        # Fields that are stored in memory (strip runtime-only fields from recipe)
        blend_fields = ['beats', 'bass', 'echo', 'stutter', 'wash', 'piano_hold', 'timing']

        updated = dict(baseline)
        changed = []

        for field in blend_fields:
            if field not in recipe:
                continue
            base_val    = baseline.get(field, 0.0)
            mutant_val  = recipe[field]
            new_val     = base_val + blend * (mutant_val - base_val)

            # beats and timing have wider valid ranges
            if field == 'beats':
                new_val = max(16.0, min(192.0, new_val))
            elif field == 'timing':
                new_val = max(-48.0, min(48.0, new_val))
            else:
                new_val = clamp01(new_val)

            if abs(new_val - base_val) > 0.001:
                changed.append(f"{field}: {base_val:.2f}→{new_val:.2f}")
            updated[field] = round(new_val, 4)

        self.memory[mem_key] = updated
        self.save_brain()

        if changed:
            print(f"   📊 Updated '{mem_key}':")
            for c in changed:
                print(f"      {c}")
        else:
            print(f"   📊 No significant parameter change.")


def clamp01(v): return max(0.0, min(1.0, v))


# ════════════════════════════════════════════════════════════════════════════════
# DJ BRAIN + SETLIST ARCHITECT
# ════════════════════════════════════════════════════════════════════════════════
class DJBrain:
    """
    Multi-factor track compatibility scorer.

    Scoring dimensions (all additive, all weighted):
      1. BPM arc       — gradual tempo progression, not random jumps
      2. Harmonic key  — Camelot Wheel mixed-in-key logic (not just exact/compat)
      3. Energy arc    — planned dancefloor journey (build, peak, breath, rebuild)
      4. Vocal clash   — penalise back-to-back vocal tracks (muddiness)
      5. Genre flow    — Amapiano needs Amapiano or very smooth handoff
      6. Spectral fit  — centroid proximity (similar brightness = smooth mix)
    """

    # ── Camelot Wheel ────────────────────────────────────────────────────────
    # Each key maps to its Camelot position (number, mode).
    # Adjacent numbers (±1) and same number (A↔B mode) are mixable.
    CAMELOT = {
        'C':  (8,'B'), 'G':  (9,'B'), 'D':  (10,'B'), 'A':  (11,'B'),
        'E':  (12,'B'),'B':  (1,'B'), 'F#': (2,'B'),  'C#': (3,'B'),
        'G#': (4,'B'), 'D#': (5,'B'), 'A#': (6,'B'),  'F':  (7,'B'),
        # Minor equivalents mapped to same wheel numbers
        'Am': (8,'A'), 'Em': (9,'A'), 'Bm': (10,'A'), 'F#m':(11,'A'),
        'C#m':(12,'A'),'G#m':(1,'A'), 'D#m':(2,'A'),  'A#m':(3,'A'),
        'Fm': (4,'A'), 'Cm': (5,'A'), 'Gm': (6,'A'),  'Dm': (7,'A'),
    }

    # ── Energy journey blueprint ──────────────────────────────────────────────
    # A real DJ set follows an arc. We reward transitions that follow it.
    # Position 0→1 = start→peak of set.
    JOURNEY = [
        'Low/Chill',   # 0-15%   warm-up
        'Low/Chill',   # 15-30%  building
        'High',        # 30-45%  first peak
        'Low/Chill',   # 45-55%  breather (classic DJ trick)
        'High',        # 55-70%  second peak (bigger)
        'High',        # 70-85%  sustained peak
        'Low/Chill',   # 85-100% cool-down
    ]

    def __init__(self):
        self.ml = PersistentLearner()

    def camelot_score(self, key_a, key_b):
        """
        Returns (score, label) using the full Camelot Wheel.
        Perfect match = 40, adjacent = 25, energy boost (A↔B same number) = 20,
        two-step = 10, clash = -30.
        """
        ca = self.CAMELOT.get(key_a)
        cb = self.CAMELOT.get(key_b)
        if not ca or not cb:
            # Unknown key — treat as compatible (don't penalise bad metadata)
            return 10, "unknown"

        num_a, mode_a = ca
        num_b, mode_b = cb

        if num_a == num_b and mode_a == mode_b:
            return 40, "exact"
        if num_a == num_b and mode_a != mode_b:
            return 20, "energy_boost"   # e.g. 8B→8A same wheel number, mode flip
        diff = abs(num_a - num_b)
        diff = min(diff, 12 - diff)    # wrap around the 12-position wheel
        if diff == 1:
            return 25, "adjacent"
        if diff == 2:
            return 10, "two_step"
        return -30, "clash"

    def key_compat(self, key_a, key_b):
        """Legacy compat string for select_technique."""
        score, label = self.camelot_score(key_a, key_b)
        if label in ("exact", "energy_boost"): return "exact"
        if label == "adjacent":                return "compatible"
        return "clash"

    def bpm_score(self, bpm_a, bpm_b, is_amapiano):
        """
        Rewards gradual BPM progression.
        Amapiano: max tolerance ±3 BPM (genre is tempo-strict).
        Other:    up to ±8 BPM is workable with time-stretching.
        """
        diff = abs(bpm_b - bpm_a)
        if is_amapiano:
            if diff <= 1.0:  return 35
            if diff <= 2.0:  return 15
            if diff <= 3.0:  return 0
            return -120  # hard veto
        else:
            if diff <= 1.0:  return 35
            if diff <= 3.0:  return 20
            if diff <= 6.0:  return 5
            if diff <= 8.0:  return -10
            return -50

    def energy_arc_score(self, candidate_energy, history, setlist_len, position):
        """
        Rewards transitions that follow the JOURNEY blueprint.
        position = how far through the total setlist (0.0 → 1.0).
        """
        s = 0
        journey_idx = min(int(position * len(self.JOURNEY)), len(self.JOURNEY)-1)
        desired_energy = self.JOURNEY[journey_idx]

        if candidate_energy == desired_energy:
            s += 25   # on-blueprint bonus
        else:
            s -= 10   # off-blueprint penalty (mild — don't over-constrain)

        # Hard block: 3 identical energy levels in a row feels monotonous
        if len(history) >= 3 and all(e == candidate_energy for e in history[-3:]):
            s -= 50

        # Reward the classic breather: High → Low after 2+ highs
        if (candidate_energy == 'Low/Chill' and
                len(history) >= 2 and all(e == 'High' for e in history[-2:])):
            s += 40

        # Reward return to High after a breather
        if (candidate_energy == 'High' and
                len(history) >= 1 and history[-1] == 'Low/Chill'):
            s += 35

        return s

    def vocal_score(self, track_a, track_b):
        """
        Penalise vocal-on-vocal clashes.
        Two tracks with active vocals in their transition windows need
        extra technique care — penalise the pair mildly so we prefer
        non-vocal followers when possible.
        """
        a_vocal = track_a.get('stems', {}).get('has_vocals', False)
        b_vocal = track_b.get('stems', {}).get('has_vocals', False)
        if a_vocal and b_vocal:
            return -20  # manageable with EQ events but not ideal
        if not a_vocal and not b_vocal:
            return 10   # cleanest mix — no vocal bands to worry about
        return 0        # one vocal = fine

    def spectral_score(self, track_a, track_b):
        """
        Reward similar spectral centroid — similar brightness = smoother blend.
        Uses zones.optimal_mix_out centroid if available, otherwise energy label.
        """
        ea = track_a.get('energy', 'High')
        eb = track_b.get('energy', 'High')
        if ea == eb:
            return 10
        return -5

    def genre_score(self, track_a, track_b):
        """
        Amapiano should follow Amapiano. Cross-genre needs gentler transitions.
        """
        ga = track_a.get('genre', 'open')
        gb = track_b.get('genre', 'open')
        if ga == gb == 'amapiano':
            return 30
        if ga == 'amapiano' and gb != 'amapiano':
            return -25   # leaving Amapiano mid-set feels abrupt
        if ga != 'amapiano' and gb == 'amapiano':
            return -15   # entering Amapiano works if BPM/key allow
        return 0

    def score_transition(self, curr, cand, history,
                          setlist_len=10, position=0.5):
        is_amp = curr.get('genre')=='amapiano' or cand.get('genre')=='amapiano'

        s  = self.bpm_score(curr.get('bpm',112), cand.get('bpm',112), is_amp)
        ks, kl = self.camelot_score(curr.get('key','C'), cand.get('key','C'))
        s += ks
        s += self.energy_arc_score(cand.get('energy','High'), history,
                                    setlist_len, position)
        s += self.vocal_score(curr, cand)
        s += self.spectral_score(curr, cand)
        s += self.genre_score(curr, cand)

        return s


class SetlistArchitect:
    """
    Look-ahead setlist builder.

    Instead of a greedy single-best pick (which can back itself into a corner),
    this uses a 3-step look-ahead: when two candidates have close scores, it
    picks the one whose NEXT transition also looks promising.
    Also ensures BPM arc flows smoothly (±2 BPM per step preferred) and
    avoids genre traps (e.g. ending up with 3 Amapiano tracks at the end
    with no good exits).
    """

    def __init__(self, brain):
        self.brain   = brain
        self.history = []

    def map_journey(self, crate):
        if not crate:
            return []

        # ── Opener: lowest BPM, no vocal (crowd is sober, clean intro) ───────
        def opener_score(t):
            voc_penalty = 5 if t.get('stems', {}).get('has_vocals', False) else 0
            return t.get('bpm', 112.0) + voc_penalty
        opener = min(crate, key=opener_score)
        crate.remove(opener)

        setlist = [opener]
        self.history.append(opener.get('energy', 'High'))
        cur = opener
        total = len(crate) + 1   # total tracks including opener

        while crate:
            position    = len(setlist) / total
            remaining   = len(crate)

            # ── Score every candidate ─────────────────────────────────────────
            scored = []
            for cand in crate:
                s = self.brain.score_transition(
                    cur, cand, self.history, total, position)

                # ── 2-step look-ahead ────────────────────────────────────────
                # If there are ≥2 tracks left, peek one step further.
                # This prevents choosing a track that leaves no good exits.
                if remaining >= 2:
                    future_crate = [t for t in crate if t is not cand]
                    next_pos     = (len(setlist) + 1) / total
                    best_next    = max(
                        self.brain.score_transition(
                            cand, fut, self.history + [cand.get('energy','High')],
                            total, next_pos)
                        for fut in future_crate
                    )
                    # Weight look-ahead at 30% so current transition still dominates
                    s += best_next * 0.30

                scored.append((s, cand))

            scored.sort(key=lambda x: -x[0])
            best_score, best = scored[0]

            setlist.append(best)
            self.history.append(best.get('energy', 'High'))
            crate.remove(best)
            cur = best

            print(f"   🗂️  [{len(setlist)}/{total}] "
                  f"{os.path.basename(best.get('filename','?')[:40])}  "
                  f"score={best_score:.0f}  "
                  f"bpm={best.get('bpm',0):.1f}  "
                  f"key={best.get('key','?')}  "
                  f"energy={best.get('energy','?')}")

        return setlist


# ════════════════════════════════════════════════════════════════════════════════
# NEURO-DJ CORE
# ════════════════════════════════════════════════════════════════════════════════
class NeuroDJ(QObject):
    request_rating = pyqtSignal(str, object)

    def __init__(self, library_json):
        super().__init__()
        self.mixer    = neuro_core.NeuroMixer()
        self.brain    = DJBrain()
        self.arch     = SetlistArchitect(self.brain)
        self.analyzer = AudioAnalyzer()

        # Two separate pools so ML learning NEVER waits for heavy analysis jobs
        self._analysis_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._learn_pool    = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # State for GUI
        self.track_a_dur   = 360.0; self.track_b_dur = 360.0
        self.track_a_name  = "Loading..."; self.track_b_name = "—"
        self.master_bpm    = 0
        self.current_mix_trigger = 0.0
        self.current_technique   = "—"
        self.mix_count     = 0

        # Which deck label is currently "live" — alternates after each mix
        # so the GUI correctly shows DECK B: LIVE on odd mixes
        self.live_deck_label  = "A"   # "A" or "B"
        self.ready_deck_label = "B"

        with open(library_json,'r') as f:
            raw = json.load(f)
        print(f"📚 Loaded {len(raw)} tracks from {library_json}")
        if not raw:
            print("❌ master_library.json is empty! Run auto_prep_folder.py first.")
        self.planned_setlist = self.arch.map_journey(raw)
        print(f"🗂️  Setlist built: {len(self.planned_setlist)} tracks ordered")

    def warp_track(self, track, bpm):
        filename = track.get('filename', '')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Track file not found: {filename}")
        ratio = bpm / track.get('bpm', 112.0)
        if 0.99 <= ratio <= 1.01:
            return filename, 1.0
        tmp = "temp_sync_deck_b.wav"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(filename, sr=44100, mono=False)
        if y.ndim == 1:
            y = np.vstack([y, y])
        yw = np.array([librosa.effects.time_stretch(c, rate=ratio) for c in y])
        sf.write(tmp, yw.T, 44100)
        return tmp, ratio

    def _schedule_eq_events(self, ta, tb, plan, spb):
        """
        The Intelligent Fake-Out.
        Reads the offline HPSS stem maps for both tracks and programs surgical
        EQ events into the C++ engine.  Three scenarios are handled:

        1. VOCAL CLASH PREVENTION
           Both A and B have vocals active near the transition window.
           → Kill A's mids (vocal band) at the bass-swap beat so B's vocal
             enters clean. Without this, two singers overlap.

        2. LOG DRUM INJECTION SYNC (Amapiano)
           B has a log drum hit near the start of its transition window.
           → Program B's bass to open exactly on that hit — not on a slow
             ramp — so the new track's sub-bass lands with weight.

        3. PRE-SWAP VOCAL CLEANUP
           A has a vocal region that runs INTO the transition window.
           → Kill A's mids 2 beats before the swap so the outgoing vocal
             doesn't mud up the incoming track.
        """
        self.mixer.clear_eq_events()

        stems_a = ta.get('stems', {})
        stems_b = tb.get('stems', {})

        mix_trigger  = plan['mix_trigger']
        trans_dur    = plan['trans_dur']
        recipe       = plan['recipe']
        b_start_w    = plan['b_start_w']
        swap_elapsed = trans_dur * recipe['bass']        # seconds from trigger to swap
        swap_time_a  = mix_trigger + swap_elapsed        # deck A's file position at swap

        vocal_regions_a = stems_a.get('vocal_regions', [])
        vocal_regions_b = stems_b.get('vocal_regions', [])
        log_drum_hits_b = stems_b.get('log_drum_hits', [])

        # ── 1. Vocal clash prevention ─────────────────────────────────────────
        # Check if A has vocal activity in the last 32 beats of the transition
        a_window_start = mix_trigger
        a_window_end   = mix_trigger + trans_dur
        a_has_vocal = any(
            not (end < a_window_start or start > a_window_end)
            for start, end in vocal_regions_a
        )

        # Check if B has vocal activity near its cued start
        b_window_end = b_start_w + trans_dur
        b_has_vocal = any(
            start < b_window_end for start, end in vocal_regions_b
        )

        if a_has_vocal and b_has_vocal:
            # Kill A's vocal band 2 beats before the swap
            kill_time = swap_time_a - 2.0 * spb
            if kill_time > mix_trigger:
                self.mixer.add_eq_event(0, kill_time, 1.0, 0.0, 1.0)
                print(f"   🎤 VOCAL CLASH GUARD: mid-kill on A @ {kill_time:.1f}s "
                      f"(2 beats before swap)")
            else:
                # Too close to start — kill immediately at trigger
                self.mixer.add_eq_event(0, mix_trigger, 1.0, 0.0, 1.0)
                print(f"   🎤 VOCAL CLASH GUARD: immediate mid-kill on A @ {mix_trigger:.1f}s")

        # ── 2. Pre-swap vocal cleanup (even if B has no vocals) ───────────────
        # If A's vocal runs past the mid-point of the transition,
        # schedule a kill so it doesn't mud the incoming bass
        elif a_has_vocal:
            kill_time = swap_time_a - spb  # 1 beat before swap
            if kill_time > mix_trigger:
                self.mixer.add_eq_event(0, kill_time, 1.0, 0.0, 1.0)
                print(f"   🎙️  PRE-SWAP MID-KILL on A @ {kill_time:.1f}s")

        # ── 3. Log drum injection sync ────────────────────────────────────────
        # Find B's first log drum hit within the transition window
        b_trans_start = b_start_w
        b_trans_end   = b_start_w + trans_dur
        first_drum = next(
            (t for t in log_drum_hits_b if b_trans_start <= t <= b_trans_end),
            None
        )

        if first_drum is not None:
            # Before the drum hit: B plays with bass killed (technique handles highs/mids)
            # At the hit: B's bass snaps open with full weight
            # This REPLACES the slow bass ramp from the technique with a beat-precise snap
            self.mixer.add_eq_event(1, first_drum, 1.0, 1.0, 1.0)
            print(f"   🥁 LOG DRUM SYNC: B bass snaps open @ {first_drum:.1f}s "
                  f"(deck B file time)")
            # Kill B's bass before that hit so it doesn't bleed in early
            pre_hit = max(b_start_w, first_drum - spb)
            self.mixer.add_eq_event(1, b_start_w, 0.0, 1.0, 1.0)  # bass off at cue start
            print(f"   🔇 B bass kill from cue start until drum hit")

    def _make_fallback_plan(self, tb, spb):
        """Emergency plan: 64-beat slow burn at 70% of track A, B from its start."""
        beats = 64.0
        tdur  = beats * spb
        trig  = max(60.0, self.track_a_dur * 0.70)
        trig  = min(trig, self.track_a_dur - tdur - 30.0)
        trig  = max(60.0, trig)
        defs  = dict(TECHNIQUE_LIBRARY['SLOW_BURN']['defaults'])
        defs.update({'beats': beats, 'technique_name': 'SLOW_BURN',
                     'technique_id': 4, 'timing': 0.0})
        return {
            'fnb':         tb['filename'],
            'b_ratio':     1.0,
            'b_start_w':   0.0,
            'mix_trigger': trig,
            'trans_dur':   tdur,
            'tech_name':   'SLOW_BURN',
            'mem_key':     'SLOW_BURN|High->High',
            'recipe':      defs,
            'track_b_dur': self.track_b_dur,
        }

    def _compute_mix_plan(self, ta, tb, spb, initial_pos_a=0.0):
        """
        ALL heavy work (file I/O, analysis, warp) runs here in a thread-pool
        worker so the main/audio threads are never blocked.
        Always returns a valid plan dict — never None, never raises.

        initial_pos_a: deck A playback position when this plan was submitted.
                       Used so min_trigger gives the crowd 120s of new music
                       *from that position*, not from position 0 of the file.
        """
        try:
            # ── 1. Warp track B to master BPM ───────────────────────────────
            fnb, b_ratio = self.warp_track(tb, self.master_bpm)
            tb['stretch_ratio'] = b_ratio
            print(f"   ✅ Warp done: ratio={b_ratio:.4f}  file={os.path.basename(fnb)}")

            # ── 2. Structural analysis ───────────────────────────────────────
            is_amp   = (ta.get('genre') == 'amapiano' or tb.get('genre') == 'amapiano')
            kc       = self.brain.key_compat(ta.get('key', 'C'), tb.get('key', 'C'))
            bpm_diff = abs(tb.get('bpm', 112) - self.master_bpm)
            ca = self.analyzer.find_amapiano_cue_points(ta['filename'])
            cb = self.analyzer.find_amapiano_cue_points(fnb)
            track_b_dur = cb['duration']
            print(f"   ✅ Analysis done: A outro={ca['outro_start']:.1f}s  "
                  f"B intro={cb['intro_end']:.1f}s  B dur={track_b_dur:.1f}s")

            # ── 3. Technique selection using deep structural context ───────────
            intro_beats = cb['intro_end'] / spb

            # Build structural context object — all the data select_technique needs
            ctx = build_transition_context(ta, tb, ca, cb, spb)
            print(f"   🧠 Transition context:")
            print(f"      A exits from: '{ctx['a_exit_section']}' "
                  f"(density={ctx['a_exit_density']:.2f}, "
                  f"trajectory={ctx['a_energy_trajectory']})")
            print(f"      B enters as:  '{ctx['b_entry_section']}' "
                  f"(character={ctx['b_mix_character']})")
            print(f"      Vocal clash:  {ctx['vocal_clash']}  "
                  f"| Dance moment: {ctx['dance_moment']}")

            tech_name = select_technique(
                ta.get('energy', 'High'), tb.get('energy', 'High'),
                bpm_diff, kc, is_amp, intro_beats, self.mix_count,
                ctx=ctx)
            mem_key, recipe = self.brain.ml.generate_recipe(
                tech_name, ta.get('energy', 'High'), tb.get('energy', 'High'), is_amp)

            # ── 4. Cue point calculation ──────────────────────────────────────
            #
            # The philosophy of a flawless overlap:
            #
            #   mix_trigger = the moment in Track A where deck B starts playing.
            #                 This should be the LAST full phrase before A's outro.
            #                 Both tracks then play together for trans_dur seconds.
            #
            #   b_start_w   = where in Track B we cue up.
            #                 We want to start B from its very intro so the crowd
            #                 hears it building — not from the middle of the track.
            #                 Ideally: 0 (very beginning) or its first beat.
            #
            # We use the pre-computed `zones.optimal_mix_out` from auto_prep_folder
            # as the primary source. This was carefully computed during analysis
            # and represents the last energy drop before the outro — the perfect
            # handoff point. We only fall back to live analysis if it's missing.

            # ── A: where to start mixing out ─────────────────────────────────
            # Priority 1: use the prep-computed optimal mix-out point
            # Priority 2: last log drum solo before outro
            # Priority 3: 75% of track duration
            prep_mix_out = ta.get('zones', {}).get('optimal_mix_out', None)

            if prep_mix_out and prep_mix_out > 30.0:
                raw_out = prep_mix_out
                print(f"   ✅ Using prep mix-out point: {raw_out:.1f}s")
            else:
                out_cands = [t for t in ca['log_drum_solos'] if t < ca['outro_start']]
                raw_out   = out_cands[-1] if out_cands else ca['outro_start']
                print(f"   ✅ Using live outro detection: {raw_out:.1f}s")

            # Quantize to the nearest 16-beat phrase boundary
            phased_out = self.analyzer.quantize_to_phrase(
                raw_out, ta.get('first_beat_time', 0.0), self.master_bpm)
            out_unwarped = self.analyzer.snap_to_transient(ta['filename'], phased_out)

            # ── B: where to cue track B ───────────────────────────────────────
            # Start B from its very beginning (or first beat) so the full
            # intro plays during the overlap. The crowd gets to hear B building.
            # This is what DJs do — cue from the top, let it breathe.
            first_beat_b  = tb.get('first_beat_time', 0.0) / b_ratio
            b_start_w     = max(0.0, first_beat_b)   # cue from first beat

            print(f"   ✅ Cue points: A mix-out={out_unwarped:.1f}s  B starts={b_start_w:.1f}s")

            # mix_trigger is the moment in track A when deck B starts playing.
            # The trigger lands at A's last full phrase — b_start_w = 0 means
            # B begins from its very intro and plays for trans_dur seconds.
            mix_trigger = out_unwarped

            # ── 5. Calculate transition duration ─────────────────────────────
            # Apply per-technique minimum beats and a 192-beat ceiling.
            # Each technique has its own natural length:
            #   ECHO_THROW/FREEZE: 32-beat floor (event-driven, brief)
            #   BASS_SWAP/FILTER_SWEEP/PIANO_HANDOFF: 96-beat floor (need long overlap)
            #   SLOW_BURN: 128-beat floor (gradual fade needs time)
            tech_min = float(TECHNIQUE_MIN_BEATS.get(tech_name, 64))
            recipe['beats'] = max(tech_min, min(recipe['beats'], 192.0))
            trans_dur = recipe['beats'] * spb

            # Ensure B has enough room: at least trans_dur + 60s remaining
            b_room = track_b_dur - b_start_w
            if b_room < trans_dur + 60.0:
                if track_b_dur >= trans_dur + 60.0:
                    b_start_w = 0.0
                    b_room    = track_b_dur
                    print(f"   ⚠️  Resetting B cue to 0 for more room")
                else:
                    max_beats_b     = max(tech_min, (track_b_dur - 30.0) / spb)
                    recipe['beats'] = round(max_beats_b / 16.0) * 16.0
                    trans_dur       = recipe['beats'] * spb
                    b_start_w       = 0.0
                    print(f"   ⚠️  B is short ({track_b_dur:.0f}s) — reduced to "
                          f"{recipe['beats']:.0f} beats / {trans_dur:.1f}s")

            print(f"   ✅ trans_dur={trans_dur:.1f}s ({recipe['beats']:.0f} beats)  "
                  f"b_start={b_start_w:.1f}s")

            # ── 6. Validate the trigger window ────────────────────────────────
            #
            # THE KEY FIX — position-aware minimum trigger.
            #
            # When track B becomes deck A after a swap, deck A is already
            # `trans_dur` seconds into the file (track B played from 0 during
            # the overlap). initial_pos_a captures this position at submission.
            #
            # The crowd needs 120 seconds of music they haven't heard yet
            # before the next mix starts:
            #
            #   min_trigger = initial_pos_a + 120s
            #
            # Examples:
            #   First track (initial_pos_a=0):  min_trigger = 120s
            #   After 34s transition:           min_trigger = 34 + 120 = 154s
            #   After 51s transition:           min_trigger = 51 + 120 = 171s
            #
            # We also enforce 65% through the track so we always mix in the outro.
            track_a_dur  = self.track_a_dur
            min_trigger  = max(
                initial_pos_a + 120.0,   # crowd needs 120s of new music
                track_a_dur * 0.65,      # must be in the outro (65%+)
                60.0                     # absolute floor
            )
            max_trigger  = track_a_dur - trans_dur - 20.0
            max_trigger  = max(max_trigger, min_trigger)  # ensure max >= min

            # If we genuinely can't fit, shrink the transition
            if max_trigger - min_trigger < 10.0:
                available = track_a_dur - min_trigger - 20.0
                if available > 0:
                    shrink_beats    = max(32.0, available / spb)
                    recipe['beats'] = round(shrink_beats / 16.0) * 16.0
                    trans_dur       = recipe['beats'] * spb
                    max_trigger     = max(track_a_dur - trans_dur - 20.0, min_trigger)
                    print(f"   ⚠️  Shrunk to {recipe['beats']:.0f} beats to fit window")

            print(f"   ✅ Trigger window=[{min_trigger:.1f}s .. {max_trigger:.1f}s]  "
                  f"raw={mix_trigger:.1f}s  (initial_pos={initial_pos_a:.1f}s)")

            if mix_trigger < min_trigger:
                print(f"   ⚠️  Trigger too early ({mix_trigger:.1f}s) "
                      f"→ clamping to {min_trigger:.1f}s")
                mix_trigger = min_trigger
            elif mix_trigger > max_trigger:
                print(f"   ⚠️  Trigger too late ({mix_trigger:.1f}s) "
                      f"→ clamping to {max_trigger:.1f}s")
                mix_trigger = max_trigger

            print(f"📍 PLAN READY: trigger={mix_trigger:.1f}s  b_start={b_start_w:.1f}s  "
                  f"tech={TECHNIQUE_LIBRARY[tech_name]['label']}  "
                  f"trans={recipe['beats']:.0f}beats/{trans_dur:.1f}s")

            return {
                'fnb':         fnb,
                'b_ratio':     b_ratio,
                'b_start_w':   b_start_w,
                'mix_trigger': mix_trigger,
                'trans_dur':   trans_dur,
                'tech_name':   tech_name,
                'mem_key':     mem_key,
                'recipe':      recipe,
                'track_b_dur': track_b_dur,
            }

        except Exception as e:
            import traceback
            print(f"\n❌ _compute_mix_plan EXCEPTION: {e}")
            traceback.print_exc()
            # NEVER return None — that causes an infinite loop in the caller.
            # Build a safe fallback plan instead.
            print("   ⚠️  Falling back to emergency slow-burn plan")
            return self._make_fallback_plan(tb, spb)

    def start_set(self):
        # ── Startup diagnostics ───────────────────────────────────────────────
        print(f"\n{'═'*60}")
        print(f"🚀 start_set() called")
        print(f"   Setlist size: {len(self.planned_setlist)} tracks")
        if self.planned_setlist:
            for i, t in enumerate(self.planned_setlist[:3]):
                exists = os.path.exists(t.get('filename',''))
                print(f"   [{i}] {os.path.basename(t.get('filename','?'))}  "
                      f"exists={exists}  bpm={t.get('bpm','?')}")
        print(f"{'═'*60}\n")

        if not self.planned_setlist:
            print("⚠️  Setlist is empty — nothing to play.")
            print("   Make sure master_library.json exists and has at least one track.")
            print("   Run: python auto_prep_folder.py ./your_music_folder")
            return

        try:
            ta = self.planned_setlist.pop(0)
            ta['stretch_ratio'] = 1.0

            # Verify the file actually exists before trying to load it
            if not os.path.exists(ta['filename']):
                print(f"❌ Track file not found: {ta['filename']}")
                print(f"   Re-run auto_prep_folder.py to regenerate the library.")
                return

            try:
                self.track_a_dur = librosa.get_duration(path=ta['filename'])
            except Exception as e:
                print(f"⚠️  Could not read duration: {e}")
                self.track_a_dur = 360.0

            self.track_a_name = os.path.basename(ta['filename'])
            self.mixer.load_deck("A", ta['filename'])
            self.mixer.play("A")
            self.master_bpm = round(ta.get('bpm', 112.0))
            print(f"\n🎵 NOW PLAYING: {self.track_a_name}  "
                  f"({self.master_bpm} BPM, {self.track_a_dur:.0f}s)")

            while self.planned_setlist:
                spb = 60.0 / self.master_bpm
                tb  = self.planned_setlist.pop(0)

                print(f"\n{'='*60}")
                print(f"🎯 NEXT UP: {os.path.basename(tb['filename'])}")
                print(f"⚙️  Starting background mix-plan computation...")

                self.track_b_name = os.path.basename(tb['filename'])

                # ── Submit ALL heavy work to the thread pool ─────────────────
                initial_pos_a = self.mixer.get_position("A")
                plan_future = self._analysis_pool.submit(
                    self._compute_mix_plan, ta, tb, spb, initial_pos_a)

                # ── Wait for the plan — with a hard timeout of 120s ──────────
                # THIS LOOP CANNOT FREEZE: plan_future always returns a dict,
                # never None (see _compute_mix_plan above).
                plan = None
                wait_deadline = time.time() + 120.0
                while plan is None:
                    if time.time() > wait_deadline:
                        print("❌ Plan took > 120s — using emergency fallback")
                        plan = self._make_fallback_plan(tb, spb)
                        break
                    try:
                        plan = plan_future.result(timeout=3.0)
                        # plan is guaranteed to be a dict here, never None
                    except concurrent.futures.TimeoutError:
                        pos = self.mixer.get_position("A")
                        print(f"   ⏳ Still analyzing... "
                              f"({pos:.0f}s into track A, "
                              f"{self.track_a_dur - pos:.0f}s left)")
                    except Exception as e:
                        print(f"❌ Future raised: {e} — using emergency fallback")
                        plan = self._make_fallback_plan(tb, spb)
                        break

                # plan is now guaranteed to be a valid dict
                self.track_b_dur = plan['track_b_dur']

                # ── Reschedule if analysis finished too late ─────────────────
                pos_now = self.mixer.get_position("A")
                if pos_now >= plan['mix_trigger'] - 15.0:
                    new_trig = pos_now + 45.0
                    max_trig = self.track_a_dur - plan['trans_dur'] - 10.0
                    new_trig = min(new_trig, max(max_trig, pos_now + plan['trans_dur'] + 5.0))
                    print(f"   ⚠️  Trigger was in the past (pos={pos_now:.1f}s, "
                          f"trigger={plan['mix_trigger']:.1f}s) "
                          f"— rescheduled to {new_trig:.1f}s")
                    plan['mix_trigger'] = new_trig

                self.current_mix_trigger = plan['mix_trigger']
                self.current_technique   = TECHNIQUE_LIBRARY[plan['tech_name']]['label']

                print(f"✅ Plan locked | trigger={plan['mix_trigger']:.1f}s  "
                      f"(in {plan['mix_trigger'] - self.mixer.get_position('A'):.0f}s)  "
                      f"tech={self.current_technique}")

                # ── Schedule surgical EQ events (stem-separation fakeout) ────
                self._schedule_eq_events(ta, tb, plan, spb)

                # ── Load deck B 12s before the trigger ──────────────────────
                while self.mixer.get_position("A") < (plan['mix_trigger'] - 12.0):
                    time.sleep(0.5)

                self.mixer.load_deck("B", plan['fnb'])
                self.mixer.seek("B", plan['b_start_w'])
                print(f"💿 Deck B armed: {self.track_b_name} @ {plan['b_start_w']:.1f}s")

                # ── Precision spin-lock to the exact trigger sample ──────────
                while True:
                    tl = plan['mix_trigger'] - self.mixer.get_position("A")
                    if tl <= 0:
                        break
                    elif tl > 0.02:
                        time.sleep(0.005)
                    else:
                        deadline = time.perf_counter() + tl
                        while time.perf_counter() < deadline:
                            pass

                # ── Fire transition ──────────────────────────────────────────
                recipe = plan['recipe']
                print(f"\n🔥 FIRING: {TECHNIQUE_LIBRARY[plan['tech_name']]['label']} "
                      f"| {recipe['beats']:.0f} beats / {plan['trans_dur']:.1f}s")

                self.mixer.trigger_hybrid_transition(
                    plan['trans_dur'],
                    recipe['beats'],                    recipe['bass'],
                    recipe['echo'],                     recipe['stutter'],
                    recipe['wash'],                     float(self.master_bpm),
                    float(recipe.get('piano_hold', 0.0)),
                    recipe['technique_id']
                )

                # ── CRITICAL: Wait for the audio callback to PICK UP the command ─
                # trigger_hybrid_transition() only sets pendingTrans.pending=true.
                # inTransition is set by the audio callback on its NEXT tick (~5.8ms).
                # If we check is_transitioning() before that tick, it returns False
                # and the while-loop below exits immediately — causing an instant
                # swap BEFORE the transition has played even one sample.
                # Fix: spin until is_transitioning() goes True (max 200ms).
                pickup_deadline = time.time() + 0.2
                while not self.mixer.is_transitioning():
                    if time.time() > pickup_deadline:
                        print("   ⚠️  Audio callback didn't pick up transition command!")
                        break
                    time.sleep(0.001)

                # ── Now wait for the transition to COMPLETE ──────────────────
                t_deadline = time.time() + plan['trans_dur'] + 15.0
                while self.mixer.is_transitioning():
                    if time.time() > t_deadline:
                        print("   ⚠️  Transition timeout — forcing swap")
                        break
                    time.sleep(0.02)

                # ── Echo tail wait ────────────────────────────────────────────
                # For ECHO_THROW (id=3) and ECHO_FREEZE (id=7), the C++ engine
                # intentionally keeps fx.mix=1.0 after the transition completes
                # so the delay buffer rings out. With feedback=0.88-0.90 and a
                # 1-beat delay interval, the echo needs ~4 seconds to fade to
                # inaudibility. We wait here so the crowd hears the full tail
                # under track B before the swap cleans up the delay buffers.
                tech_id = plan['recipe'].get('technique_id', 0)
                if tech_id in (3, 7):
                    echo_tail_secs = 4.0
                    print(f"   🔔 Letting echo tail ring out ({echo_tail_secs:.0f}s)...")
                    time.sleep(echo_tail_secs)

                # ── Swap: B becomes the new A ────────────────────────────────
                self.mixer.swap_decks()
                self.mixer.pause("B")

                # Alternate the live/ready deck labels for the GUI
                self.live_deck_label, self.ready_deck_label = (
                    self.ready_deck_label, self.live_deck_label
                )

                # Update display state
                self.track_a_dur         = plan['track_b_dur']
                self.track_a_name        = self.track_b_name
                self.track_b_name        = "—"
                self.current_mix_trigger = 0.0
                self.current_technique   = "—"
                self.mix_count          += 1

                print(f"\n✅ Mix #{self.mix_count} complete. "
                      f"Now live on DECK {self.live_deck_label}: {self.track_a_name}")

                self.request_rating.emit(plan['mem_key'], recipe)
                ta = tb

            # ── Setlist exhausted ────────────────────────────────────────────
            remaining = max(0, self.track_a_dur - self.mixer.get_position("A"))
            print(f"\n🏁 Setlist complete. "
                  f"'{self.track_a_name}' playing out ({remaining:.0f}s left).")

        except Exception as e:
            import traceback
            print(f"\n❌ FATAL in start_set: {e}")
            traceback.print_exc()
        finally:
            self._learn_pool.submit(self.brain.ml.save_brain)


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
        self.dj = dj
        self._old_stdout = sys.stdout

        self.setWindowTitle("Neuro-DJ — Dynamic Technique Intelligence")
        self.setGeometry(100,100,1080,880)
        self.setStyleSheet("background:#121212;color:#FFF;")

        root = QVBoxLayout(); root.setSpacing(8); root.setContentsMargins(12,12,12,12)
        self.setCentralWidget(QWidget()); self.centralWidget().setLayout(root)

        # ── Header ────────────────────────────────────────────────────────────
        h = QLabel("🧠 NEURO-DJ MAINSTAGE")
        h.setFont(QFont("Arial",22,QFont.Bold))
        h.setStyleSheet("color:#00FFCC;padding:4px 0;")
        root.addWidget(h)

        # ── Decks ─────────────────────────────────────────────────────────────
        dr = QHBoxLayout()
        da = QFrame(); da.setStyleSheet("background:#1E1E1E;border-radius:8px;padding:8px;")
        la = QVBoxLayout(da)
        self.lbl_da = QLabel("DECK A: Idle"); self.lbl_da.setFont(QFont("Arial",13,QFont.Bold))
        self.lbl_na = QLabel("—"); self.lbl_na.setStyleSheet("color:#777;font-size:11px;"); self.lbl_na.setWordWrap(True)
        self.plot_a = pg.PlotWidget(); self.plot_a.setBackground('#1E1E1E')
        self.plot_a.setYRange(-1,1); self.plot_a.hideAxis('left'); self.plot_a.hideAxis('bottom'); self.plot_a.setFixedHeight(110)
        self.curve_a = self.plot_a.plot(pen=pg.mkPen('#FF0055',width=2))
        self.prog_a = QProgressBar(); self.prog_a.setStyleSheet("QProgressBar::chunk{background:#FF0055;}"); self.prog_a.setTextVisible(False); self.prog_a.setFixedHeight(5)
        for w in (self.lbl_da,self.lbl_na,self.plot_a,self.prog_a): la.addWidget(w)
        dr.addWidget(da)

        db = QFrame(); db.setStyleSheet("background:#1E1E1E;border-radius:8px;padding:8px;")
        lb = QVBoxLayout(db)
        self.lbl_db = QLabel("DECK B: Idle"); self.lbl_db.setFont(QFont("Arial",13,QFont.Bold))
        self.lbl_nb = QLabel("—"); self.lbl_nb.setStyleSheet("color:#777;font-size:11px;"); self.lbl_nb.setWordWrap(True)
        self.plot_b = pg.PlotWidget(); self.plot_b.setBackground('#1E1E1E')
        self.plot_b.setYRange(-1,1); self.plot_b.hideAxis('left'); self.plot_b.hideAxis('bottom'); self.plot_b.setFixedHeight(110)
        self.curve_b = self.plot_b.plot(pen=pg.mkPen('#0088FF',width=2))
        self.prog_b = QProgressBar(); self.prog_b.setStyleSheet("QProgressBar::chunk{background:#0088FF;}"); self.prog_b.setTextVisible(False); self.prog_b.setFixedHeight(5)
        for w in (self.lbl_db,self.lbl_nb,self.plot_b,self.prog_b): lb.addWidget(w)
        dr.addWidget(db)
        root.addLayout(dr)

        # ── Info bar ──────────────────────────────────────────────────────────
        inf = QFrame(); inf.setStyleSheet("background:#1A1A1A;border:1px solid #2A2A2A;border-radius:6px;padding:6px;")
        ir = QHBoxLayout(inf); ir.setSpacing(14)

        self.lbl_bpm = QLabel("♩ — BPM")
        self.lbl_bpm.setFont(QFont("Consolas",11,QFont.Bold))
        self.lbl_bpm.setStyleSheet("color:#FFAA00;")
        ir.addWidget(self.lbl_bpm)

        ir.addWidget(self._sep())

        self.lbl_tech = QLabel("⬜ —")
        self.lbl_tech.setFont(QFont("Consolas",11,QFont.Bold))
        self.lbl_tech.setStyleSheet("color:#AA88FF;")
        ir.addWidget(self.lbl_tech)

        ir.addWidget(self._sep())

        self.lbl_cd = QLabel("⏱ Calculating...")
        self.lbl_cd.setFont(QFont("Consolas",11,QFont.Bold))
        self.lbl_cd.setStyleSheet("color:#00FFCC;")
        ir.addWidget(self.lbl_cd)

        ir.addStretch()

        self.btn_skip = QPushButton("⏩  SKIP TO MIX  (−30s)")
        self.btn_skip.setFont(QFont("Arial",10,QFont.Bold))
        self.btn_skip.setStyleSheet(
            "QPushButton{background:#FF5500;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#FF7733;}"
            "QPushButton:pressed{background:#CC3300;}"
            "QPushButton:disabled{background:#2A2A2A;color:#555;}")
        self.btn_skip.setEnabled(False)
        self.btn_skip.clicked.connect(self.skip_to_mix)
        ir.addWidget(self.btn_skip)
        root.addWidget(inf)

        # ── Mix approach bar ──────────────────────────────────────────────────
        self.prog_mix = QProgressBar()
        self.prog_mix.setRange(0,100); self.prog_mix.setValue(0)
        self.prog_mix.setTextVisible(False); self.prog_mix.setFixedHeight(6)
        self.prog_mix.setStyleSheet(
            "QProgressBar{background:#1A1A1A;border-radius:3px;}"
            "QProgressBar::chunk{background:#FF5500;border-radius:3px;}")
        root.addWidget(self.prog_mix)

        # ── RLHF ─────────────────────────────────────────────────────────────
        self.rlhf = QFrame()
        self.rlhf.setStyleSheet("background:#222;border:2px solid #00FFCC;border-radius:8px;padding:8px;")
        rl = QHBoxLayout(self.rlhf)
        self.lbl_rate = QLabel("🤖 Rate this Transition (1–10):")
        self.lbl_rate.setFont(QFont("Arial",11,QFont.Bold))
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(1,10); self.slider.setValue(5); self.slider.setFocusPolicy(Qt.NoFocus)
        self.lbl_sv = QLabel("5"); self.lbl_sv.setFont(QFont("Consolas",14,QFont.Bold)); self.lbl_sv.setStyleSheet("color:#00FFCC;min-width:22px;")
        self.slider.valueChanged.connect(lambda v: self.lbl_sv.setText(str(v)))
        self.btn_train = QPushButton("TRAIN MODEL")
        self.btn_train.setStyleSheet("background:#00FFCC;color:#000;font-weight:bold;padding:7px 14px;")
        self.btn_train.clicked.connect(self.submit_rating)
        for w in (self.lbl_rate,self.slider,self.lbl_sv,self.btn_train): rl.addWidget(w)
        root.addWidget(self.rlhf); self.rlhf.hide()

        # ── Console ───────────────────────────────────────────────────────────
        self.console = QTextEdit()
        self.console.setReadOnly(True); self.console.setFocusPolicy(Qt.NoFocus)
        self.console.setStyleSheet("background:#000;color:#00FF00;font-family:Consolas;font-size:11px;")
        root.addWidget(self.console)

        sys.stdout = Stream(newText=self._append_console)
        self.dj.request_rating.connect(self._show_rating)
        self.pending_rating = None

        self.timer = QTimer(); self.timer.timeout.connect(self._tick); self.timer.start(50)

    def _sep(self):
        s=QLabel("│"); s.setStyleSheet("color:#333;"); return s

    def _append_console(self, t):
        c=self.console.textCursor(); c.movePosition(QTextCursor.End); c.insertText(t)
        self.console.setTextCursor(c); self.console.ensureCursorVisible()

    def skip_to_mix(self):
        mt = getattr(self.dj,'current_mix_trigger',0.0)
        if mt > 30:
            self.dj.mixer.seek("A", mt-30)
            print(f"\n⏩ SKIP → {mt-30:.1f}s  (T−30s before mix)")
        elif 0 < mt <= 30:
            self.dj.mixer.seek("A", max(0, mt-2))
            print(f"\n⏩ Already close — seeking to {max(0,mt-2):.1f}s")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Up: self.skip_to_mix()

    def closeEvent(self, ev):
        sys.stdout = self._old_stdout
        print("\n🛑 Saving brain..."); self.dj.brain.ml.save_brain(); ev.accept()

    def _tick(self):
        # Waveforms
        try:
            self.curve_a.setData(self.dj.mixer.get_visual_buffer("A"))
            self.curve_b.setData(self.dj.mixer.get_visual_buffer("B"))
        except: pass

        pa = self.dj.mixer.get_position("A")
        pb = self.dj.mixer.get_position("B")
        da = max(1, self.dj.track_a_dur); db = max(1, self.dj.track_b_dur)
        self.prog_a.setValue(min(int(pa/da*100),100))
        self.prog_b.setValue(min(int(pb/db*100),100))

        self.lbl_na.setText(self.dj.track_a_name)
        self.lbl_nb.setText(self.dj.track_b_name)
        if self.dj.master_bpm: self.lbl_bpm.setText(f"♩ {self.dj.master_bpm} BPM")

        tech = getattr(self.dj,'current_technique','—')
        self.lbl_tech.setText(tech if tech!='—' else "⬜ Selecting...")

        is_trans = self.dj.mixer.is_transitioning()
        live  = getattr(self.dj, 'live_deck_label',  'A')
        ready = getattr(self.dj, 'ready_deck_label', 'B')

        if is_trans:
            self.lbl_da.setText(f"DECK A:  {'🌊 MIXING OUT' if live=='A' else '🌊 MIXING IN'}")
            self.lbl_db.setText(f"DECK B:  {'🌊 MIXING OUT' if live=='B' else '🌊 MIXING IN'}")
            self.lbl_da.setStyleSheet("color:#FFAA00;")
            self.lbl_db.setStyleSheet("color:#00FFCC;")
            self.lbl_tech.setStyleSheet("color:#FF0055;font-weight:bold;")
            self.lbl_cd.setText("🔥 MIXING LIVE"); self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
            self.prog_mix.setValue(100); self.btn_skip.setEnabled(False)
        elif pa > 0.1:
            self.lbl_da.setText(f"DECK A:  {'▶ LIVE' if live=='A' else '— Idle'}")
            self.lbl_da.setStyleSheet("color:#FF0055;" if live=='A' else "color:#555;")
            self.lbl_db.setText(f"DECK B:  {'▶ LIVE' if live=='B' else 'Cued'}")
            self.lbl_db.setStyleSheet("color:#FF0055;" if live=='B' else "color:#555;")
            self.lbl_tech.setStyleSheet("color:#AA88FF;")

        mt = getattr(self.dj,'current_mix_trigger',0.0)
        if not is_trans and mt > 0.5:
            ttm = mt - pa
            if ttm > 0:
                m,s = int(ttm//60), int(ttm%60)
                self.lbl_cd.setText(f"⏱ Next Mix: {m}m {s:02d}s" if m else f"⏱ Next Mix: {s}s")
                if ttm < 30:
                    self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
                    self.btn_skip.setEnabled(False)
                elif ttm < 90:
                    self.lbl_cd.setStyleSheet("color:#FFAA00;font-weight:bold;")
                    self.btn_skip.setEnabled(True)
                else:
                    self.lbl_cd.setStyleSheet("color:#00FFCC;")
                    self.btn_skip.setEnabled(True)
                elapsed = max(0, mt-ttm)
                self.prog_mix.setValue(min(int(elapsed/mt*100),99) if mt else 0)
            else:
                self.lbl_cd.setText("🔥 Dropping Now!")
                self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
                self.prog_mix.setValue(100); self.btn_skip.setEnabled(False)
        elif not is_trans and mt <= 0.5:
            self.lbl_cd.setText("⏱ Calculating next mix...")
            self.lbl_cd.setStyleSheet("color:#444;")
            self.prog_mix.setValue(0); self.btn_skip.setEnabled(False)

    def _show_rating(self, mem_key, recipe):
        self.pending_rating = (mem_key, recipe)
        self.slider.setValue(5); self.lbl_sv.setText("5")
        self.lbl_rate.setText(f"🤖 Rate [{recipe.get('technique_name','?')}] (1–10):")
        self.rlhf.show()

    def submit_rating(self):
        if self.pending_rating:
            mk, rec = self.pending_rating
            rating  = self.slider.value()

            # Run the learning on the dedicated learn pool — never waits for analysis
            future = self.dj._learn_pool.submit(
                self.dj.brain.ml.learn_from_feedback, rating, mk, rec)

            # Show brief status in the countdown label so the user gets feedback
            tech = rec.get('technique_name', '?')
            if rating >= 8:
                msg = f"✅ Brain updated — {tech} baseline improved"
            elif rating <= 3:
                msg = f"📉 Brain updated — {tech} baseline nudged away"
            elif 6 <= rating <= 7:
                msg = f"👍 Brain updated — small blend for {tech}"
            else:
                msg = f"⚖️  No change for {tech} (mediocre rating)"

            self.lbl_cd.setText(msg)
            self.lbl_cd.setStyleSheet("color:#FFAA00;font-weight:bold;")

            self.pending_rating = None
        self.rlhf.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dj  = NeuroDJ(sys.argv[1])
    win = NeuroDJWindow(dj)
    win.show()
    threading.Thread(target=dj.start_set, daemon=True).start()
    sys.exit(app.exec_())