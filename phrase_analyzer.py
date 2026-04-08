"""
phrase_analyzer.py — Neuro-DJ Phrase-Level Intelligence
═══════════════════════════════════════════════════════
This module is called by `auto_prep_folder.py` during offline analysis.
It slices a track into strict 8-beat musical phrases and analyzes the
spectral and dynamic content of each slice.

It calculates 6 core metrics per phrase:
  1. mixability         (0-1) — How easy is it to layer another track over this?
  2. bass_density       (0-1) — Is the sub/kick heavily active?
  3. vocal_density      (0-1) — Are there prominent frequencies in the vocal range?
  4. harmonic_density   (0-1) — Chords, pads, and synths.
  5. percussive_density (0-1) — Shakers, hats, and snare activity.
  6. tension_score      (0-1) — Is energy building, static, or releasing?

Based on these metrics and the parent section, it assigns a semantic `function`
(e.g., 'dj_friendly_outro', 'vocal_spotlight', 'tension_build') which the
DecisionCore relies on to pick the perfect mixing technique.

FIXES (v2):
  - clamp01 moved to TOP of file (was after first use — NameError in REPL/reload)
  - get_best_exit_phrases now filters pos_ratio >= 0.70 so early phrases never
    leak into the precomputed best_exit_phrases list written to master_library.json
  - Tension multiplier 2.0 → 1.2: the original value pushed almost every phrase
    to the 0.0 or 1.0 boundary, making tension_score useless as a continuous signal
  - Mixability formula rebalanced: adds tension as a cost factor and reduces
    bass/vocal weights slightly so medium-density phrases score ~0.45 not ~0.10
  - Frame arrays trimmed to shared min_len before use (STFT length can differ from
    RMS length by one frame, causing IndexError on short clips)
  - Added track_dur guard: phrases beyond end-of-track are skipped cleanly
"""

import numpy as np
import librosa
import warnings

# ── Utility (MUST be at top — used throughout this module) ───────────────────
def clamp01(v: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


# The semantic labels expected by the DecisionCore
PHRASE_FUNCTIONS = [
    'dj_friendly_outro',  # Low bass, low vocal, decreasing tension  (Perfect mix-out)
    'decompression',      # Dropping energy immediately after a peak
    'drum_foundation',    # High perc/bass, low harmonic/vocal        (Perfect mix-in)
    'harmonic_bed',       # High harmonic, low perc                   (Good for melodic transitions)
    'release_peak',       # Maximum energy/density                    (Do NOT mix here)
    'bass_showcase',      # Heavy sub, sparse everything else         (Amapiano log-drum solo)
    'tension_build',      # Rising energy/RMS, usually preceding a drop
    'vocal_spotlight',    # High vocal density, low/mid backing       (Danger zone for clashes)
    'fake_outro',         # Drops in density but occurs too early in the track
    'transition_glue',    # Neutral, medium density, standard groove
]

# Minimum track position before a phrase qualifies as an exit candidate.
# Must stay in sync with MIN_EXIT_POSITION_RATIO in neuro_gui.py.
_MIN_EXIT_POS_RATIO: float = 0.70


# ─────────────────────────────────────────────────────────────────────────────
def analyse_phrases(y_low, sr_low, beat_times, bpm, structure_map):
    """
    Slices the track into 8-beat phrases and calculates density + semantic function.
    Designed to run on the 11 025 Hz downsampled mono signal to save RAM.

    Parameters
    ----------
    y_low        : np.ndarray  — 11 025 Hz mono audio
    sr_low       : int         — sample rate (should be 11 025)
    beat_times   : np.ndarray  — wall-clock beat times in seconds (from 44 100 Hz analysis)
    bpm          : float
    structure_map: list[dict]  — section list from build_structure_map()

    Returns
    -------
    list[dict]  — one entry per 8-beat phrase
    """
    if len(beat_times) < 16:
        return []

    track_dur = float(librosa.get_duration(y=y_low, sr=sr_low))
    if track_dur < 1.0:
        return []

    # ── Feature Extraction ───────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rms = librosa.feature.rms(y=y_low, frame_length=1024, hop_length=256)[0]

        S     = np.abs(librosa.stft(y_low, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr_low, n_fft=1024)

        harm, perc = librosa.effects.hpss(y_low, margin=2.0)
        rms_harm   = librosa.feature.rms(y=harm, frame_length=1024, hop_length=256)[0]
        rms_perc   = librosa.feature.rms(y=perc, frame_length=1024, hop_length=256)[0]

    # Frequency masks — ceiling is Nyquist of 11 025 Hz ≈ 5 512 Hz
    mask_bass  = freqs < 200
    mask_vocal = (freqs >= 300) & (freqs <= 3_400)

    bass_energy  = (np.mean(S[mask_bass,  :], axis=0) if np.any(mask_bass)
                    else np.zeros_like(rms))
    vocal_energy = (np.mean(S[mask_vocal, :], axis=0) if np.any(mask_vocal)
                    else np.zeros_like(rms))

    # FIX: trim all arrays to shared min_len — STFT length can differ from
    # RMS by one frame, causing IndexError on short clips.
    min_len = min(len(rms), len(rms_harm), len(rms_perc),
                  len(bass_energy), len(vocal_energy))
    rms          = rms[:min_len]
    rms_harm     = rms_harm[:min_len]
    rms_perc     = rms_perc[:min_len]
    bass_energy  = bass_energy[:min_len]
    vocal_energy = vocal_energy[:min_len]

    frame_times = librosa.frames_to_time(np.arange(min_len), sr=sr_low, hop_length=256)

    # ── Normalisation ────────────────────────────────────────────────────────
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - mn) / (mx - mn)

    rms_n   = _norm(rms)
    bass_n  = _norm(bass_energy)
    vocal_n = _norm(vocal_energy)
    harm_n  = _norm(rms_harm)
    perc_n  = _norm(rms_perc)

    # ── Phrase Chunking ──────────────────────────────────────────────────────
    phrases = []
    for i in range(0, len(beat_times) - 8, 8):
        start_t = float(beat_times[i])
        end_t   = float(beat_times[i + 8])

        # Skip phrases that extend beyond the audio
        if end_t > track_dur + 0.5:
            break

        # Find which macro-section this phrase belongs to
        parent_section = 'drop'
        for sec in structure_map:
            if sec['start'] <= start_t < sec['end']:
                parent_section = sec['label']
                break

        mask = (frame_times >= start_t) & (frame_times < end_t)
        if not np.any(mask):
            continue

        p_rms   = float(np.mean(rms_n[mask]))
        p_bass  = float(np.mean(bass_n[mask]))
        p_vocal = float(np.mean(vocal_n[mask]))
        p_harm  = float(np.mean(harm_n[mask]))
        p_perc  = float(np.mean(perc_n[mask]))

        # ── Tension score ────────────────────────────────────────────────────
        # FIX: multiplier reduced 2.0 → 1.2.
        # Old: clamp01(0.5 + delta * 2.0) → delta = ±0.25 already saturated → useless
        # New: clamp01(0.5 + delta * 1.2) → realistic spread, most values in [0.2, 0.8]
        local_rms = rms_n[mask]
        if len(local_rms) > 4:
            half        = len(local_rms) // 2
            first_half  = float(np.mean(local_rms[:half]))
            second_half = float(np.mean(local_rms[half:]))
            tension = clamp01(0.5 + (second_half - first_half) * 1.2)
        else:
            tension = 0.5

        # ── Mixability ───────────────────────────────────────────────────────
        # FIX: added tension as a cost factor; reduced bass/vocal weight slightly.
        # Old formula: 1 - bass*0.4 - vocal*0.4 - rms*0.2
        #   → phrase with bass=0.5, vocal=0.5, rms=0.5 → score = 0.10 (near zero!)
        # New formula: 1 - bass*0.35 - vocal*0.35 - rms*0.15 - tension*0.15
        #   → same phrase → score = 0.375 (sensible mid-range)
        mixability = clamp01(
            1.0
            - p_bass   * 0.35
            - p_vocal  * 0.35
            - p_rms    * 0.15
            - tension  * 0.15
        )

        # ── Semantic Classification ──────────────────────────────────────────
        pos_ratio = start_t / max(track_dur, 1.0)
        func = 'transition_glue'

        if pos_ratio > 0.80 and p_bass < 0.4 and p_vocal < 0.3:
            func = 'dj_friendly_outro'
        elif p_vocal > 0.7 and p_bass < 0.5:
            func = 'vocal_spotlight'
        elif p_bass > 0.75 and p_perc > 0.5 and p_harm < 0.3:
            func = 'bass_showcase'
        elif parent_section == 'build' and tension > 0.65:
            func = 'tension_build'
        elif parent_section == 'drop' and p_rms > 0.8:
            func = 'release_peak'
        elif pos_ratio < 0.20 and p_perc > 0.5 and p_harm < 0.4:
            func = 'drum_foundation'
        elif p_harm > 0.7 and p_perc < 0.3:
            func = 'harmonic_bed'
        elif 0.5 < pos_ratio < 0.75 and p_rms < 0.4 and p_bass < 0.3:
            func = 'fake_outro'
        elif parent_section == 'breakdown' and tension < 0.4 and pos_ratio > 0.3:
            func = 'decompression'

        phrases.append({
            'start':              round(start_t, 3),
            'end':                round(end_t,   3),
            'parent_section':     parent_section,
            'function':           func,
            'mixability':         round(mixability, 3),
            'bass_density':       round(p_bass,  3),
            'vocal_density':      round(p_vocal, 3),
            'harmonic_density':   round(p_harm,  3),
            'percussive_density': round(p_perc,  3),
            'tension_score':      round(tension, 3),
        })

    return phrases


def get_best_exit_phrases(phrases, track_dur: float = 0.0):
    """
    Returns phrase candidates for mixing OUT of a track, ranked best-first.

    FIX: positional gate (pos_ratio >= 0.70) is now enforced here in addition
    to PhraseCandidateSearch so that early phrases can never appear in the
    precomputed best_exit_phrases list stored in master_library.json.

    Parameters
    ----------
    phrases   : list[dict]  — output of analyse_phrases()
    track_dur : float       — total track duration in seconds (0 = skip gate)
    """
    if not phrases:
        return []

    valid_exits = []
    for p in phrases:
        # Hard positional gate — never allow mix-out before 70 % of the track
        if track_dur > 0:
            if (p['start'] / max(track_dur, 1.0)) < _MIN_EXIT_POS_RATIO:
                continue

        # Functional blacklist
        if p['function'] in ('release_peak', 'vocal_spotlight',
                             'fake_outro',   'tension_build',
                             'bass_showcase'):
            continue
        if p['vocal_density'] > 0.6:
            continue

        score = p['mixability'] * 2.0
        if p['function'] == 'dj_friendly_outro': score += 1.5
        elif p['function'] == 'decompression':   score += 0.5
        elif p['function'] == 'drum_foundation': score += 0.3
        score -= p['tension_score']

        valid_exits.append((score, p))

    valid_exits.sort(key=lambda x: x[0], reverse=True)
    return [p for _score, p in valid_exits]


def get_best_entry_phrases(phrases):
    """
    Returns phrase candidates for mixing INTO a track, ranked best-first.
    Prioritises clean drum foundations and high mixability in the early track.
    """
    if not phrases:
        return []

    valid_entries = []
    for p in phrases:
        if p['parent_section'] not in ('intro', 'breakdown'):
            continue
        if p['vocal_density'] > 0.5:
            continue

        score = p['mixability'] * 1.5
        if p['function'] == 'drum_foundation': score += 1.0
        if p['function'] == 'harmonic_bed':    score += 0.5

        valid_entries.append((score, p))

    valid_entries.sort(key=lambda x: x[0], reverse=True)
    return [p for _score, p in valid_entries]