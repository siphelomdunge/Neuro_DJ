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
V27 DecisionCore relies on to pick the perfect mixing technique.
"""

import numpy as np
import librosa
import warnings

# The semantic labels expected by the V27 DecisionCore
PHRASE_FUNCTIONS = [
    'dj_friendly_outro',  # Low bass, low vocal, decreasing tension (Perfect mix-out)
    'decompression',      # Dropping energy immediately after a peak
    'drum_foundation',    # High perc/bass, low harmonic/vocal (Perfect mix-in)
    'harmonic_bed',       # High harmonic, low perc (Good for melodic transitions)
    'release_peak',       # Maximum energy/density (Do not mix here)
    'bass_showcase',      # Heavy sub, sparse everything else (Amapiano log drum solo)
    'tension_build',      # Rising energy/RMS, usually preceding a drop
    'vocal_spotlight',    # High vocal density, low/mid backing (Danger zone for clashes)
    'fake_outro',         # Drops in density but occurs too early in the track
    'transition_glue'     # Neutral, medium density, standard groove
]

def analyse_phrases(y_low, sr_low, beat_times, bpm, structure_map):
    """
    Slices the track into 8-beat phrases and calculates density and semantic function.
    Runs on the 11025Hz downsampled audio to save RAM.
    """
    if len(beat_times) < 16:
        return []

    track_dur = float(librosa.get_duration(y=y_low, sr=sr_low))
    
    # ── Feature Extraction ───────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # RMS Energy
        rms = librosa.feature.rms(y=y_low, frame_length=1024, hop_length=256)[0]
        
        # Spectrogram for frequency banding
        S = np.abs(librosa.stft(y_low, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr_low, n_fft=1024)
        
        # HPSS for harmonic vs percussive distinction
        harm, perc = librosa.effects.hpss(y_low, margin=2.0)
        rms_harm = librosa.feature.rms(y=harm, frame_length=1024, hop_length=256)[0]
        rms_perc = librosa.feature.rms(y=perc, frame_length=1024, hop_length=256)[0]

    # Frequency masks (adapted for 11025Hz Nyquist limit)
    mask_bass  = freqs < 200
    mask_vocal = (freqs >= 300) & (freqs <= 3400)
    
    bass_energy  = np.mean(S[mask_bass, :], axis=0) if np.any(mask_bass) else np.zeros_like(rms)
    vocal_energy = np.mean(S[mask_vocal, :], axis=0) if np.any(mask_vocal) else np.zeros_like(rms)
    
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr_low, hop_length=256)
    
    # Normalization helper
    def norm(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-6: return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    rms_n   = norm(rms)
    bass_n  = norm(bass_energy)
    vocal_n = norm(vocal_energy)
    harm_n  = norm(rms_harm)
    perc_n  = norm(rms_perc)

    # ── Phrase Chunking ──────────────────────────────────────────────────────
    # Group beats into 8-beat chunks
    phrases = []
    for i in range(0, len(beat_times) - 8, 8):
        start_t = float(beat_times[i])
        end_t   = float(beat_times[i+8])
        
        # Find which macro-section this phrase belongs to
        parent_section = 'drop'
        for sec in structure_map:
            if sec['start'] <= start_t < sec['end']:
                parent_section = sec['label']
                break
                
        # Isolate frames for this phrase
        mask = (frame_times >= start_t) & (frame_times < end_t)
        if not np.any(mask):
            continue
            
        # Calculate local densities
        p_rms   = float(np.mean(rms_n[mask]))
        p_bass  = float(np.mean(bass_n[mask]))
        p_vocal = float(np.mean(vocal_n[mask]))
        p_harm  = float(np.mean(harm_n[mask]))
        p_perc  = float(np.mean(perc_n[mask]))
        
        # Calculate tension (is energy rising across this specific 8-beat window?)
        local_rms = rms_n[mask]
        if len(local_rms) > 4:
            first_half = np.mean(local_rms[:len(local_rms)//2])
            second_half = np.mean(local_rms[len(local_rms)//2:])
            tension = clamp01(0.5 + (second_half - first_half) * 2.0)
        else:
            tension = 0.5
            
        # Mixability: High when bass is low, vocals are low, and energy isn't peaking
        mixability = clamp01(1.0 - (p_bass * 0.4 + p_vocal * 0.4 + p_rms * 0.2))
        
        # ── Semantic Classification ──
        pos_ratio = start_t / track_dur
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
        elif pos_ratio > 0.5 and pos_ratio < 0.75 and p_rms < 0.4 and p_bass < 0.3:
            func = 'fake_outro'
        elif parent_section == 'breakdown' and tension < 0.4 and pos_ratio > 0.3:
            func = 'decompression'

        phrases.append({
            'start': round(start_t, 3),
            'end': round(end_t, 3),
            'parent_section': parent_section,
            'function': func,
            'mixability': round(mixability, 3),
            'bass_density': round(p_bass, 3),
            'vocal_density': round(p_vocal, 3),
            'harmonic_density': round(p_harm, 3),
            'percussive_density': round(p_perc, 3),
            'tension_score': round(tension, 3)
        })

    return phrases

def get_best_exit_phrases(phrases):
    """
    Returns the top phrase candidates for mixing OUT of a track.
    Prioritizes 'dj_friendly_outro', high mixability, and low vocals.
    """
    if not phrases:
        return []
        
    valid_exits = []
    for p in phrases:
        # Avoid mixing out too early, during a peak, or over heavy vocals
        if p['function'] in ('release_peak', 'vocal_spotlight', 'fake_outro', 'tension_build'):
            continue
        if p['vocal_density'] > 0.6:
            continue
            
        score = p['mixability'] * 2.0
        if p['function'] == 'dj_friendly_outro': score += 1.5
        if p['function'] == 'decompression': score += 0.5
        
        # Penalize high tension
        score -= p['tension_score']
        
        valid_exits.append((score, p))
        
    valid_exits.sort(key=lambda x: x[0], reverse=True)
    return [p for score, p in valid_exits]

def get_best_entry_phrases(phrases):
    """
    Returns the top phrase candidates for mixing INTO a track.
    Prioritizes clean drum foundations and high mixability in the early track.
    """
    if not phrases:
        return []
        
    valid_entries = []
    for p in phrases:
        # We only want to mix into the beginning of a track
        if p['parent_section'] not in ('intro', 'breakdown'):
            continue
        if p['vocal_density'] > 0.5:
            continue
            
        score = p['mixability'] * 1.5
        if p['function'] == 'drum_foundation': score += 1.0
        if p['function'] == 'harmonic_bed': score += 0.5
        
        valid_entries.append((score, p))
        
    valid_entries.sort(key=lambda x: x[0], reverse=True)
    return [p for score, p in valid_entries]

def clamp01(v):
    return max(0.0, min(1.0, v))