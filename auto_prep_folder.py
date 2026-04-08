"""
auto_prep_folder.py  —  Neuro-DJ Deep Structure Analyser v6.2 (Stem Edition)
═══════════════════════════════════════════════════════════════
What this produces and why each field matters to the DJ engine:

  bpm, key, energy, genre      → static compatibility scoring
  first_beat_time              → phrase quantisation anchor
  piano_entries                → Amapiano structural anchors
  stems                        → vocal/log-drum maps for EQ events
  harmonic_stem                → Path to the drumless vocal stem for ACAPELLA_MASHUP

  structure_map:
    A timestamped list of labelled sections (intro/build/drop/
    breakdown/outro) with texture_density and energy_level
    for each. The DJ engine uses this to:
      a) Pick the mix-out point in the right section of A
      b) Pick the mix-in point at the right section of B
      c) Choose the technique based on WHAT both tracks are
         doing at the transition moment, not just global energy

  texture_profile:
    A coarse time-series of texture density (0-1) sampled every
    ~4 beats. Read at mix_trigger to understand if A is sparse
    (good for mixing) or dense (risky).

  energy_trajectory:
    "rising", "peak", "falling", "floor" — direction of energy
    at the outro, telling the engine whether A is fading
    naturally or abruptly dying.

  mix_character:
    "clean_intro" / "drum_heavy" / "vocal_heavy" / "melodic" /
    "sparse" — character of the first 32 beats of the track.

FIXES (v6.2 Stem Edition):
  - Integrated extract_harmonic_stem before garbage collecting the 44.1kHz 
    array to save an offline HPSS vocal stem for the Acapella Mashup engine.
"""

import librosa
import numpy as np
import json
import soundfile as sf
import os
import sys
import glob
import warnings
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# STEM EXTRACTION (V6.2 Addition)
# ─────────────────────────────────────────────────────────────────────────────

def extract_harmonic_stem(y_mono, sr, wav_path, margin=3.0):
    """
    Separates melodic/vocal content from drums using HPSS math.
    Saves a file named [original]_harmonic.wav
    Runs on the 44.1kHz array before garbage collection.
    """
    harmonic_path = wav_path.replace('.wav', '_harmonic.wav')
    
    if os.path.exists(harmonic_path):
        return os.path.basename(harmonic_path)
    
    try:
        print("      🎤 Extracting studio-quality harmonic stem (takes a moment)...")
        D = librosa.stft(y_mono)
        H, P = librosa.decompose.hpss(D, margin=margin)
        y_harmonic = librosa.istft(H)
        
        # Normalize
        peak = np.max(np.abs(y_harmonic))
        if peak > 0.01:
            y_harmonic = y_harmonic * (0.90 / peak)
        else:
            return None
        
        sf.write(harmonic_path, y_harmonic.astype(np.float32), sr)
        return os.path.basename(harmonic_path)
    except Exception as e:
        print(f"      ⚠️ Harmonic stem extraction failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# GENRE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_genre(y_mono, sr, bpm):
    if not (96.0 <= bpm <= 128.0):
        return 'open'
    stft      = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sub_ratio = np.mean(stft[freqs < 100]) / (np.mean(stft) + 1e-9)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
    return 'amapiano' if sub_ratio > 0.12 and centroid < 3200.0 else 'open'


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL SECTION MAP
# ─────────────────────────────────────────────────────────────────────────────

SECTION_LABELS = ['intro', 'build', 'drop', 'breakdown', 'outro']


def build_structure_map(y_mono, sr, beat_times, bpm):
    """
    Segments the track into labelled sections.
    Runs at whatever SR is passed in — designed for 11 025 Hz input.
    Peak RAM at 11 025 Hz: ~80 MB for a 5-minute track vs ~320 MB at 44 100 Hz.
    """
    try:
        import gc
        dur = float(librosa.get_duration(y=y_mono, sr=sr))
        hop = max(128, sr // 48)          # ~23 ms frames at 11 025 Hz

        rms      = librosa.feature.rms(y=y_mono, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr, hop_length=hop)[0]

        harm, perc = librosa.effects.hpss(y_mono)
        chroma   = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=hop)
        flux     = np.sum(np.diff(chroma, axis=1) ** 2, axis=0)
        flux     = np.concatenate([[0], flux])
        perc_rms = librosa.feature.rms(y=perc, hop_length=hop)[0]
        del harm, perc, chroma
        gc.collect()

        min_len  = min(len(rms), len(centroid), len(flux), len(perc_rms))
        rms      = rms[:min_len]
        centroid = centroid[:min_len]
        flux     = flux[:min_len]
        perc_rms = perc_rms[:min_len]

        frame_times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop)

        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-9)

        rms_n  = norm(rms)
        cent_n = norm(centroid)
        flux_n = norm(flux)
        perc_n = norm(perc_rms)

        density = rms_n * 0.35 + cent_n * 0.25 + perc_n * 0.25 + flux_n * 0.15

        smooth  = lambda x, w=21: np.convolve(x, np.ones(w) / w, mode='same')
        novelty = (np.abs(np.diff(smooth(rms_n),  prepend=rms_n[0]))  * 0.40 +
                   np.abs(np.diff(smooth(cent_n), prepend=cent_n[0])) * 0.30 +
                   np.abs(np.diff(smooth(flux_n), prepend=flux_n[0])) * 0.30)
        novelty = smooth(novelty, w=31)

        spb               = 60.0 / bpm
        min_section_beats = 16
        min_dist_frames   = int((min_section_beats * spb * sr) / hop)

        boundary_frames, _ = find_peaks(
            novelty,
            height=np.percentile(novelty, 70),
            distance=max(1, min_dist_frames),
        )
        boundary_times = ([0.0]
                          + [float(frame_times[f]) for f in boundary_frames]
                          + [dur])

        def snap_to_beat(t):
            if len(beat_times) == 0:
                return t
            idx = np.argmin(np.abs(beat_times - t))
            return float(beat_times[idx])

        boundary_times = sorted(set(snap_to_beat(t) for t in boundary_times))

        def segment_features(start, end):
            mask = (frame_times >= start) & (frame_times < end)
            if not np.any(mask):
                return 0.5, 0.5, 0.5, 0.5
            return (float(np.mean(rms_n[mask])), float(np.mean(cent_n[mask])),
                    float(np.mean(perc_n[mask])), float(np.mean(density[mask])))

        all_rms  = [segment_features(boundary_times[i], boundary_times[i + 1])[0]
                    for i in range(len(boundary_times) - 1)]
        rms_max  = max(all_rms) + 1e-9

        sections = []
        for i in range(len(boundary_times) - 1):
            start = boundary_times[i]
            end   = boundary_times[i + 1]
            if end - start < 4.0:
                continue

            r, c, p, d = segment_features(start, end)
            pos   = start / dur
            r_rel = r / rms_max

            if   pos < 0.12 and r_rel < 0.65:   label = 'intro'
            elif pos > 0.82 and r_rel < 0.65:   label = 'outro'
            elif r_rel > 0.80 and p > 0.60:     label = 'drop'
            elif r_rel < 0.55 and c < 0.55:     label = 'breakdown'
            elif 0.55 < r_rel <= 0.80:           label = 'build'
            elif pos < 0.25:                     label = 'intro'
            elif pos > 0.75:                     label = 'outro'
            else:                                label = 'drop' if r_rel > 0.70 else 'build'

            sections.append({
                'label':           label,
                'start':           round(start, 2),
                'end':             round(end, 2),
                'texture_density': round(float(d), 3),
                'energy_level':    round(float(r_rel), 3),
            })
        return sections

    except Exception as e:
        print(f"   ⚠️  Structure map failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# TEXTURE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def build_texture_profile(y_mono, sr, bpm, sample_every_beats=4):
    """
    Returns [{time, density}] sampled every `sample_every_beats` beats.
    Lightweight — uses only RMS + centroid, no HPSS.
    """
    try:
        hop  = max(128, sr // 48)
        spb  = 60.0 / bpm
        step = sample_every_beats * spb
        dur  = float(librosa.get_duration(y=y_mono, sr=sr))

        rms  = librosa.feature.rms(y=y_mono, hop_length=hop)[0]
        cent = librosa.feature.spectral_centroid(y=y_mono, sr=sr, hop_length=hop)[0]
        n    = min(len(rms), len(cent))
        rms  = rms[:n]
        cent = cent[:n]
        times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)

        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-9)

        density = norm(rms) * 0.6 + norm(cent) * 0.4

        profile = []
        t = 0.0
        while t < dur:
            mask = (times >= t) & (times < t + step)
            if np.any(mask):
                profile.append({'time':    round(t, 2),
                                'density': round(float(np.mean(density[mask])), 3)})
            t += step
        return profile
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def detect_energy_trajectory(y_mono, sr):
    """
    Describes the direction of energy in the last 60 seconds of the track.
    'rising'  → energy still growing at outro
    'peak'    → at or near maximum energy
    'falling' → steadily declining (natural outro — best to mix)
    'floor'   → already very quiet
    """
    try:
        dur    = float(librosa.get_duration(y=y_mono, sr=sr))
        window = min(60.0, dur * 0.25)
        y_end  = y_mono[int((dur - window)     * sr):]
        y_mid  = y_mono[int((dur - window * 2) * sr): int((dur - window) * sr)]

        rms_end = float(np.sqrt(np.mean(y_end ** 2)))
        rms_mid = float(np.sqrt(np.mean(y_mid ** 2)))
        rms_max = float(np.sqrt(np.mean(y_mono ** 2))) * 2.0 + 1e-9

        ratio_end = rms_end / rms_max
        delta     = rms_end - rms_mid

        if ratio_end < 0.20:             return 'floor'
        if delta > rms_max * 0.05:       return 'rising'
        if ratio_end > 0.60:             return 'peak'
        return 'falling'
    except Exception:
        return 'falling'


# ─────────────────────────────────────────────────────────────────────────────
# MIX CHARACTER
# ─────────────────────────────────────────────────────────────────────────────

def detect_mix_character(y_mono, sr):
    """
    Describes the opening character of a track using the first 32 seconds.
    Works at any SR — designed for 11 025 Hz input.
    """
    try:
        window = min(32.0, float(librosa.get_duration(y=y_mono, sr=sr)) * 0.15)
        y_open = y_mono[:int(window * sr)]
        if len(y_open) < sr:
            return 'clean_intro'

        rms_total = float(np.sqrt(np.mean(y_open ** 2)))
        if rms_total < 0.015:
            return 'clean_intro'

        stft  = np.abs(librosa.stft(y_open, n_fft=512, hop_length=128))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)

        vmask = (freqs >= 300) & (freqs <= 3400)
        bmask = freqs < 200
        total_e  = float(np.mean(stft)) + 1e-9
        vocal_e  = float(np.mean(stft[vmask])) if vmask.any() else 0.0
        bass_e   = float(np.mean(stft[bmask]))  if bmask.any() else 0.0

        vocal_ratio = vocal_e / total_e
        bass_ratio  = bass_e  / total_e

        if   vocal_ratio > 0.45:  return 'vocal_heavy'
        elif bass_ratio  > 0.18:  return 'drum_heavy'
        elif vocal_ratio > 0.30:  return 'melodic'
        elif rms_total   < 0.04:  return 'sparse'
        return 'clean_intro'
    except Exception:
        return 'clean_intro'


# ─────────────────────────────────────────────────────────────────────────────
# PIANO ENTRY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_piano_entries(y_mono, sr):
    try:
        harmonic, _ = librosa.effects.hpss(y_mono)
        chroma      = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
        flux        = np.sum(np.diff(chroma, axis=1) ** 2, axis=0)
        flux_s      = np.convolve(flux, np.ones(20) / 20, mode='same')
        peaks, _    = find_peaks(flux_s,
                                 height=np.percentile(flux_s, 78),
                                 distance=int(sr / 512 * 8))
        times = librosa.frames_to_time(peaks, sr=sr)
        return [float(t) for t in times if t > 10.0]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE OUTRO SCAN
# ─────────────────────────────────────────────────────────────────────────────

def find_outro_beat(beat_times, rms, sr, total_duration):
    """
    Scans from the last N beats backwards to find a quiet 32-beat window
    suitable for mixing out.
    """
    if len(beat_times) == 0:
        return 0, total_duration * 0.75

    max_rms = np.max(rms) + 1e-9
    for window_beats in (96, 128, 160):
        start_idx = max(0, len(beat_times) - window_beats)
        for i in range(start_idx, len(beat_times) - 32, 32):
            s = librosa.time_to_frames(beat_times[i],      sr=sr)
            e = librosa.time_to_frames(beat_times[i + 32], sr=sr)
            e = max(s + 1, min(e, len(rms) - 1))   # ensure non-empty slice
            phrase_rms = np.mean(rms[s:e])
            if phrase_rms < max_rms * 0.40:
                return i, float(beat_times[i])

    fb = max(0, len(beat_times) - 65)
    return fb, float(beat_times[fb])


# ─────────────────────────────────────────────────────────────────────────────
# STEM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_stems(y_mono, sr):
    """
    Detects vocal regions and log-drum hit times.
    """
    try:
        target_sr = 22_050
        if sr > target_sr:
            y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
            sr     = target_sr
        # If sr <= target_sr, use as-is (11 025 Hz is fine)

        harmonic, percussive = librosa.effects.hpss(y_mono, margin=3.0)
        S_harm = np.abs(librosa.stft(harmonic, n_fft=1024, hop_length=256))
        freqs  = librosa.fft_frequencies(sr=sr, n_fft=1024)

        vmask        = (freqs >= 300) & (freqs <= 3400)
        vocal_energy = np.mean(S_harm[vmask, :], axis=0) if vmask.any() else np.zeros(S_harm.shape[1])
        smf          = max(1, int(sr / 256 * 0.5))
        vocal_smooth = np.convolve(vocal_energy, np.ones(smf) / smf, mode='same')
        vocal_active = vocal_smooth > np.percentile(vocal_smooth, 55)

        frame_times  = librosa.frames_to_time(np.arange(len(vocal_active)),
                                              sr=sr, hop_length=256)
        vocal_regions, in_r, r_start = [], False, 0.0
        for i, a in enumerate(vocal_active):
            if a and not in_r:
                r_start = float(frame_times[i])
                in_r    = True
            elif not a and in_r:
                if float(frame_times[i]) - r_start >= 1.5:
                    vocal_regions.append([round(r_start, 2),
                                          round(float(frame_times[i]), 2)])
                in_r = False
        if in_r:
            vocal_regions.append([round(r_start, 2),
                                   round(float(frame_times[-1]), 2)])

        S_perc = np.abs(librosa.stft(percussive, n_fft=1024, hop_length=256))
        bmask  = freqs < 200
        bass_e = np.mean(S_perc[bmask, :], axis=0) if bmask.any() else np.zeros(S_perc.shape[1])
        bass_t = librosa.frames_to_time(np.arange(len(bass_e)), sr=sr, hop_length=256)
        mdist  = max(1, int(sr / 256 * 0.4))
        pk, _  = find_peaks(bass_e,
                             height=np.percentile(bass_e, 78),
                             distance=mdist)
        log_drum_hits = [round(float(bass_t[p]), 3)
                         for p in pk if bass_t[p] > 5.0]

        return {
            'has_vocals':     len(vocal_regions) > 0,
            'vocal_regions':  vocal_regions,
            'log_drum_hits':  log_drum_hits[:50],
        }
    except Exception as e:
        print(f"   ⚠️  Stem analysis failed: {e}")
        return {'has_vocals': False, 'vocal_regions': [], 'log_drum_hits': []}


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def prep_song_logic(input_path):
    """
    Memory-efficient analysis pipeline.
    Strategy: load at 44 100 Hz ONCE for BPM/key/RMS (accuracy-critical),
    write the WAV, extract the harmonic stem, then IMMEDIATELY free the 
    full-res array.
    All structural analysis (HPSS, structure map, texture) runs at 11 025 Hz
    which uses 4× less RAM.
    """
    import gc
    print(f"🧬 Analyzing: {os.path.basename(input_path)}")

    # ── Full-resolution pass (44 100 Hz) ─────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(input_path, sr=44100, mono=False)
    y_mono = librosa.to_mono(y)
    total_duration = float(librosa.get_duration(y=y_mono, sr=sr))

    # BPM & beats
    onset_env    = librosa.onset.onset_strength(y=y_mono, sr=sr, aggregate=np.median)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env,
                                                 sr=sr, start_bpm=112.0)
    bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

    # Amapiano tempo correction
    if   70.0 <= bpm <= 85.0:  bpm = bpm * 1.5
    elif bpm > 130.0:          bpm = bpm / 2.0
    elif bpm < 70.0:           bpm = bpm * 2.0
    bpm = round(bpm, 1)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    del onset_env

    # Key
    chroma       = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    key_names    = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    detected_key = key_names[int(np.argmax(np.mean(chroma, axis=1)))]
    del chroma

    # Energy label
    cent         = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
    avg_centroid = float(np.mean(cent))
    energy_score = "High" if avg_centroid > 2500 else "Low/Chill"
    del cent

    genre = detect_genre(y_mono, sr, bpm)

    # RMS outro scan at full rate
    rms = librosa.feature.rms(y=y_mono)[0]
    _, optimal_mix_out_time = find_outro_beat(beat_times, rms, sr, total_duration)
    outro_start_time = (float(beat_times[-33])
                        if len(beat_times) > 33
                        else total_duration * 0.75)
    del rms

    # Write output WAV before freeing the full-res array
    base, _ = os.path.splitext(input_path)
    output_wav = base + "_ready.wav"
    audio_out  = y.T if y.ndim > 1 else y
    sf.write(output_wav, audio_out.astype(np.float32), sr)

    # ── V6.2: EXTRACT HARMONIC ACAPELLA STEM ──
    h_stem_file = extract_harmonic_stem(y_mono, sr, output_wav)
    if h_stem_file is None:
        print(f"      ⚠️ No harmonic stem saved. Engine will use real-time M/S fallback.")

    # Free massive 44.1kHz arrays
    del y, y_mono
    gc.collect()

    # ── Low-resolution pass (11 025 Hz) ──────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_low, sr_low = librosa.load(input_path, sr=11025, mono=True)

    piano_entries     = find_piano_entries(y_low, sr_low)
    energy_trajectory = detect_energy_trajectory(y_low, sr_low)
    mix_character     = detect_mix_character(y_low, sr_low)
    print(f"      Energy trajectory: {energy_trajectory}  |  Mix character: {mix_character}")

    print(f"   📈 Building texture profile...")
    texture_profile = build_texture_profile(y_low, sr_low, bpm)
    gc.collect()

    print(f"   🗺️  Building structural section map...")
    structure_map  = build_structure_map(y_low, sr_low, beat_times, bpm)
    section_labels = [s['label'] for s in structure_map]
    print(f"      Sections: {' → '.join(section_labels)}")
    gc.collect()

    print(f"   🔬 Running stem analysis...")
    stems = analyze_stems(y_low, sr_low)
    print(f"      {'🎤 vocals' if stems['has_vocals'] else '🔇 no vocals'}  |  "
          f"{len(stems['log_drum_hits'])} log drum hits")

    print(f"   🎼 Building phrase-level intelligence map...")
    try:
        from phrase_analyzer import analyse_phrases, get_best_exit_phrases, get_best_entry_phrases
        phrases       = analyse_phrases(y_low, sr_low, beat_times, bpm, structure_map)
        # FIX: pass total_duration so the positional gate in v2 fires correctly
        exit_phrases  = get_best_exit_phrases(phrases, track_dur=total_duration)
        entry_phrases = get_best_entry_phrases(phrases)
        func_counts   = {}
        for p in phrases:
            func_counts[p['function']] = func_counts.get(p['function'], 0) + 1
        print(f"      {len(phrases)} phrases: " +
              "  ".join(f"{k}×{v}"
                        for k, v in sorted(func_counts.items(),
                                           key=lambda x: -x[1])[:4]))
        print(f"      Best exits:   {[round(p['start'], 1) for p in exit_phrases[:3]]}")
        print(f"      Best entries: {[round(p['start'], 1) for p in entry_phrases[:3]]}")
    except Exception as e:
        print(f"      ⚠️  Phrase analysis skipped: {e}")
        phrases       = []
        exit_phrases  = []
        entry_phrases = []

    del y_low
    gc.collect()

    result = {
        "filename":           os.path.abspath(output_wav),
        "harmonic_stem":      h_stem_file,             # V6.2 Addition
        "bpm":                round(bpm, 2),
        "key":                detected_key,
        "energy":             energy_score,
        "genre":              genre,
        "first_beat_time":    round(float(beat_times[0]), 3) if len(beat_times) else 0.0,
        "piano_entries":      [round(t, 3) for t in piano_entries[:10]],
        "stems":              stems,
        "structure_map":      structure_map,
        "texture_profile":    texture_profile,
        "energy_trajectory":  energy_trajectory,
        "mix_character":      mix_character,
        "phrases":            phrases,
        "best_exit_phrases":  [p['start'] for p in exit_phrases[:6]],
        "best_entry_phrases": [p['start'] for p in entry_phrases[:5]],
        "zones": {
            "optimal_mix_out": round(optimal_mix_out_time, 3),
            "outro_start":     round(outro_start_time, 3),
        },
    }

    tag = "🎹 Amapiano" if genre == 'amapiano' else f"🎵 {energy_score}"
    print(f"   ✅ {tag} | {bpm:.1f} BPM | {detected_key} | "
          f"Mix-out@{optimal_mix_out_time:.1f}s | "
          f"{len(structure_map)} sections | {len(phrases)} phrases")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def _load_existing_library():
    if not os.path.exists("master_library.json"):
        return {}
    try:
        with open("master_library.json") as f:
            records = json.load(f)
        result = {}
        for rec in records:
            ready = rec.get('filename', '')
            if ready.endswith('_ready.wav'):
                base = ready[:-len('_ready.wav')]
                for orig_ext in ('.mp3', '.wav', '.flac', '.aiff'):
                    candidate = base + orig_ext
                    if os.path.exists(candidate):
                        result[os.path.abspath(candidate)] = rec
                        break
        return result
    except Exception:
        return {}


def _save_library(master_library):
    tmp = "master_library.json.tmp"
    try:
        sorted_lib = sorted(master_library, key=lambda x: x['bpm'])
        with open(tmp, 'w') as f:
            json.dump(sorted_lib, f, indent=4)
        os.replace(tmp, "master_library.json")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        try:
            os.remove(tmp)
        except Exception:
            pass


def _worker_entry(input_path, result_queue):
    try:
        result = prep_song_logic(input_path)
        result_queue.put(('ok', result))
    except Exception as e:
        import traceback
        result_queue.put(('error', str(e) + '\n' + traceback.format_exc()))


def process_folder(folder_path):
    import multiprocessing

    extensions = ('*.mp3', '*.wav', '*.flac', '*.aiff')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files = [f for f in files if "_ready.wav" not in f and "_harmonic.wav" not in f]

    if not files:
        print("⚠️  No audio files found.")
        return

    existing     = _load_existing_library()
    todo         = [f for f in files if os.path.abspath(f) not in existing]
    already_done = len(files) - len(todo)

    print(f"📂 {len(files)} file(s) found in folder.")
    if already_done:
        print(f"   ✅ {already_done} already in library — skipping.")
    if not todo:
        print("   Nothing to do. All tracks are up to date.")
        return
    print(f"   🔄 {len(todo)} track(s) to process. "
          f"Estimated time: ~{len(todo) * 30}–{len(todo) * 45}s\n")

    master_library = list(existing.values())
    failed = []

    for i, filepath in enumerate(todo):
        print(f"\n[{i + 1}/{len(todo)}] {'─' * 50}")
        print(f"Processing: {os.path.basename(filepath)}")

        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_worker_entry,
            args=(filepath, result_queue),
            daemon=True,
        )

        try:
            proc.start()
            proc.join(timeout=300)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                print(f"   ❌ TIMEOUT (>300s) — skipping {os.path.basename(filepath)}")
                failed.append(filepath)
                continue

            if proc.exitcode != 0:
                code_str = str(proc.exitcode)
                if proc.exitcode == -11:
                    code_str = "-11 (SIGSEGV — likely corrupted audio)"
                print(f"   ❌ Subprocess exited {code_str}")
                failed.append(filepath)
                continue

            try:
                status, payload = result_queue.get(timeout=5)
            except Exception:
                print(f"   ❌ Subprocess produced no result.")
                failed.append(filepath)
                continue

            if status == 'error':
                print(f"   ❌ Analysis error:\n{payload}")
                failed.append(filepath)
                continue

            master_library.append(payload)
            _save_library(master_library)
            print(f"   ✅ Saved. Library now has {len(master_library)} tracks.")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted. Saving progress...")
            proc.terminate()
            _save_library(master_library)
            print(f"   Library saved: {len(master_library)} track(s) complete.")
            return

        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            failed.append(filepath)
            try:
                proc.terminate()
            except Exception:
                pass

    amp = sum(1 for t in master_library if t.get('genre') == 'amapiano')
    print(f"\n{'─' * 60}")
    print(f"✅ Done: {len(master_library)} total tracks in library")
    print(f"   🎹 Amapiano: {amp}")
    print(f"   ✅ Processed this run: {len(todo) - len(failed)}")
    if failed:
        print(f"   ❌ Failed ({len(failed)}):")
        for f in failed:
            print(f"      • {os.path.basename(f)}")
        print(f"   Try re-encoding with:")
        print(f"   ffmpeg -i bad_file.mp3 -codec:a libmp3lame fixed.mp3")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_prep_folder.py ./my_music_folder")
    else:
        process_folder(sys.argv[1])