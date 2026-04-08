"""
Track warping and timestamp rescaling.
"""
from __future__ import annotations
import os
import uuid
import warnings
import numpy as np
import librosa
import soundfile as sf
import copy

from core.constants import MIN_STRETCH_RATIO, MAX_STRETCH_RATIO


class RunwayExhausted(Exception):
    """Raised when there's insufficient time remaining for a transition."""
    pass


class TrackWarper:
    """Handles time-stretching and metadata rescaling."""
    
    def __init__(self):
        self._temp_files = []
    
    def warp_track(self, track: dict, target_bpm: float) -> tuple[str, float]:
        """
        Time-stretch track to match target BPM.
        
        Args:
            track: Track metadata dict
            target_bpm: Target BPM to match
        
        Returns:
            (warped_file_path, actual_ratio)
        
        Raises:
            FileNotFoundError: If track file doesn't exist
        
        Note:
            Ratio semantics: ratio > 1.0 means sped up (shorter duration)
            Physical time = logical time / ratio
        """
        filename = track.get('filename', '')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Track file not found: {filename}")
        
        track_bpm = track.get('bpm', target_bpm)
        if track_bpm <= 0:
            track_bpm = target_bpm
        
        ratio = target_bpm / track_bpm
        
        # If out of stretch range, don't stretch
        if ratio < MIN_STRETCH_RATIO or ratio > MAX_STRETCH_RATIO:
            ratio = 1.0
        
        # If essentially no stretch needed
        if 0.99 <= ratio <= 1.01:
            if filename.lower().endswith('.wav'):
                return os.path.abspath(filename), 1.0
            
            # Convert to WAV
            tmp = os.path.abspath(f"temp_sync_deck_b_{uuid.uuid4().hex[:8]}.wav")
            self._temp_files.append(tmp)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(filename, sr=44100, mono=False)
            
            if y.ndim == 1:
                y = np.array([y, y])
            
            sf.write(tmp, y.T.astype(np.float32), 44100)
            return tmp, 1.0
        
        # Time-stretch
        tmp = os.path.abspath(f"temp_sync_deck_b_{uuid.uuid4().hex[:8]}.wav")
        self._temp_files.append(tmp)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(filename, sr=44100, mono=False)
        
        if y.ndim == 1:
            y = np.array([y, y])
        
        yw = np.array([librosa.effects.time_stretch(c, rate=ratio) for c in y])
        sf.write(tmp, yw.T.astype(np.float32), 44100)
        
        return tmp, ratio
    
    def rescale_track_timestamps(self, track: dict, ratio: float) -> dict:
        """
        Rescale all timestamps in track metadata after time-stretching.
        
        Args:
            track: Original track metadata
            ratio: Time-stretch ratio (physical_time = logical_time / ratio)
        
        Returns:
            Deep copy of track with rescaled timestamps
        """
        if abs(ratio - 1.0) < 0.005:
            return track
        
        track_copy = copy.deepcopy(track)
        inv = 1.0 / ratio
        
        # Phrase timestamps
        for key in ('phrases', 'phrase_map', 'phrase_analysis', 'phrase_windows'):
            for p in track_copy.get(key, []):
                for fld in ('start', 'end'):
                    if fld in p:
                        p[fld] = float(p[fld]) * inv
        
        # Structure map
        for s in track_copy.get('structure_map', []):
            for fld in ('start', 'end'):
                if fld in s:
                    s[fld] = float(s[fld]) * inv
        
        # Stems
        stems = track_copy.get('stems', {})
        if 'vocal_regions' in stems:
            stems['vocal_regions'] = [(s * inv, e * inv) for s, e in stems['vocal_regions']]
        if 'log_drum_hits' in stems:
            stems['log_drum_hits'] = [t * inv for t in stems['log_drum_hits']]
        
        # Best exit/entry phrases
        for key in ('best_exit_phrases', 'best_entry_phrases', 'piano_entries'):
            if key in track_copy:
                track_copy[key] = [float(t) * inv for t in track_copy[key]]
        
        # Zones
        zones = track_copy.get('zones', {})
        for zk in ('optimal_mix_out', 'optimal_mix_in'):
            if zk in zones:
                zones[zk] = float(zones[zk]) * inv
        
        # First beat
        if 'first_beat_time' in track_copy:
            track_copy['first_beat_time'] = float(track_copy['first_beat_time']) * inv
        
        # Duration and BPM
        track_copy['_duration'] = track_copy.get('_duration', 300.0) * inv
        track_copy['bpm'] = track_copy.get('bpm', 112.0) * ratio
        track_copy['_stretch_ratio'] = ratio
        
        return track_copy
    
    def cleanup_temp_files(self, keep_last: bool = True):
        """
        Delete temporary warped files.
        
        Args:
            keep_last: If True, keep the most recent 2 files (for safety)
        """
        if not self._temp_files:
            return
        
        safe_count = 2 if keep_last else 0
        files_to_delete = self._temp_files[:-safe_count] if safe_count else self._temp_files
        
        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception:
                pass
        
        self._temp_files = self._temp_files[-safe_count:] if safe_count else []