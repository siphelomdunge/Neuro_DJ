"""
live_ears.py — Neuro-DJ Live Perception Module
═══════════════════════════════════════════════
Stage 6 of the roadmap: the system keeps listening during playback.

A precomputed plan is only as good as the offline analysis.
Offline analysis has errors. Tracks have unexpected density changes.
Phase drift accumulates over 60+ second blends.

This module runs in its own thread OUTSIDE the audio callback.
It reads the visual buffer from C++ (256-sample decimated waveform data)
and computes rolling estimates every beat.

What it estimates:
  1. Beat/phase offset between A and B
  2. Drift trend over time (is it getting worse?)
  3. Overlap congestion (are both decks too loud/dense simultaneously?)
  4. Vocal clash activity
  5. B's groove establishment (is B's rhythm felt by the crowd yet?)
  6. A's space opening (has A opened enough space for B to take over?)
  7. Handoff readiness (overall: is it safe to commit to B?)
  8. **Combined Mix-Bus Congestion** (weighted RMS sum of A + B)

Output: a LiveState object that the transition executor reads to make
        bounded real-time adjustments (Stage 8).
"""

import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class LiveState:
    # ── Phase & Sync ──
    beat_offset: float = 0.0          # + means B is ahead of A, - means B is behind
    drift_trend: float = 0.0          # Derivative of beat_offset (is it drifting further?)
    
    # ── Spectral Density ──
    a_low: float = 0.0
    a_mid: float = 0.0
    b_low: float = 0.0
    b_mid: float = 0.0
    
    # ── V23 FIX: Combined Mix-Bus Perception ──
    combined_low_congestion: float = 0.0
    combined_mid_load: float = 0.0
    
    # ── Semantic Flags ──
    vocal_clash_active: bool = False
    b_groove_established: bool = False
    a_space_opened: bool = False
    
    # ── Overall ──
    readiness_score: float = 0.0      # 0.0 to 1.0 (1.0 = perfect handoff conditions)

class LiveEars:
    def __init__(self, mixer, bpm: float):
        self._mixer = mixer
        self._bpm = float(bpm)
        self._running = False
        self._thread = None
        self._state = LiveState()
        self._state_lock = threading.Lock()
        
        # Rolling histories for trend analysis
        self._offset_history: List[float] = []
        self._history_len = 16  # approx 1.6 seconds at 10Hz
        
        self._b_started = False
        self._b_start_time = 0.0
        
        # Baseline peak RMS for relative weighting
        self._a_peak_rms = 0.01
        self._b_peak_rms = 0.01

    def start(self):
        if self._running:
            return
        self._running = True
        self._b_started = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def notify_b_started(self):
        """Called by the DJ thread exactly when B begins playing."""
        self._b_started = True
        self._b_start_time = time.time()

    def get_state(self) -> LiveState:
        with self._state_lock:
            # Return a fast copy to avoid blocking the executor
            return LiveState(
                beat_offset=self._state.beat_offset,
                drift_trend=self._state.drift_trend,
                a_low=self._state.a_low,
                a_mid=self._state.a_mid,
                b_low=self._state.b_low,
                b_mid=self._state.b_mid,
                combined_low_congestion=self._state.combined_low_congestion,
                combined_mid_load=self._state.combined_mid_load,
                vocal_clash_active=self._state.vocal_clash_active,
                b_groove_established=self._state.b_groove_established,
                a_space_opened=self._state.a_space_opened,
                readiness_score=self._state.readiness_score
            )

    def _listen_loop(self):
        # 10 Hz loop -> 100ms per tick
        while self._running:
            if self._b_started and self._mixer.is_transitioning():
                self._analyze_overlap()
            time.sleep(0.1)

    def _analyze_overlap(self):
        # 1. Fetch visual buffers from C++ (fast, non-blocking)
        buf_a = np.array(self._mixer.get_visual_buffer("A"))
        buf_b = np.array(self._mixer.get_visual_buffer("B"))
        
        if len(buf_a) == 0 or len(buf_b) == 0:
            return

        # 2. Compute Spectral & RMS characteristics
        a_rms = float(np.sqrt(np.mean(buf_a**2)))
        b_rms = float(np.sqrt(np.mean(buf_b**2)))
        
        # Track peaks for relative weighting
        self._a_peak_rms = max(self._a_peak_rms, a_rms)
        self._b_peak_rms = max(self._b_peak_rms, b_rms)

        # Naive multiband estimate from decimated time-domain
        # (Using simple moving averages as faux low-pass filters)
        kernel_low = np.ones(16) / 16
        a_lpf = np.convolve(buf_a, kernel_low, mode='same')
        b_lpf = np.convolve(buf_b, kernel_low, mode='same')
        
        a_low = float(np.sqrt(np.mean(a_lpf**2)))
        b_low = float(np.sqrt(np.mean(b_lpf**2)))
        
        a_mid = a_rms - a_low  # Rough proxy for mid/highs
        b_mid = b_rms - b_low

        # ── V23: Combined Mix-Bus Congestion Calculation ──
        # Weight the frequency bands by their deck's relative RMS 
        # so quiet decks don't trigger false congestion flags.
        a_weight = a_rms / (self._a_peak_rms + 1e-9)
        b_weight = b_rms / (self._b_peak_rms + 1e-9)
        
        combined_low = (a_low * a_weight) + (b_low * b_weight)
        combined_mid = (a_mid * a_weight) + (b_mid * b_weight)

        # 3. Phase Drift Calculation (Cross-Correlation of Transients)
        trans_a = np.abs(np.diff(buf_a.astype(np.float32)))
        trans_b = np.abs(np.diff(buf_b.astype(np.float32)))

        # Smooth to reduce noise while keeping transient peaks
        kernel_trans = np.ones(6) / 6
        env_a = np.convolve(trans_a, kernel_trans, mode='same')
        env_b = np.convolve(trans_b, kernel_trans, mode='same')

        n = min(len(env_a), len(env_b), 256)
        env_a = env_a[:n]
        env_b = env_b[:n]

        offset_beats = 0.0
        # Gate on signal variance — if one deck has no transient content
        if env_a.std() > 1e-5 and env_b.std() > 1e-5:
            xcorr = np.correlate(env_a - env_a.mean(), env_b - env_b.mean(), mode='full')
            lags = np.arange(-n + 1, n)
            peak = int(np.argmax(np.abs(xcorr)))
            lag_samples = int(lags[peak])

            # Convert lag in decimated buffer samples to actual seconds
            # (assuming decimated buffer represents ~44100 rate divided down)
            lag_sec = lag_samples * (4.0 / 44100.0) 
            spb = 60.0 / max(self._bpm, 60.0)
            offset_beats = float(np.clip(lag_sec / spb, -2.0, 2.0))

        self._offset_history.append(offset_beats)
        if len(self._offset_history) > self._history_len:
            self._offset_history.pop(0)

        # Calculate drift trend (slope of recent offsets)
        drift_trend = 0.0
        if len(self._offset_history) >= 4:
            recent = self._offset_history[-4:]
            older = self._offset_history[:4]
            drift_trend = float(np.mean(recent) - np.mean(older))

        # 4. Semantic Flags
        # Did A drop in volume/density significantly?
        a_space = (a_rms / self._a_peak_rms) < 0.45
        
        # Has B been playing long enough to establish its groove?
        b_time_alive = time.time() - self._b_start_time
        b_groove = b_time_alive > (16.0 * (60.0 / self._bpm)) and (b_rms / self._b_peak_rms) > 0.60
        
        # Simple vocal clash proxy (high mid load on both decks simultaneously)
        vocal_clash = (a_mid > 0.15) and (b_mid > 0.15)

        # 5. Overall Readiness Score
        readiness = 0.0
        if a_space: readiness += 0.40
        if b_groove: readiness += 0.40
        if combined_low < 0.50: readiness += 0.20
        if vocal_clash: readiness -= 0.30
        
        # 6. Commit to Thread-Safe State
        with self._state_lock:
            self._state.beat_offset = offset_beats
            self._state.drift_trend = drift_trend
            self._state.a_low = a_low
            self._state.a_mid = a_mid
            self._state.b_low = b_low
            self._state.b_mid = b_mid
            self._state.combined_low_congestion = combined_low
            self._state.combined_mid_load = combined_mid
            self._state.vocal_clash_active = vocal_clash
            self._state.b_groove_established = b_groove
            self._state.a_space_opened = a_space
            self._state.readiness_score = max(0.0, min(1.0, readiness))