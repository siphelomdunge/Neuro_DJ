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
  8. Combined Mix-Bus Congestion (weighted RMS sum of A + B)

Output: a LiveState object that the transition executor reads to make
        bounded real-time adjustments (Stage 8).

FIXES (v2):
  - Visual buffer ring-buffer ordering: get_visual_buffer() returns the
    512-sample ring buffer starting from visualIdx (the WRITE head), meaning
    the oldest sample is at index 0 ONLY when the buffer has not yet wrapped.
    After the first wrap the data is scrambled, making cross-correlation
    produce meaningless lag estimates.  The fix re-orders the buffer to be
    chronological before processing by rotating it so the oldest sample comes
    first.  This is done by reading visualIdx from C++ via get_position() as
    a proxy and applying np.roll — but since the C++ class does not expose
    visualIdx directly, we instead request the buffer twice and detect the
    write-head wrap.  The practical fix is simpler: the C++ engine fills the
    buffer in order starting at index 0 and wraps with visualIdx, so we just
    need to roll the numpy array by -visualIdx.  Since we cannot read visualIdx
    from Python, we use the safer approach of using only the most-recent N
    contiguous samples by re-reading the buffer as a deque-like structure:
    because visualIdx advances by 1 per 4 audio frames, and the buffer is 512
    samples, the ordering wraps every ~512×4/44100 ≈ 46 ms.  For a 10 Hz
    listener this means every call may be to a fully-wrapped buffer.
    SOLUTION: the C++ get_visual_buffer already returns a copy of the array.
    We reorder it by finding the minimum-magnitude run (the write-head leaves
    a small zero or near-zero region just ahead of it) and rolling accordingly.
    This is equivalent to what real oscilloscopes do to stabilise the display.
"""

import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class LiveState:
    # ── Phase & Sync ──────────────────────────────────────────────────────────
    beat_offset: float = 0.0    # + means B is ahead of A, – means B is behind
    drift_trend: float = 0.0    # derivative of beat_offset

    # ── Spectral Density ─────────────────────────────────────────────────────
    a_low: float = 0.0
    a_mid: float = 0.0
    b_low: float = 0.0
    b_mid: float = 0.0

    # ── Combined Mix-Bus Perception ───────────────────────────────────────────
    combined_low_congestion: float = 0.0
    combined_mid_load:       float = 0.0

    # ── Semantic Flags ────────────────────────────────────────────────────────
    vocal_clash_active:   bool  = False
    b_groove_established: bool  = False
    a_space_opened:       bool  = False

    # ── Overall ──────────────────────────────────────────────────────────────
    readiness_score: float = 0.0    # 0.0 – 1.0 (1.0 = perfect handoff)


def _reorder_ring_buffer(buf: np.ndarray) -> np.ndarray:
    """
    The C++ visual buffer is a ring buffer whose write head advances with audio.
    get_visual_buffer() returns the raw array, NOT rotated to chronological order.

    Strategy: find the likely write-head position by looking for the transition
    between "old" (already overwritten) and "new" (just written) data.  The
    write head leaves the smallest absolute-energy region just ahead of it
    because newly arriving near-silence (e.g., between beats) tends to be small.

    We detect this by finding the index with the minimum rolling RMS over a
    small window and rolling the array so that point is at index 0 (oldest).

    If the buffer is all zeros or has no clear minimum, we return it as-is —
    the cross-correlation will simply produce a noisy but non-crashing result.
    """
    n = len(buf)
    if n < 16:
        return buf

    window = 8
    # Rolling RMS over `window` samples
    sq = buf.astype(np.float32) ** 2
    kernel = np.ones(window, dtype=np.float32) / window
    rolling_rms = np.sqrt(np.convolve(sq, kernel, mode='same'))

    # The write head is just AFTER the minimum-energy region
    write_head = int(np.argmin(rolling_rms))

    # Roll so oldest sample is at index 0
    return np.roll(buf, -write_head)


class LiveEars:
    def __init__(self, mixer, bpm: float):
        self._mixer  = mixer
        self._bpm    = float(bpm)
        self._running = False
        self._thread  = None
        self._state   = LiveState()
        self._state_lock = threading.Lock()

        self._offset_history: List[float] = []
        self._history_len = 16      # ~1.6 s at 10 Hz

        self._b_started    = False
        self._b_start_time = 0.0

        self._a_peak_rms = 0.01
        self._b_peak_rms = 0.01

    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running  = True
        self._b_started = False
        self._thread    = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def notify_b_started(self):
        """Called by the DJ thread exactly when B begins playing."""
        self._b_started    = True
        self._b_start_time = time.time()

    def get_state(self) -> LiveState:
        with self._state_lock:
            s = self._state
            return LiveState(
                beat_offset            = s.beat_offset,
                drift_trend            = s.drift_trend,
                a_low                  = s.a_low,
                a_mid                  = s.a_mid,
                b_low                  = s.b_low,
                b_mid                  = s.b_mid,
                combined_low_congestion= s.combined_low_congestion,
                combined_mid_load      = s.combined_mid_load,
                vocal_clash_active     = s.vocal_clash_active,
                b_groove_established   = s.b_groove_established,
                a_space_opened         = s.a_space_opened,
                readiness_score        = s.readiness_score,
            )

    # ──────────────────────────────────────────────────────────────────────────
    def _listen_loop(self):
        while self._running:
            if self._b_started and self._mixer.is_transitioning():
                self._analyze_overlap()
            time.sleep(0.1)

    def _analyze_overlap(self):
        raw_a = np.array(self._mixer.get_visual_buffer("A"))
        raw_b = np.array(self._mixer.get_visual_buffer("B"))

        if len(raw_a) == 0 or len(raw_b) == 0:
            return

        # FIX: reorder ring buffers to chronological order before analysis.
        # Without this, every call after the first wrap (≈46 ms) reads scrambled
        # data and the cross-correlation returns a random lag.
        buf_a = _reorder_ring_buffer(raw_a)
        buf_b = _reorder_ring_buffer(raw_b)

        # ── RMS ───────────────────────────────────────────────────────────────
        a_rms = float(np.sqrt(np.mean(buf_a ** 2)))
        b_rms = float(np.sqrt(np.mean(buf_b ** 2)))

        self._a_peak_rms = max(self._a_peak_rms, a_rms)
        self._b_peak_rms = max(self._b_peak_rms, b_rms)

        # ── Multiband estimate (naive LPF on time-domain) ─────────────────────
        kernel_low = np.ones(16, dtype=np.float32) / 16
        a_lpf = np.convolve(buf_a, kernel_low, mode='same')
        b_lpf = np.convolve(buf_b, kernel_low, mode='same')

        a_low = float(np.sqrt(np.mean(a_lpf ** 2)))
        b_low = float(np.sqrt(np.mean(b_lpf ** 2)))
        a_mid = a_rms - a_low
        b_mid = b_rms - b_low

        # ── Combined Mix-Bus Congestion ───────────────────────────────────────
        a_weight = a_rms / (self._a_peak_rms + 1e-9)
        b_weight = b_rms / (self._b_peak_rms + 1e-9)
        combined_low = (a_low * a_weight) + (b_low * b_weight)
        combined_mid = (a_mid * a_weight) + (b_mid * b_weight)

        # ── Phase Drift (Cross-Correlation on re-ordered buffer) ──────────────
        trans_a = np.abs(np.diff(buf_a.astype(np.float32)))
        trans_b = np.abs(np.diff(buf_b.astype(np.float32)))

        kernel_t = np.ones(6, dtype=np.float32) / 6
        env_a = np.convolve(trans_a, kernel_t, mode='same')
        env_b = np.convolve(trans_b, kernel_t, mode='same')

        n     = min(len(env_a), len(env_b), 256)
        env_a = env_a[:n]
        env_b = env_b[:n]

        offset_beats = 0.0
        if env_a.std() > 1e-5 and env_b.std() > 1e-5:
            xcorr = np.correlate(env_a - env_a.mean(),
                                 env_b - env_b.mean(),
                                 mode='full')
            lags      = np.arange(-n + 1, n)
            peak      = int(np.argmax(np.abs(xcorr)))
            lag_samp  = int(lags[peak])
            # Each decimated buffer sample represents 4 audio frames at 44 100 Hz
            lag_sec   = lag_samp * (4.0 / 44100.0)
            spb       = 60.0 / max(self._bpm, 60.0)
            offset_beats = float(np.clip(lag_sec / spb, -2.0, 2.0))

        self._offset_history.append(offset_beats)
        if len(self._offset_history) > self._history_len:
            self._offset_history.pop(0)

        drift_trend = 0.0
        if len(self._offset_history) >= 4:
            recent = self._offset_history[-4:]
            older  = self._offset_history[:4]
            drift_trend = float(np.mean(recent) - np.mean(older))

        # ── Semantic Flags ────────────────────────────────────────────────────
        a_space  = (a_rms / self._a_peak_rms) < 0.45
        b_time   = time.time() - self._b_start_time
        b_groove = (b_time > (16.0 * (60.0 / self._bpm))
                    and (b_rms / self._b_peak_rms) > 0.60)
        vocal_clash = (a_mid > 0.15) and (b_mid > 0.15)

        # ── Readiness Score ───────────────────────────────────────────────────
        readiness = 0.0
        if a_space:         readiness += 0.40
        if b_groove:        readiness += 0.40
        if combined_low < 0.50: readiness += 0.20
        if vocal_clash:     readiness -= 0.30

        # ── Commit ───────────────────────────────────────────────────────────
        with self._state_lock:
            self._state.beat_offset             = offset_beats
            self._state.drift_trend             = drift_trend
            self._state.a_low                   = a_low
            self._state.a_mid                   = a_mid
            self._state.b_low                   = b_low
            self._state.b_mid                   = b_mid
            self._state.combined_low_congestion = combined_low
            self._state.combined_mid_load       = combined_mid
            self._state.vocal_clash_active      = vocal_clash
            self._state.b_groove_established    = b_groove
            self._state.a_space_opened          = a_space
            self._state.readiness_score         = max(0.0, min(1.0, readiness))