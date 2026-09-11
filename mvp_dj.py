#!/usr/bin/env python3
"""
Neuro-DJ safe MVP
==================

A deliberately small, deterministic DJ runner for real-world playback.
It is separate from the experimental neuro_gui.py engine so the original
research code remains available while the party path stays conservative.

Design rules:
  * the audio callback never touches the network, disk, or analysis code;
  * only the next track is downloaded/decoded in the background;
  * genre is explicit (playlist metadata or the folder name);
  * BPM/key are optional metadata, never guessed with expensive analysis;
  * transitions use one fixed equal-power crossfade;
  * a local/cache fallback is preferred over an ambitious effect.

Examples:
  python mvp_dj.py --folder ./music --dry-run
  python mvp_dj.py --folder ./music --transition-beats 32
  python mvp_dj.py --playlist mvp_playlist.json --cache-dir .mvp_cache

A playlist can be JSON, standard M3U/M3U8, or XSPF. JSON can be a list or
{"tracks": [...]}. Each JSON track can contain path or url, title, artist,
genre, bpm, key, cue_in and energy. M3U entries preserve their file order and
can use #EXTINF artist/title metadata. XSPF files exported by VLC and similar
players are read from their track locations. A URL must be a direct, legally
playable audio URL supplied by the provider; this runner does not bypass
provider authentication or download protections.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

try:  # Optional at import time so --dry-run and selector tests stay lightweight.
    import numpy as np
except ImportError:  # pragma: no cover - exercised on minimal installations
    np = None

try:  # Optional at import time; only required for actual playback.
    import soundfile as sf
except ImportError:  # pragma: no cover - exercised on minimal installations
    sf = None

try:  # Compiled DSP helper used by the transition EQ; dry-run does not need it.
    from scipy.signal import lfilter
except ImportError:  # pragma: no cover - exercised on minimal installations
    lfilter = None


SAMPLE_RATE = 44_100
CHANNELS = 2
DEFAULT_BPM = 120.0
DEFAULT_TRANSITION_BEATS = 32
MIN_BPM = 60.0
MAX_BPM = 200.0
AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg", ".m4a", ".aac"}
PathLike = Union[str, Path]


class MVPError(RuntimeError):
    """An expected, user-actionable MVP failure."""


class AudioDependencyError(MVPError):
    """Raised when live playback dependencies are not installed."""


@dataclass
class Track:
    """Small metadata record used by the MVP selector and cache."""

    track_id: str
    source: str
    title: str
    artist: str = ""
    genre: str = "open"
    bpm: Optional[float] = None
    key: str = ""
    energy: float = 0.5
    cue_in: float = 0.0
    duration: Optional[float] = None

    @property
    def is_url(self) -> bool:
        return self.source.startswith(("http://", "https://"))

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        base_dir: Optional[Path] = None,
        genre_hint: str = "open",
    ) -> "Track":
        source_value = value.get("path") or value.get("file") or value.get("filename") or value.get("url")
        if not source_value:
            raise MVPError("Every playlist track needs a path or url")
        source = str(source_value)
        if not source.startswith(("http://", "https://")):
            source_path = Path(source).expanduser()
            if base_dir and not source_path.is_absolute():
                source_path = base_dir / source_path
            source = str(source_path.resolve())

        source_name = Path(unquote(urlparse(source).path)).stem or "untitled"
        title = str(value.get("title") or source_name)
        artist = str(value.get("artist") or "")
        genre = _normalise_genre(value.get("genre") or genre_hint)
        bpm = _coerce_bpm(value.get("bpm"))
        if bpm is None:
            bpm = _bpm_from_name(source_name)
        key = str(value.get("key") or "").strip()
        energy = _coerce_energy(value.get("energy", 0.5))
        cue_in = max(0.0, _coerce_float(value.get("cue_in"), 0.0))
        duration = _coerce_float(value.get("duration", value.get("_duration")), None)
        track_id = str(value.get("id") or hashlib.sha1(source.encode("utf-8")).hexdigest()[:16])
        return cls(track_id, source, title, artist, genre, bpm, key, energy, cue_in, duration)


def _coerce_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bpm(value: Any) -> Optional[float]:
    result = _coerce_float(value, None)
    if result is None or not MIN_BPM <= result <= MAX_BPM:
        return None
    return round(result, 2)


def _coerce_energy(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"high", "peak", "energetic"}:
            return 0.85
        if text in {"low", "low/chill", "chill", "warmup"}:
            return 0.25
        value = _coerce_float(value, 0.5)
    result = _coerce_float(value, 0.5)
    return max(0.0, min(1.0, result if result is not None else 0.5))


def _normalise_genre(value: Any) -> str:
    text = str(value or "open").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text or "open"


def _bpm_from_name(name: str) -> Optional[float]:
    patterns = (
        r"(?:bpm|tempo)[ _-]*(\d{2,3}(?:\.\d+)?)",
        r"[@_ -](\d{2,3}(?:\.\d+)?)(?:\s*bpm)?(?:\.|_|-|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            bpm = _coerce_bpm(match.group(1))
            if bpm is not None:
                return bpm
    return None


def _folder_genre(path: Path, root: Path) -> str:
    try:
        relative_parent = path.resolve().parent.relative_to(root.resolve())
        if relative_parent.parts:
            return _normalise_genre(relative_parent.parts[0])
    except ValueError:
        pass
    return "open"


def discover_folder(folder: PathLike) -> list[Track]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise MVPError(f"Music folder does not exist: {root}")

    tracks: list[Track] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.name.endswith("_ready.wav") or ".mvp_cache" in path.parts:
            continue
        tracks.append(
            Track.from_mapping(
                {"path": str(path)},
                genre_hint=_folder_genre(path, root),
            )
        )
    if not tracks:
        raise MVPError(f"No supported audio files found under {root}")
    return tracks


def load_m3u(path: PathLike) -> list[Track]:
    """Load a standard M3U/M3U8 playlist without requiring JSON editing."""
    playlist_path = Path(path).expanduser().resolve()
    try:
        lines = playlist_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except FileNotFoundError as exc:
        raise MVPError(f"Playlist does not exist: {playlist_path}") from exc

    tracks: list[Track] = []
    pending_title = ""
    pending_artist = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.upper() == "#EXTM3U":
            continue
        if line.upper().startswith("#EXTINF:"):
            description = line.split(",", 1)[1].strip() if "," in line else ""
            if " - " in description:
                pending_artist, pending_title = description.split(" - ", 1)
            else:
                pending_artist, pending_title = "", description
            continue
        if line.startswith("#"):
            continue

        is_url = line.startswith(("http://", "https://"))
        if is_url:
            source = line
            genre_hint = "open"
        else:
            source_path = Path(line).expanduser()
            if not source_path.is_absolute():
                source_path = playlist_path.parent / source_path
            source_path = source_path.resolve()
            source = str(source_path)
            genre_hint = _folder_genre(source_path, playlist_path.parent)

        tracks.append(
            Track.from_mapping(
                {
                    "path" if not is_url else "url": source,
                    "title": pending_title,
                    "artist": pending_artist,
                },
                genre_hint=genre_hint,
            )
        )
        pending_title = ""
        pending_artist = ""

    if not tracks:
        raise MVPError(f"Playlist contains no playable entries: {playlist_path}")
    return tracks


def load_xspf(path: PathLike) -> list[Track]:
    """Load XSPF playlists exported by VLC and other music players."""
    playlist_path = Path(path).expanduser().resolve()
    try:
        root = ET.fromstring(playlist_path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError as exc:
        raise MVPError(f"Playlist does not exist: {playlist_path}") from exc
    except ET.ParseError as exc:
        raise MVPError(f"Invalid XSPF playlist: {exc}") from exc

    def child_text(element: ET.Element, name: str) -> str:
        for child in element:
            if child.tag.rsplit("}", 1)[-1] == name:
                return (child.text or "").strip()
        return ""

    tracks: list[Track] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "track":
            continue
        location = child_text(element, "location")
        if not location:
            continue

        parsed = urlparse(location)
        is_http = parsed.scheme.lower() in {"http", "https"}
        if parsed.scheme.lower() == "file":
            # XSPF commonly stores local files as file:///absolute/path.
            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                location = f"//{parsed.netloc}{unquote(parsed.path)}"
            else:
                location = unquote(parsed.path)
            is_http = False

        value: dict[str, Any] = {
            "url" if is_http else "path": location,
            "title": child_text(element, "title"),
            "artist": child_text(element, "creator") or child_text(element, "artist"),
            "genre": child_text(element, "genre"),
            "key": child_text(element, "key"),
            "bpm": child_text(element, "bpm"),
            "cue_in": child_text(element, "cue_in") or child_text(element, "cue-in"),
        }
        duration_ms = _coerce_float(child_text(element, "duration"), None)
        if duration_ms is not None:
            value["duration"] = duration_ms / 1000.0
        tracks.append(Track.from_mapping(value, base_dir=playlist_path.parent))

    if not tracks:
        raise MVPError(f"XSPF playlist contains no playable tracks: {playlist_path}")
    return tracks


def load_playlist(path: PathLike) -> list[Track]:
    playlist_path = Path(path).expanduser().resolve()
    if playlist_path.suffix.lower() in {".m3u", ".m3u8"}:
        return load_m3u(playlist_path)
    if playlist_path.suffix.lower() == ".xspf":
        return load_xspf(playlist_path)
    try:
        payload = json.loads(playlist_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MVPError(f"Playlist does not exist: {playlist_path}") from exc
    except json.JSONDecodeError as exc:
        raise MVPError(f"Invalid playlist JSON: {exc}") from exc

    items = payload.get("tracks") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise MVPError("Playlist must be a list or an object with a 'tracks' list")

    tracks = []
    for item in items:
        if not isinstance(item, dict):
            raise MVPError("Every playlist item must be an object")
        tracks.append(Track.from_mapping(item, base_dir=playlist_path.parent))
    if not tracks:
        raise MVPError("Playlist contains no tracks")
    return tracks


class TrackSelector:
    """Deterministic, explainable next-track selection."""

    def __init__(
        self,
        tracks: Iterable[Track],
        *,
        strict_genre: bool = True,
        manual_order: bool = False,
    ):
        self.tracks = list(tracks)
        self.strict_genre = strict_genre
        self.manual_order = manual_order
        self._claimed: set[str] = set()
        self._recent_artists: list[str] = []
        self._recent_energies: list[float] = []

    def claim(self, track: Track) -> None:
        self._claimed.add(track.track_id)
        artist = _artist_key(track.artist)
        if artist:
            self._recent_artists = [item for item in self._recent_artists if item != artist]
            self._recent_artists.append(artist)
            self._recent_artists = self._recent_artists[-3:]
        self._recent_energies.append(track.energy)
        self._recent_energies = self._recent_energies[-3:]

    def release(self, track: Track) -> None:
        self._claimed.discard(track.track_id)

    def reset(self) -> None:
        self._claimed.clear()

    def add_tracks(self, tracks: Iterable[Track]) -> int:
        """Add newly appended queue items without disturbing the current set."""
        known = {track.track_id for track in self.tracks}
        added = 0
        for track in tracks:
            if track.track_id not in known:
                self.tracks.append(track)
                known.add(track.track_id)
                added += 1
        return added

    def choose(self, current: Optional[Track]) -> Optional[Track]:
        candidates = [t for t in self.tracks if t.track_id not in self._claimed]
        if not candidates:
            return None

        # Manual mode is intentionally absolute: playlist order wins over
        # genre/BPM/key scoring. This is useful when the operator has planned
        # the party arc and only wants Neuro-DJ to perform the transitions.
        if self.manual_order:
            return candidates[0]

        same_genre = [t for t in candidates if current and _same_genre(current, t)]
        fallback = bool(current and current.genre != "open" and not same_genre)
        pool = same_genre if (same_genre and self.strict_genre) else candidates
        ranked = sorted(
            pool,
            key=lambda track: self.score(current, track, fallback=fallback),
            reverse=True,
        )
        return ranked[0] if ranked else None

    def score(self, current: Optional[Track], candidate: Track, *, fallback: bool = False) -> float:
        if current is None:
            # Stable opener choice: prefer a known BPM and avoid an arbitrary shuffle.
            return (10.0 if candidate.bpm is not None else 0.0) - candidate.energy

        score = 0.0
        if _same_genre(current, candidate):
            score += 60.0
        elif fallback:
            score -= 25.0

        if current.bpm is None or candidate.bpm is None:
            score += 8.0  # Unknown tempo is allowed, but never preferred to a known match.
        else:
            diff = abs(current.bpm - candidate.bpm)
            if diff <= 1.0:
                score += 25.0
            elif diff <= 3.0:
                score += 18.0
            elif diff <= 5.0:
                score += 8.0
            else:
                score -= min(25.0, diff * 2.0)

        score += _key_score(current.key, candidate.key)
        energy_delta = abs(current.energy - candidate.energy)
        score += max(0.0, 10.0 - energy_delta * 12.0)

        # Low-risk variety: do not play the same artist back-to-back when an
        # equally compatible alternative exists. This changes selection only;
        # it never compromises genre or tempo safety.
        candidate_artist = _artist_key(candidate.artist)
        current_artist = _artist_key(current.artist)
        if candidate_artist and candidate_artist == current_artist:
            score -= 18.0
        elif candidate_artist and candidate_artist in self._recent_artists[-2:]:
            score -= 7.0

        # If the last couple of claimed tracks have been flat in energy, give a
        # small bonus to a nearby lift/drop. Large jumps remain penalised by the
        # normal energy score above.
        if len(self._recent_energies) >= 2:
            recent_span = max(self._recent_energies[-2:]) - min(self._recent_energies[-2:])
            if recent_span < 0.08:
                if 0.12 <= energy_delta <= 0.35:
                    score += 5.0
                elif energy_delta < 0.08:
                    score -= 2.0

        if candidate.duration is not None and candidate.duration >= 120.0:
            score += 3.0
        return score


def _artist_key(value: str) -> str:
    return re.sub(r"\\s+", " ", (value or "").strip().lower())


def _same_genre(a: Track, b: Track) -> bool:
    # An explicit genre is required for a strict same-genre transition. The
    # special `open` value means "unknown", not "compatible with everything".
    # Two unknown tracks can still be played together as a last-resort crate.
    return a.genre == b.genre


def _key_score(a: str, b: str) -> float:
    """Small, forgiving Camelot-aware score. Unknown keys are not vetoed."""
    if not a or not b:
        return 6.0
    left, right = a.strip().upper(), b.strip().upper()
    camelot = re.compile(r"^(1[0-2]|[1-9])([AB])$")
    ma, mb = camelot.match(left), camelot.match(right)
    if ma and mb:
        na, nb = int(ma.group(1)), int(mb.group(1))
        mode_a, mode_b = ma.group(2), mb.group(2)
        distance = min(abs(na - nb), 12 - abs(na - nb))
        if distance == 0 and mode_a == mode_b:
            return 15.0
        if distance == 0 and mode_a != mode_b:
            return 10.0
        if distance == 1 and mode_a == mode_b:
            return 11.0
        return max(0.0, 8.0 - distance * 3.0)
    return 6.0 if left == right else 3.0


class AudioCache:
    """Downloads only explicitly supplied direct URLs and decodes to safe audio."""

    def __init__(self, root: PathLike, *, timeout: float = 30.0):
        self.root = Path(root).expanduser().resolve()
        self.raw_root = self.root / "raw"
        self.decoded_root = self.root / "decoded"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.decoded_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def source_path(self, track: Track) -> Path:
        if not track.is_url:
            path = Path(track.source)
            if not path.exists():
                raise MVPError(f"Track file does not exist: {path}")
            return path

        suffix = Path(unquote(urlparse(track.source).path)).suffix.lower()
        suffix = suffix if suffix in AUDIO_EXTENSIONS else ".audio"
        destination = self.raw_root / f"{track.track_id}{suffix}"
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        request = Request(track.source, headers={"User-Agent": "Neuro-DJ-MVP/1.0"})
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"   ↓ caching {track.title} from {urlparse(track.source).netloc}")
        try:
            with urlopen(request, timeout=self.timeout) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 256)
            temporary.replace(destination)
        except Exception as exc:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise MVPError(f"Could not cache {track.title}: {exc}") from exc
        return destination

    def load_for_mix(self, track: Track, *, target_bpm: Optional[float]) -> tuple[np.ndarray, Track]:
        _require_audio_dependencies()
        source = self.source_path(track)
        data, sample_rate = self._read_audio(source, track)
        data = _to_stereo(data)
        if sample_rate != SAMPLE_RATE:
            data = _resample(data, sample_rate, SAMPLE_RATE)

        cue_frames = int(max(0.0, track.cue_in) * SAMPLE_RATE)
        if cue_frames and cue_frames < len(data):
            data = data[cue_frames:]

        effective_bpm = track.bpm
        if target_bpm and track.bpm and abs(track.bpm - target_bpm) <= target_bpm * 0.08:
            ratio = target_bpm / track.bpm
            if abs(ratio - 1.0) > 0.002:
                data = _speed_change(data, ratio)
                effective_bpm = target_bpm

        data = _normalise(data)
        prepared = replace(
            track,
            bpm=effective_bpm,
            duration=len(data) / SAMPLE_RATE,
            cue_in=0.0,
        )
        return data.astype(np.float32, copy=False), prepared

    def _read_audio(self, source: Path, track: Track) -> tuple[np.ndarray, int]:
        try:
            data, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
            return data, int(sample_rate)
        except Exception as first_error:
            decoded = self.decoded_root / f"{track.track_id}.wav"
            if not decoded.exists():
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise MVPError(
                        f"Cannot decode {source.name}. Install ffmpeg for compressed audio."
                    ) from first_error
                temporary = decoded.with_suffix(".wav.part")
                command = [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    str(CHANNELS),
                    "-c:a",
                    "pcm_f32le",
                    str(temporary),
                ]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
                if completed.returncode != 0:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                    raise MVPError(
                        f"ffmpeg could not decode {source.name}: {completed.stderr.strip()}"
                    ) from first_error
                temporary.replace(decoded)
            data, sample_rate = sf.read(str(decoded), dtype="float32", always_2d=True)
            return data, int(sample_rate)


def _require_audio_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if sf is None:
        missing.append("soundfile")
    if lfilter is None:
        missing.append("scipy")
    if missing:
        raise AudioDependencyError(
            "Live playback needs " + ", ".join(missing) + ". Install requirements-mvp.txt."
        )


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] > 2:
        return data[:, :2]
    return data


def _resample(data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate or len(data) < 2:
        return data
    new_length = max(1, int(round(len(data) * target_rate / original_rate)))
    old_x = np.linspace(0.0, 1.0, len(data), endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_length, endpoint=False)
    channels = [np.interp(new_x, old_x, data[:, channel]) for channel in range(data.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def _speed_change(data: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0.0 or abs(ratio - 1.0) < 0.002:
        return data
    # Resampling changes pitch slightly, but is stable and acceptable for the MVP's
    # deliberately small BPM window. It also keeps beats aligned without a live DSP loop.
    new_length = max(1, int(round(len(data) / ratio)))
    old_x = np.linspace(0.0, 1.0, len(data), endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_length, endpoint=False)
    channels = [np.interp(new_x, old_x, data[:, channel]) for channel in range(data.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def _normalise(data: np.ndarray, target_rms: float = 0.125) -> np.ndarray:
    if len(data) == 0:
        raise MVPError("Audio file contains no samples")
    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms > 1e-6:
        data = data * min(1.5, target_rms / rms)
    peak = float(np.max(np.abs(data)))
    if peak > 0.96:
        data = data * (0.96 / peak)
    return np.clip(data, -1.0, 1.0)


class DeckEQ:
    """Lightweight three-band EQ with state retained across audio blocks."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # Two first-order low-pass filters give low, mid and high bands without
        # doing heavy spectral analysis in the audio callback.
        low_cut = 180.0
        high_cut = 4000.0
        self._low_alpha = 1.0 - math.exp(-2.0 * math.pi * low_cut / sample_rate)
        self._high_alpha = 1.0 - math.exp(-2.0 * math.pi * high_cut / sample_rate)
        self._low_zi = np.zeros((1, CHANNELS), dtype=np.float32)
        self._high_zi = np.zeros((1, CHANNELS), dtype=np.float32)

    def process(self, samples: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
        if len(samples) == 0:
            return samples
        # scipy.signal.lfilter is compiled code, so the callback processes a
        # complete block instead of running a Python filter loop per sample.
        low_b = np.array([self._low_alpha], dtype=np.float32)
        low_a = np.array([1.0, -(1.0 - self._low_alpha)], dtype=np.float32)
        high_b = np.array([self._high_alpha], dtype=np.float32)
        high_a = np.array([1.0, -(1.0 - self._high_alpha)], dtype=np.float32)
        low, self._low_zi = lfilter(low_b, low_a, samples, axis=0, zi=self._low_zi)
        high_lpf, self._high_zi = lfilter(
            high_b, high_a, samples, axis=0, zi=self._high_zi
        )
        high = samples - high_lpf
        mid = high_lpf - low
        low_gain, mid_gain, high_gain = gains
        return (
            low * low_gain + mid * mid_gain + high * high_gain
        ).astype(np.float32, copy=False)


class MixRenderer:
    """Two-buffer renderer with the conservative 32-beat bass-swap profile."""

    def __init__(
        self,
        first: np.ndarray,
        *,
        transition_frames: int,
        transition_beats: int = DEFAULT_TRANSITION_BEATS,
        bpm: float = DEFAULT_BPM,
        sample_rate: int = SAMPLE_RATE,
        mix_style: str = "club",
    ):
        self.sample_rate = sample_rate
        self.transition_frames = max(1, int(transition_frames))
        self.transition_beats = max(8, int(transition_beats))
        self.bpm = max(1.0, float(bpm))
        self.mix_style = mix_style if mix_style in {"club", "smooth"} else "club"
        self._frames_per_beat = self.sample_rate * 60.0 / self.bpm
        self._lock = threading.Lock()
        self._current = first
        self._current_pos = 0
        self._current_eq = DeckEQ(sample_rate)
        self._next: Optional[np.ndarray] = None
        self._next_pos = 0
        self._next_eq: Optional[DeckEQ] = None
        self._transition_start = max(0, len(first) - self.transition_frames)
        self._transition_length = max(1, len(first) - self._transition_start)
        self._hold_enabled = True
        self._hold_frames = max(1, int(sample_rate * 8.0))
        self._changed = threading.Event()
        self._ended = threading.Event()
        self._stopped = threading.Event()
        self._paused = False
        self.callback_errors: list[str] = []

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(1.0, max(0.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _profile(self, progress: float) -> tuple[float, ...]:
        """Return A/B EQ and fader gains for the requested transition phase.

        The profile follows the requested 32-beat plan:
          1-8: B enters with bass cut and restrained highs;
          9-16: B reaches full mids/highs while A highs soften;
          17-24: B teases in low end and A makes room;
          25-32: A fades and the low-end swap completes on beat 32.
        """
        if self.mix_style == "smooth":
            # R&B/pop material often has no clean log-drum phrase. Use a
            # shorter, gentle equal-power blend with no dramatic bass cut or
            # final slam. This is safer for vocals and sparse arrangements.
            t = min(1.0, max(0.0, progress))
            eased = self._smoothstep(t)
            a_fader = math.cos(t * math.pi / 2.0)
            b_fader = math.sin(t * math.pi / 2.0)
            b_low = 0.72 + 0.28 * self._smoothstep(min(1.0, t * 1.5))
            b_mid = 0.78 + 0.22 * eased
            b_high = 0.82 + 0.18 * eased
            a_high = 1.0 - 0.12 * eased
            return 1.0, 1.0, a_high, b_low, b_mid, b_high, a_fader, b_fader

        beat = min(float(self.transition_beats), max(0.0, progress * self.transition_beats))
        a_low = a_mid = a_high = 1.0
        b_low = 0.0
        b_mid = 0.25
        b_high = 0.35  # restrained high EQ during the intro
        a_fader = 1.0
        b_fader = 0.0

        if beat <= 8.0:
            b_fader = 0.70 * self._smoothstep(beat / 8.0)
        elif beat <= 16.0:
            b_fader = 0.70 + 0.30 * self._smoothstep((beat - 8.0) / 8.0)
            b_mid = 0.25 + 0.75 * self._smoothstep((beat - 8.0) / 8.0)
            b_high = 0.35 + 0.65 * self._smoothstep((beat - 8.0) / 8.0)
            a_high = 1.0 - 0.25 * self._smoothstep((beat - 8.0) / 8.0)
        elif beat <= 24.0:
            b_fader = 1.0
            b_mid = b_high = 1.0
            b_low = 0.35 * self._smoothstep((beat - 16.0) / 8.0)
            a_low = 1.0 - 0.45 * self._smoothstep((beat - 16.0) / 8.0)
            a_high = 0.75
        else:
            b_fader = 1.0
            b_mid = b_high = 1.0
            b_low = 0.35
            a_low = 0.55
            a_high = 0.75
            fade = self._smoothstep((beat - 24.0) / 8.0)
            a_fader = 1.0 - fade
            # The final beat is the deliberate low-end handoff.
            if beat >= 31.0:
                swap = self._smoothstep(beat - 31.0)
                b_low = 0.35 + 0.65 * swap
                a_low = 0.55 * (1.0 - swap)

        return a_low, a_mid, a_high, b_low, b_mid, b_high, a_fader, b_fader

    def set_next(self, audio: np.ndarray) -> bool:
        """Install a prepared track. Returns False if another track is queued."""
        with self._lock:
            if self._next is not None:
                return False
            if audio is None or len(audio) == 0:
                return False
            self._next = audio
            self._next_pos = 0
            self._next_eq = DeckEQ(self.sample_rate)
            remaining = max(1, len(self._current) - self._current_pos)
            self._transition_length = min(self.transition_frames, remaining)
            self._transition_start = len(self._current) - self._transition_length
            if self._current_pos > self._transition_start:
                self._transition_start = self._current_pos
                self._transition_length = max(1, len(self._current) - self._current_pos)
            return True

    def force_transition(self) -> None:
        with self._lock:
            if self._next is None:
                return
            self._transition_start = self._current_pos
            self._transition_length = max(1, len(self._current) - self._current_pos)

    def set_hold_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._hold_enabled = enabled

    def has_next(self) -> bool:
        with self._lock:
            return self._next is not None

    def current_position(self) -> float:
        with self._lock:
            return self._current_pos / self.sample_rate

    def consume_changed(self) -> bool:
        if self._changed.is_set():
            self._changed.clear()
            return True
        return False

    def is_ended(self) -> bool:
        return self._ended.is_set()

    def stop(self) -> None:
        self._stopped.set()

    def toggle_pause(self) -> bool:
        """Pause/resume without advancing the audio position."""
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @staticmethod
    def _safe_mix(a: np.ndarray, b: np.ndarray, a_gain: float, b_gain: float) -> np.ndarray:
        mixed = a * a_gain + b * b_gain
        # Leave headroom for two full-range tracks without hard clipping.
        return (np.tanh(mixed * 0.90) / 0.90).astype(np.float32, copy=False)

    def __call__(self, outdata, frames, time_info, status) -> None:  # sounddevice callback API
        try:
            outdata.fill(0)
            with self._lock:
                if self._paused:
                    return
                output_pos = 0
                while output_pos < frames:
                    if self._stopped.is_set():
                        break
                    if self._current_pos >= len(self._current):
                        if self._next is not None:
                            self._current = self._next
                            self._current_pos = self._next_pos
                            self._current_eq = self._next_eq or DeckEQ(self.sample_rate)
                            self._next = None
                            self._next_pos = 0
                            self._next_eq = None
                            self._transition_start = max(0, len(self._current) - self.transition_frames)
                            self._transition_length = min(self.transition_frames, len(self._current))
                            self._changed.set()
                        elif self._hold_enabled and len(self._current):
                            self._current_pos = max(0, len(self._current) - self._hold_frames)
                        else:
                            self._ended.set()
                            break

                    remaining = min(frames - output_pos, len(self._current) - self._current_pos)
                    if remaining <= 0:
                        continue

                    # Before the transition, only render A. This also advances
                    # A's EQ state so the handoff has no filter discontinuity.
                    if self._next is None or self._current_pos < self._transition_start:
                        pre = remaining
                        if self._next is not None:
                            pre = min(pre, self._transition_start - self._current_pos)
                        a_block = self._current[self._current_pos:self._current_pos + pre]
                        outdata[output_pos:output_pos + pre] = self._current_eq.process(
                            a_block, (1.0, 1.0, 1.0)
                        )
                        self._current_pos += pre
                        output_pos += pre
                        continue

                    # Render the overlap as one vectorised audio block. EQ is
                    # compiled DSP; Python only schedules the slowly changing
                    # gains once per callback block.
                    overlap = min(remaining, len(self._current) - self._current_pos)
                    a_start = self._current_pos
                    a_block = self._current[a_start:a_start + overlap]
                    b_start = self._next_pos
                    b_block = np.zeros((overlap, CHANNELS), dtype=np.float32)
                    available = max(0, min(overlap, len(self._next) - b_start))
                    if available:
                        b_block[:available] = self._next[b_start:b_start + available]

                    midpoint = a_start + overlap * 0.5
                    progress = (midpoint - self._transition_start) / max(1, self._transition_length)
                    a_low, a_mid, a_high, b_low, b_mid, b_high, a_fader, b_fader = self._profile(progress)
                    a_processed = self._current_eq.process(a_block, (a_low, a_mid, a_high))
                    b_processed = (self._next_eq or DeckEQ(self.sample_rate)).process(
                        b_block, (b_low, b_mid, b_high)
                    )
                    # Fader gains are intentionally explicit: B is at 70% by
                    # beat 8, reaches unity by beat 16, and A exits after beat 24.
                    mixed = self._safe_mix(a_processed, b_processed, a_fader, b_fader)
                    outdata[output_pos:output_pos + overlap] = mixed
                    self._current_pos += overlap
                    self._next_pos += overlap
                    output_pos += overlap
        except Exception as exc:  # Never allow an exception to kill the callback silently.
            self.callback_errors.append(repr(exc))
            outdata.fill(0)


class MVPDJ:
    def __init__(
        self,
        tracks: list[Track],
        *,
        cache_dir: PathLike = ".mvp_cache",
        transition_beats: int = DEFAULT_TRANSITION_BEATS,
        strict_genre: bool = True,
        repeat: bool = False,
        playlist_path: Optional[PathLike] = None,
        max_tracks: Optional[int] = None,
        manual_order: bool = False,
        mix_style: str = "club",
    ):
        if not tracks:
            raise MVPError("No tracks supplied")
        self.tracks = tracks
        self.cache = AudioCache(cache_dir)
        self.selector = TrackSelector(
            tracks,
            strict_genre=strict_genre,
            manual_order=manual_order,
        )
        self.transition_beats = max(8, int(transition_beats))
        self.mix_style = mix_style if mix_style in {"club", "smooth"} else "club"
        self.repeat = repeat
        self.playlist_path = Path(playlist_path).expanduser().resolve() if playlist_path else None
        self.max_tracks = max_tracks if max_tracks is None or max_tracks > 0 else None
        self._playlist_mtime = self.playlist_path.stat().st_mtime_ns if self.playlist_path and self.playlist_path.exists() else None
        self._playlist_error_mtime: Optional[int] = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _refresh_playlist(self) -> None:
        """Pick up tracks appended to a playlist while playback continues."""
        if self.playlist_path is None or not self.playlist_path.exists():
            return
        try:
            mtime = self.playlist_path.stat().st_mtime_ns
            if self._playlist_mtime == mtime:
                return
            incoming = load_playlist(self.playlist_path)
            added = self.selector.add_tracks(incoming)
            self._playlist_mtime = mtime
            self._playlist_error_mtime = None
            if added:
                print(f"📥 Added {added} track(s) from the live playlist queue.")
        except (OSError, MVPError, json.JSONDecodeError) as exc:
            # A file can be observed halfway through an editor's save. Keep the
            # old queue and retry after the file changes, rather than interrupting
            # audio or printing the same warning on every audio-loop tick.
            if self._playlist_error_mtime != mtime:
                print(f"⚠️ Live playlist refresh postponed: {exc}")
                self._playlist_error_mtime = mtime

    def _console_control_loop(self, renderer: MixRenderer) -> None:
        """Read simple terminal controls without touching the audio callback."""
        if not sys.stdin.isatty():
            return
        print("Controls: type p + Enter to pause/resume; q + Enter to stop.")
        while not self._stop.is_set():
            try:
                command = input().strip().lower()
            except (EOFError, OSError):
                return
            if command in {"p", "pause", "resume"}:
                paused = renderer.toggle_pause()
                print("⏸ Paused" if paused else "▶ Resumed")
            elif command in {"q", "quit", "exit", "stop"}:
                self.stop()
                renderer.stop()
                return

    def run(self) -> None:
        _require_audio_dependencies()
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDependencyError(
                "Live playback needs sounddevice. Install requirements-mvp.txt."
            ) from exc

        current = self.selector.choose(None)
        first_audio: Optional[np.ndarray] = None
        prepared_current: Optional[Track] = None
        while current is not None:
            self.selector.claim(current)
            try:
                print(f"▶ {current.title} [{current.genre}] {current.bpm or '?'} BPM")
                first_audio, prepared_current = self.cache.load_for_mix(current, target_bpm=None)
                break
            except Exception as exc:
                print(f"⚠️ Cannot load opener {current.title}: {exc}")
                self.selector.release(current)
                current = self.selector.choose(None)
        if first_audio is None or prepared_current is None:
            raise MVPError("Could not load any opening track")
        current = prepared_current
        target_bpm = current.bpm or DEFAULT_BPM
        transition_frames = int(self.transition_beats * 60.0 / target_bpm * SAMPLE_RATE)
        renderer = MixRenderer(
            first_audio,
            transition_frames=transition_frames,
            transition_beats=self.transition_beats,
            bpm=target_bpm,
            mix_style=self.mix_style,
        )
        tracks_played = 1

        installed_track: Optional[Track] = None
        candidate_track: Optional[Track] = None
        candidate_future: Optional[concurrent.futures.Future] = None
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        try:
            # Prepare the first transition before opening the stream. After that,
            # all network/decode work runs in the background while audio continues.
            if self.max_tracks is None or tracks_played < self.max_tracks:
                installed_track, installed_audio = self._prepare_next(current)
            else:
                installed_track, installed_audio = None, None
            if installed_track and installed_audio is not None:
                renderer.set_next(installed_audio)
                # `installed_track` is already the second planned track. Only
                # prepare another follow-up when the requested test limit allows
                # at least one more track beyond it.
                if self.max_tracks is None or tracks_played + 2 <= self.max_tracks:
                    candidate_track, candidate_future = self._schedule_followup(installed_track, executor)

            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=1024,
                callback=renderer,
            ):
                print("🎧 Safe MVP is playing. Ctrl+C stops playback.")
                threading.Thread(
                    target=self._console_control_loop,
                    args=(renderer,),
                    daemon=True,
                ).start()
                while not self._stop.is_set() and not renderer.is_ended():
                    self._refresh_playlist()
                    if renderer.callback_errors:
                        raise MVPError(f"Audio callback failed: {renderer.callback_errors[-1]}")

                    if renderer.consume_changed():
                        if installed_track is not None:
                            current = installed_track
                            tracks_played += 1
                            print(f"▶ {current.title} [{current.genre}] {current.bpm or '?'} BPM")
                        installed_track = None
                        if self.max_tracks is not None and tracks_played >= self.max_tracks:
                            renderer.set_hold_enabled(False)

                    if installed_track is None and candidate_future is not None and candidate_future.done():
                        try:
                            prepared_audio, prepared_track = candidate_future.result()
                            if renderer.set_next(prepared_audio):
                                installed_track = prepared_track
                                candidate_track = None
                                candidate_future = None
                                if self.max_tracks is None or tracks_played + 2 <= self.max_tracks:
                                    candidate_track, candidate_future = self._schedule_followup(
                                        installed_track, executor
                                    )
                        except Exception as exc:
                            if candidate_track is not None:
                                print(f"⚠️ Skipping {candidate_track.title}: {exc}")
                                self.selector.release(candidate_track)
                            candidate_track = None
                            candidate_future = None

                    # If a download fails or there are no same-genre tracks left,
                    # schedule the best remaining fallback. The selector never
                    # blocks the audio callback.
                    can_schedule_next = self.max_tracks is None or tracks_played + 1 <= self.max_tracks
                    if installed_track is None and candidate_future is None and can_schedule_next:
                        candidate_track, candidate_future = self._schedule_fallback(current, executor)
                        if candidate_track is None:
                            renderer.set_hold_enabled(False)
                    elif installed_track is None and candidate_future is None and not can_schedule_next:
                        renderer.set_hold_enabled(False)

                    time.sleep(0.05)
        finally:
            renderer.stop()
            executor.shutdown(wait=False, cancel_futures=True)

    def _prepare_next(self, current: Track) -> tuple[Optional[Track], Optional[np.ndarray]]:
        # Try candidates in selector order. One corrupt file or unavailable URL
        # must not prevent the set from starting when another track is usable.
        attempted = 0
        while attempted < len(self.tracks):
            chosen = self.selector.choose(current)
            if chosen is None and self.repeat:
                self.selector.reset()
                self.selector.claim(current)
                chosen = self.selector.choose(current)
            if chosen is None:
                print("🏁 No unused track remains; ending after the current track.")
                return None, None
            attempted += 1
            self.selector.claim(chosen)
            print(f"   queued: {chosen.title} [{chosen.genre}]")
            try:
                audio, prepared = self.cache.load_for_mix(chosen, target_bpm=current.bpm or DEFAULT_BPM)
                return prepared, audio
            except Exception as exc:
                print(f"⚠️ Skipping {chosen.title}: {exc}")
                self.selector.release(chosen)
        return None, None

    def _schedule_followup(
        self, installed: Track, executor: concurrent.futures.ThreadPoolExecutor
    ) -> tuple[Optional[Track], Optional[concurrent.futures.Future]]:
        chosen = self.selector.choose(installed)
        if chosen is None:
            return None, None
        self.selector.claim(chosen)
        future = executor.submit(self.cache.load_for_mix, chosen, target_bpm=installed.bpm or DEFAULT_BPM)
        return chosen, future

    def _schedule_fallback(
        self, current: Track, executor: concurrent.futures.ThreadPoolExecutor
    ) -> tuple[Optional[Track], Optional[concurrent.futures.Future]]:
        chosen = self.selector.choose(current)
        if chosen is None and self.repeat:
            self.selector.reset()
            self.selector.claim(current)
            chosen = self.selector.choose(current)
        if chosen is None:
            return None, None
        self.selector.claim(chosen)
        future = executor.submit(self.cache.load_for_mix, chosen, target_bpm=current.bpm or DEFAULT_BPM)
        return chosen, future


def dry_run(
    tracks: list[Track],
    *,
    strict_genre: bool = True,
    manual_order: bool = False,
    limit: Optional[int] = None,
) -> None:
    selector = TrackSelector(
        tracks,
        strict_genre=strict_genre,
        manual_order=manual_order,
    )
    current = selector.choose(None)
    count = 0
    while current is not None and (limit is None or count < limit):
        selector.claim(current)
        print(f"{count + 1:02d}. {current.title} | genre={current.genre} | bpm={current.bpm or '?'} | key={current.key or '?'}")
        current = selector.choose(current)
        count += 1
    print(f"\n{count} track(s) selected from {len(tracks)} available.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic Neuro-DJ safe MVP")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", help="Music folder; first subfolder is used as genre")
    source.add_argument("--playlist", help="JSON, M3U/M3U8, or XSPF playlist containing local paths or direct URLs")
    parser.add_argument("--cache-dir", default=".mvp_cache", help="Cache directory for online tracks")
    parser.add_argument("--transition-beats", type=int, default=DEFAULT_TRANSITION_BEATS)
    parser.add_argument(
        "--style",
        choices=("club", "smooth"),
        default="club",
        help="club=32-beat bass swap; smooth=gentler vocal/R&B blend",
    )
    parser.add_argument("--allow-cross-genre", action="store_true", help="Use another genre only if needed")
    parser.add_argument("--manual-order", action="store_true", help="Play playlist/folder order exactly; do not select by compatibility")
    parser.add_argument("--repeat", action="store_true", help="Reuse tracks when the crate is exhausted")
    parser.add_argument("--watch-playlist", action="store_true", help="Import tracks appended to --playlist while running")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned order without audio playback")
    parser.add_argument("--max-tracks", type=int, default=None, help="Stop live playback after this many tracks; also limits dry-run output")
    return parser


def load_tracks(args: argparse.Namespace) -> list[Track]:
    if args.folder:
        return discover_folder(args.folder)
    return load_playlist(args.playlist)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.watch_playlist and not args.playlist:
        print("ERROR: --watch-playlist requires --playlist", file=sys.stderr)
        return 2
    try:
        tracks = load_tracks(args)
        print(f"Loaded {len(tracks)} track(s).")
        if args.dry_run:
            dry_run(
                tracks,
                strict_genre=not args.allow_cross_genre,
                manual_order=args.manual_order,
                limit=args.max_tracks,
            )
            return 0

        dj = MVPDJ(
            tracks,
            cache_dir=args.cache_dir,
            transition_beats=args.transition_beats,
            strict_genre=not args.allow_cross_genre,
            repeat=args.repeat,
            playlist_path=args.playlist if args.watch_playlist else None,
            max_tracks=args.max_tracks,
            manual_order=args.manual_order,
            mix_style=args.style,
        )
        signal.signal(signal.SIGINT, lambda *_: dj.stop())
        dj.run()
        return 0
    except (MVPError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
