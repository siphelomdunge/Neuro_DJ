🧠 Neuro-DJ: Dynamic Technique Intelligence

> **Party-safe MVP:** Use [`mvp_dj.py`](MVP.md) for a deterministic, conservative runner. It uses explicit genres, optional BPM/key metadata, background caching, and one fixed equal-power crossfade. The original `neuro_gui.py` engine remains experimental and is not the recommended path for an unattended party.

Neuro-DJ is an autonomous, hybrid AI DJ engine designed to replicate mainstage, producer-level mixing logic. It goes beyond simple beatmatching and crossfading by actively analyzing the acoustic context of tracks (density, vocals, tension, genre) to dynamically select the most mathematically and musically appropriate transition technique in real-time.

Built with a high-performance C++ audio core and a Python intelligence layer, Neuro-DJ features live telemetry, zero-latency stem extraction, dynamic stall detection, and a reinforcement learning (RLHF) loop to adapt to human feedback.

## 🤝 Development Methodology & AI Collaboration

**Role: Systems Architect & Principal QA Lead**

Neuro-DJ was built using a heavily iterative, AI-assisted development workflow. My role in this project was designing the acoustic scoring logic, defining the state-machine architecture, and guiding an LLM to generate the raw Python and C++ syntax.

Rather than just writing raw code, my focus was on high-level system orchestration and rigorous mathematical debugging. Key engineering problems I solved during this build include:

  * **The "Early Commitment" Panic:** Fixed a mathematical scoring cliff where the AI would aggressively penalize late transitions, forcing the engine into a dead-lock. I designed a continuous slope penalty that gives the AI the "patience" to hold for the perfect outro phrase.
  * **The "Endless Hold" Paradox:** Decoupled the 80-beat overlap scoring window from the 12-second exit-room filter, preventing the AI from systematically deleting valid outro phrases and freezing the set.
  * **Zero-Latency Vocal Extraction:** Instead of relying on computationally heavy, high-latency deep learning models (like Demucs/Spleeter) for live mashups, I architected a **Mid/Side (M/S) DSP filter** in C++. By high-passing the Mid channel at 250Hz (Butterworth) and attenuating the Side channel, the engine extracts center-panned vocals with 0.008% CPU cost and zero latency.

-----

## 🧪 Experimental engine features (legacy)

The following section describes the original research engine. These features are intentionally not used by the safe MVP because they have a larger dependency/build surface and need further real-audio testing.

  * **Acoustic Context Awareness:** Analyzes track structures, vocal regions, harmonic density, and tension scores to determine the "Dance Moment" (e.g., *controlled\_rebuild*, *vocal\_relief*, *peak\_swap*).
  * **Dynamic Technique Selection:** Chooses from a library of professional techniques based on acoustic context and runway:
      * 🔊 Bass Swap
      * 🌊 Filter Sweep
      * 🎯 Echo Throw / 🧊 Echo Freeze
      * 🕯️ Slow Burn
      * 🎹 Piano Handoff (Amapiano specifically)
      * 🎤 Acapella Mashup
  * **The Acapella Mashup Engine:** Automatically loads offline HPSS (Harmonic/Percussive) stems for studio-quality vocal isolation, or seamlessly falls back to the real-time C++ Mid/Side extractor for live, on-the-fly drops.
  * **Adaptive Executor (Live Ears):** Monitors the live master bus at 10Hz. If it detects "sonic mud" (bass fights or vocal clashes) or phase drift during a transition, it autonomously intervenes with temporary EQ cuts to save the mix.
  * **Reinforcement Learning (RLHF):** Learns from 1–10 human ratings and failure tags (e.g., "bass fight," "too abrupt") to mathematically tune its technique recipes (bass crossover timing, echo intensity, transition duration) for future sets.
  * **High-Res Telemetry:** Records complete 10Hz flight-data logs of every transition (EQ curves, bass congestion, phase drift) for future deep learning model training.
  * **Bulletproof Fail-safes:** Built with strict mathematical guards against early commitment panics, endless holds, and EOF audio stalling using dynamic buffer monitoring.

-----

## 🏗️ System Architecture

Neuro-DJ operates on a dual-layer architecture:

1.  **The Intelligence Layer (Python):** Handles crate management, phrase analysis, context building, technique scoring, adaptive monitoring, and the GUI.
2.  **The Audio Core (C++):** A custom, high-performance DSP engine built with PortAudio. It handles mathematically continuous constant-power crossfading, multi-stage drop timing, Reverb/Delay effects, and M/S acapella extraction.

### Core Files

  * `neuro_gui.py`: The Master Engine. Handles the decision loop, scheduling, and the PyQt5 interface.
  * `core_audio.cpp`: The C++ DSP engine. Compiled via Pybind11 into the `neuro_core` module.
  * `auto_prep_folder.py`: The offline analysis pipeline. Extracts BPM, keys, structure, phrases, and generates HPSS harmonic stems.
  * `adaptive_executor.py`: The live intervention watchdog.
  * `live_ears.py`: The acoustic feature extractor for the live master bus.
  * `telemetry_logger.py`: The 10Hz flight-data recorder.
  * `crate_ranker.py`: The contextual setlist selector.

-----

## ✅ Safe MVP: recommended playback path

The repository now includes a deliberately small playback path in `mvp_dj.py`. It is designed for reliability rather than experimental technique selection:

- no phrase/HPSS analysis in the live path;
- automatic same-genre/BPM/key/energy scoring with artist variety, or exact manual playlist order;
- 44.1 kHz stereo conversion and conservative loudness normalization;
- one deterministic equal-power crossfade;
- background preparation of the next local or explicitly supplied direct URL track;
- cache and end-hold fallback when an online track is late;
- no network or disk work in the audio callback.

Install the MVP dependencies:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-mvp.txt
```

If `python` is from Miniconda/Anaconda and `venv` hangs in `ensurepip`, use a clean environment instead:

```bash
conda create -n neuro-dj python=3.11 -y
conda activate neuro-dj
python -m pip install --upgrade pip
python -m pip install -r requirements-mvp.txt
```

On Pop!_OS/Debian, install compressed-audio and device libraries once:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 portaudio19-dev
```

For compressed audio, install `ffmpeg` separately and put it on `PATH`.

Preview the order without playing audio:

```bash
python mvp_dj.py --folder ./music --dry-run
# or
python mvp_dj.py --playlist mvp_playlist.example.json --dry-run
```

Run the safe MVP:

```bash
python mvp_dj.py --folder ./music --transition-beats 32
# or a standard playlist exported by VLC/Rhythmbox/etc.
python mvp_dj.py --playlist party.m3u --manual-order --transition-beats 32
# VLC XSPF playlists are supported too
python mvp_dj.py --playlist ~/Music/shubile.xspf --watch-playlist --manual-order --transition-beats 32
# or a JSON queue with live updates
python mvp_dj.py --playlist party.json --cache-dir .mvp_cache --transition-beats 32 --watch-playlist
```

M3U/M3U8 playlists preserve the order you create in your music player, so you do not need to write JSON. With `--watch-playlist`, append tracks to the JSON or M3U queue while the set runs. They are validated and cached only when they become candidates. During playback, type `p` + Enter to pause/resume or `q` + Enter to stop. See [`MVP.md`](MVP.md) for the folder layout, playlist formats, online-cache behavior, and party checklist. URLs must be direct, legally playable audio URLs supplied by a provider; this project does not bypass provider authentication or download protections.

## 🛠️ Experimental engine installation

### 1\. Prerequisites

You will need a C++ compiler to build the DSP engine, and a Python environment for the intelligence layer.

  * **Python:** 3.9+
  * **C++ Compiler:** MSVC (Windows), Clang/GCC (macOS/Linux)
  * **PortAudio:** Required for audio playback.

### 2\. Environment Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/neuro-dj.git
cd neuro-dj
```

Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install the required Python dependencies:

```bash
pip install numpy scipy librosa soundfile pyqt5 pyqtgraph pybind11
```

### 3\. Compiling the C++ Audio Core

Neuro-DJ relies on `neuro_core`, a custom C++ module. You must compile this before running the application.

**Dependencies needed for compilation:**

  * PortAudio headers and binaries
  * `dr_wav.h` (Included in the repo, or available from [mackron/dr\_libs](https://github.com/mackron/dr_libs))

You can compile using `setuptools` (requires a `setup.py` configured for your OS):

```bash
python setup.py build_ext --inplace
```

*Note: Ensure your `setup.py` points to your local PortAudio include and library paths.*

-----

## 🎧 Usage

### 1\. Prep Your Crate (Offline)

Neuro-DJ requires tracks to be pre-analyzed. The offline prep script extracts metadata, structural phases, and high-quality HPSS harmonic stems for the Acapella Mashup engine.

```bash
python auto_prep_folder.py /path/to/your/music/folder
```

This will generate a `master_library.json` file in the directory.

### 2\. Launch the Engine

Start the Master GUI and pass your prepared library file. You can optionally specify a target set duration in seconds (default is 3600s / 1 hour).

```bash
python neuro_gui.py /path/to/your/music/folder/master_library.json 3600
```

### 3\. The RLHF Interface

As the AI completes transitions, a rating panel will appear on the GUI.

  * Rate the mix from 1–10.
  * Select any failure tags if the mix was flawed (e.g., "Vocals clashed", "Bass was muddy").
  * Click **TRAIN MODEL**. The AI will instantly adjust its memory weights and technique recipes for future decisions.

-----

## 📊 Telemetry Data

For every completed mix, Neuro-DJ generates a JSON file in `logs/transitions/`. These files contain:

  * Pre-mix acoustic context (density, tension, trajectory).
  * The chosen technique and the exact recipe parameters used.
  * A 10Hz timeline of the exact EQ curves, bass congestion, and phase drift during the overlap.
  * A log of any emergency interventions taken by the Adaptive Executor.
  * The human evaluation (Rating 1-10 + tags).

This dataset is structurally ready to train a deep neural network to predict optimal transition recipes entirely autonomously.

-----

## 📜 License

[MIT License](https://www.google.com/search?q=LICENSE)
