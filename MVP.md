# Neuro-DJ Safe MVP

The repository contains an experimental, feature-heavy engine in `neuro_gui.py` and a smaller deterministic runner in `mvp_dj.py`.

For a live party, use the MVP runner. It intentionally does **not** use the experimental phrase analyser, live ears, adaptive executor, echo effects, or RL-style recipe updates.

## Why this path is safer

The audio callback only mixes already-prepared PCM arrays. It never performs network requests, disk I/O, metadata analysis, or candidate selection. The next track is prepared by a background worker and installed before it is needed.

The MVP uses:

- explicit genre metadata from the playlist or the first music-folder subdirectory;
- optional BPM, Camelot key, energy, and cue-in metadata;
- same-genre selection first, then a clearly marked fallback when the genre is exhausted;
- a single deterministic equal-power crossfade;
- 44.1 kHz stereo output;
- an on-disk cache for explicitly supplied direct URLs;
- a local/cache fallback instead of allowing a network failure to enter the audio callback;
- an eight-second end hold if a background download is late, rather than immediately emitting silence.

It does **not** search or bypass music providers. A playlist URL must be a direct, legally playable audio URL supplied by a provider that permits this use. Provider authentication, attribution, and public-performance rights remain the operator's responsibility.

## Install

Install the small runtime:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-mvp.txt
```

If `python` is coming from Miniconda/Anaconda and `python -m venv .venv` hangs inside `ensurepip`, use a clean Conda environment instead:

```bash
conda create -n neuro-dj python=3.11 -y
conda activate neuro-dj
python -m pip install --upgrade pip
python -m pip install -r requirements-mvp.txt
```

Python 3.11 is recommended for the party setup because it has the broadest audio-package compatibility. On Pop!_OS/Debian, install the system audio libraries once:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 portaudio19-dev
```

For MP3, AAC, M4A, and other compressed formats, install `ffmpeg` and make sure it is on `PATH`. WAV files can be decoded directly by `soundfile`.

## Prepare a reliable crate

The simplest safe layout is:

```text
music/
├── afro-house/
│   ├── 01-opener.wav
│   └── 02-groove.wav
├── amapiano/
│   └── 01-log-drum.wav
└── house/
    └── 01-house.wav
```

The first subdirectory is used as the genre. BPM is not guessed by the MVP. Supply it in a playlist JSON, use a previously generated `master_library.json`, or include `bpm` in the filename, for example `song_bpm_123.wav`.

A playlist can contain local paths and direct URLs:

```json
{
  "tracks": [
    {
      "path": "music/afro-house/01-opener.wav",
      "title": "Opener",
      "genre": "afro-house",
      "bpm": 122,
      "key": "8A",
      "energy": 0.55
    },
    {
      "url": "https://provider.example/authorized-track.mp3",
      "title": "Online Track",
      "genre": "afro-house",
      "bpm": 123,
      "key": "9A",
      "energy": 0.65
    }
  ]
}
```

See `mvp_playlist.example.json` for all supported fields.

## Dry-run the selection order

Always do this first:

```bash
python mvp_dj.py --playlist party.json --dry-run
# or
python mvp_dj.py --folder ./music --dry-run
```

The selector prefers, in order:

1. tracks that have not been used;
2. the same explicit genre;
3. BPM within a small difference window;
4. compatible Camelot keys when supplied;
5. a small energy change rather than a sudden jump.

Unknown genre is not treated as compatible with every genre. If no same-genre track remains, the next least-bad candidate is used only as a fallback.

## Run the MVP

```bash
python mvp_dj.py \
  --playlist party.json \
  --cache-dir .mvp_cache \
  --transition-beats 32
```

Or:

```bash
python mvp_dj.py --folder ./music --transition-beats 32
```

The first online track is prepared before playback starts. Later tracks are downloaded and decoded in a background worker. The audio callback does not wait for the network.

To add tracks while the set is running, edit the playlist JSON and append a new item, then start with:

```bash
python mvp_dj.py --playlist party.json --watch-playlist --transition-beats 32
```

The file is polled safely; a partially-written JSON file is retried without interrupting audio. Newly added tracks are only cached when they become the next candidate.

A 32-beat transition is a conservative starting point. At 120 BPM it is approximately 16 seconds. Use 64 beats only after listening to the result on the actual party sound system.

## Party checklist

- Run a full dry-run and inspect the genre/BPM metadata.
- Test the exact speakers, interface, and laptop that will be used.
- Run at least 60 minutes continuously before the party.
- Keep a local emergency playlist available.
- Keep the laptop on power and disable sleep.
- Do not introduce a new provider or effect on the day of the party.

## Experimental engine

The original `neuro_gui.py` remains in the repository for future work. It requires a compiled `neuro_core` extension and the larger analysis dependency set. It is not the recommended party path until the audio core, effects, analysis, and online-source boundaries have been tested independently.
