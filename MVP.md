# Neuro-DJ Safe MVP

The repository contains an experimental, feature-heavy engine in `neuro_gui.py` and a smaller deterministic runner in `mvp_dj.py`.

For a live party, use the MVP runner. It intentionally does **not** use the experimental phrase analyser, live ears, adaptive executor, echo effects, or RL-style recipe updates.

## Why this path is safer

The audio callback only mixes already-prepared PCM arrays. It never performs network requests, disk I/O, metadata analysis, or candidate selection. The next track is prepared by a background worker and installed before it is needed.

The MVP uses:

- explicit genre metadata from the playlist or the first music-folder subdirectory;
- optional BPM, Camelot key, energy, and cue-in metadata;
- automatic mode: same-genre selection first, then a clearly marked fallback when the genre is exhausted;
- an artist-variety guard to avoid the same artist back-to-back when an equally safe alternative exists;
- manual mode: play the playlist order exactly and use Neuro-DJ only for the transitions;
- a deterministic club/amapiano 32-beat crossfade with compiled three-band EQ and an explicit low-end swap, or a gentler smooth profile for vocal/R&B material;
- 44.1 kHz stereo output;
- optional recording of the exact rendered output to WAV, FLAC, OGG, or AIFF on a background writer thread;
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

A playlist can be a standard `.m3u`/`.m3u8` file, an `.xspf` file exported by VLC, or JSON. M3U/XSPF is the easiest option when you want to build a playlist in VLC, Rhythmbox, Strawberry, a file manager, or another music application. It preserves the order from the playlist:

```text
#EXTM3U
#EXTINF:-1,DJ One - First Track
/home/siphelo/Music/house/first.mp3
#EXTINF:-1,DJ Two - Second Track
/home/siphelo/Music/house/second.mp3
```

Run an M3U playlist in exact order:

```bash
python mvp_dj.py --playlist party.m3u --manual-order --transition-beats 32
```

Record the rendered set while it plays:

```bash
python mvp_dj.py \
  --playlist party.m3u \
  --manual-order \
  --transition-beats 32 \
  --record recordings/party-set.wav
```

`--record` captures the actual post-EQ, post-crossfade stereo output sent to the
sound device. Recording is written by a background worker so the audio callback
never performs disk I/O. If you pause playback, the recording contains silence
for the pause; the playback position and transition state remain frozen.
Supported output extensions are `.wav`, `.flac`, `.ogg`, `.aif`, and `.aiff`.
Use a new path for each set because an existing file is replaced.

JSON is still supported for detailed BPM/key/cue metadata:

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

Automatic mode prefers, in order:

1. tracks that have not been used;
2. the same explicit genre;
3. BPM within a small difference window;
4. compatible Camelot keys when supplied;
5. a different artist when an equally safe option exists;
6. a small energy change rather than a sudden jump.

If you want to decide the exact running order yourself, use `--manual-order`. In that mode the JSON list order is absolute; genre, BPM, key, energy and artist scoring do not reorder anything. The audio engine still performs the safe transition and BPM adjustment.

```bash
python mvp_dj.py \
  --playlist party.json \
  --manual-order \
  --watch-playlist \
  --transition-beats 32
```

When using `--watch-playlist`, append new tracks to the end of the JSON `tracks` list. They will be played in the order they are added.

Unknown genre is not treated as compatible with every genre in automatic mode. If no same-genre track remains, the next least-bad candidate is used only as a fallback.

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

For club/amapiano material, a 32-beat transition is a conservative starting point. At 120 BPM it is approximately 16 seconds. The MVP's club transition profile is:

- beats 1–8: B fader rises to 70%, B low is fully cut, and B highs stay restrained;
- beats 9–16: B reaches unity, B mids/highs open, and A highs soften;
- beats 17–24: B low rises gently while A low is reduced;
- beats 25–32: A fades, then B low completes the swap on beat 32.

For R&B, soul, pop or vocal-heavy playlists, use the gentler smooth profile:

```bash
python mvp_dj.py --playlist rnb.xspf --manual-order --style smooth --transition-beats 16
```

The smooth profile uses a shorter blend, avoids the dramatic bass slam, and keeps the incoming low end mostly present instead of applying the amapiano log-drum swap. It is safer for sparse arrangements and exposed vocals.

The MVP does not create a separate headphone cue mix. It assumes the supplied `cue_in` or first sample is the intended phrase entry. Use an explicit `cue_in` in the playlist when a file has leading silence.

While playing in a terminal:

- type `p` and press Enter to pause/resume;
- type `q` and press Enter to stop;
- `Ctrl+C` also stops playback.

Pause freezes the audio position and transition state; it does not reset the current mix. Use 64 beats only after listening to the result on the actual party sound system.

## Party checklist

- Run a full dry-run and inspect the genre/BPM metadata.
- Test the exact speakers, interface, and laptop that will be used.
- Run at least 60 minutes continuously before the party.
- Keep a local emergency playlist available.
- Keep the laptop on power and disable sleep.
- Do not introduce a new provider or effect on the day of the party.

## Experimental engine

The original `neuro_gui.py` remains in the repository for future work. It requires a compiled `neuro_core` extension and the larger analysis dependency set. It is not the recommended party path until the audio core, effects, analysis, and online-source boundaries have been tested independently.
