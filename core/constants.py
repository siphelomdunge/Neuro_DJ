"""
Neuro-DJ Constants & Configuration
All magic numbers, technique definitions, and thresholds in one place.
"""

# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL SAFETY MARGINS
# ═══════════════════════════════════════════════════════════════════════════
FINALIZE_SAFETY_BUFFER_SEC = 3.0    # Pre-execution planning buffer
EXECUTION_RESERVE_SEC = 3.0         # Runtime execution buffer
TOTAL_SAFETY_MARGIN = FINALIZE_SAFETY_BUFFER_SEC + EXECUTION_RESERVE_SEC
MINIMUM_EXIT_ROOM_SEC = 15.0        # Minimum space before EOF
PRE_TRIGGER_BUFFER_SEC = 12.0       # Wait buffer before trigger
EMERGENCY_HANDOFF_THRESHOLD = 5.0   # V48.4: Force handoff threshold

# ═══════════════════════════════════════════════════════════════════════════
# TRACK PLAYBACK
# ═══════════════════════════════════════════════════════════════════════════
MIN_EXIT_POSITION_RATIO = 0.70      # Don't exit before 70% through track
MAX_WAIT_SECONDS = 120.0            # Max decision wait time
DEFAULT_TRACK_DURATION = 300.0      # Fallback duration
MAX_PAUSE_EXTENSION = 1800.0        # 30 min pause cap

# ═══════════════════════════════════════════════════════════════════════════
# TIME STRETCH
# ═══════════════════════════════════════════════════════════════════════════
MIN_STRETCH_RATIO = 0.92
MAX_STRETCH_RATIO = 1.08
MIN_SAFE_BPM = 0.01                 # Prevent division by zero

# ═══════════════════════════════════════════════════════════════════════════
# TECHNIQUE FAMILIES
# ═══════════════════════════════════════════════════════════════════════════
IMMEDIATE_TECHNIQUES = {"ECHO_FREEZE", "ECHO_THROW"}
SMOOTH_TECHNIQUES = {"BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"}
SPECIAL_TECHNIQUES = {"PIANO_HANDOFF", "ACAPELLA_MASHUP"}

# ═══════════════════════════════════════════════════════════════════════════
# CAMELOT WHEEL
# ═══════════════════════════════════════════════════════════════════════════
CAMELOT_WHEEL = {
    'C': (8, 'B'), 'G': (9, 'B'), 'D': (10, 'B'), 'A': (11, 'B'),
    'E': (12, 'B'), 'B': (1, 'B'), 'F#': (2, 'B'), 'C#': (3, 'B'),
    'G#': (4, 'B'), 'D#': (5, 'B'), 'A#': (6, 'B'), 'F': (7, 'B'),
    'Am': (8, 'A'), 'Em': (9, 'A'), 'Bm': (10, 'A'), 'F#m': (11, 'A'),
    'C#m': (12, 'A'), 'G#m': (1, 'A'), 'D#m': (2, 'A'), 'A#m': (3, 'A'),
    'Fm': (4, 'A'), 'Cm': (5, 'A'), 'Gm': (6, 'A'), 'Dm': (7, 'A'),
}

# ═══════════════════════════════════════════════════════════════════════════
# TECHNIQUE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
TECHNIQUE_MIN_BEATS = {
    "BASS_SWAP": 64.0,
    "FILTER_SWEEP": 48.0,
    "ECHO_THROW": 32.0,
    "ECHO_FREEZE": 32.0,
    "SLOW_BURN": 64.0,
    "PIANO_HANDOFF": 64.0,
    "ACAPELLA_MASHUP": 48.0,
}

TECHNIQUE_SAFE_MIN_BEATS = {
    "BASS_SWAP": 64.0,
    "FILTER_SWEEP": 48.0,
    "ECHO_THROW": 16.0,
    "ECHO_FREEZE": 16.0,
    "SLOW_BURN": 64.0,
    "PIANO_HANDOFF": 64.0,
    "ACAPELLA_MASHUP": 32.0,
}

TECHNIQUE_IDEAL_BEATS = {
    "BASS_SWAP": 96.0,
    "FILTER_SWEEP": 80.0,
    "ECHO_THROW": 32.0,
    "ECHO_FREEZE": 32.0,
    "SLOW_BURN": 96.0,
    "PIANO_HANDOFF": 80.0,
    "ACAPELLA_MASHUP": 64.0,
}

TECHNIQUE_LIBRARY = {
    "BASS_SWAP": {
        "id": 0, "label": "🔊 Bass Swap",
        "defaults": {"beats": 96.0, "bass": 0.75, "echo": 0.0, "wash": 0.2, "piano_hold": 0.0}
    },
    "FILTER_SWEEP": {
        "id": 2, "label": "🌊 Filter Sweep",
        "defaults": {"beats": 96.0, "bass": 0.55, "echo": 0.0, "wash": 1.0, "piano_hold": 0.0}
    },
    "ECHO_THROW": {
        "id": 3, "label": "🎯 Echo Out",
        "defaults": {"beats": 32.0, "bass": 0.75, "echo": 1.0, "wash": 0.0, "piano_hold": 0.0}
    },
    "ECHO_FREEZE": {
        "id": 7, "label": "🧊 Echo Freeze",
        "defaults": {"beats": 32.0, "bass": 0.90, "echo": 1.0, "wash": 0.0, "piano_hold": 0.0}
    },
    "SLOW_BURN": {
        "id": 4, "label": "🕯️ Slow Burn",
        "defaults": {"beats": 128.0, "bass": 0.5, "echo": 0.0, "wash": 0.0, "piano_hold": 0.0}
    },
    "PIANO_HANDOFF": {
        "id": 5, "label": "🎹 Piano Handoff",
        "defaults": {"beats": 128.0, "bass": 0.50, "echo": 0.0, "wash": 0.0, "piano_hold": 1.0}
    },
    "ACAPELLA_MASHUP": {
        "id": 8, "label": "🎤 Acapella Mashup",
        "defaults": {"beats": 64.0, "bass": 0.85, "echo": 0.0, "wash": 0.0, "piano_hold": 0.0, "acapella": 1.0}
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# DANCE MOMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════
DANCE_MOMENT_SCORES = {
    'hard_reset': {
        'ECHO_FREEZE': 70, 'ECHO_THROW': 45, 'BASS_SWAP': 20, 'SLOW_BURN': -40
    },
    'reboot': {
        'ECHO_FREEZE': 60, 'ECHO_THROW': 40, 'BASS_SWAP': 30, 'SLOW_BURN': -20
    },
    'controlled_rebuild': {
        'BASS_SWAP': 50, 'PIANO_HANDOFF': 45, 'SLOW_BURN': 35, 'ECHO_FREEZE': -15
    },
    'build_release': {
        'ECHO_FREEZE': 50, 'BASS_SWAP': 40, 'PIANO_HANDOFF': 30, 'SLOW_BURN': -30
    },
    'peak_swap': {
        'BASS_SWAP': 50, 'PIANO_HANDOFF': 30, 'ACAPELLA_MASHUP': 25,
        'ECHO_FREEZE': -10, 'SLOW_BURN': -20
    },
    'harmonic_lift': {
        'PIANO_HANDOFF': 55, 'FILTER_SWEEP': 40, 'BASS_SWAP': 30, 'ECHO_FREEZE': -20
    },
    'melodic_reset': {
        'PIANO_HANDOFF': 50, 'SLOW_BURN': 45, 'FILTER_SWEEP': 35,
        'ECHO_FREEZE': -35, 'ECHO_THROW': -35
    },
    'groove_extension': {
        'BASS_SWAP': 55, 'PIANO_HANDOFF': 40, 'FILTER_SWEEP': 30, 'ACAPELLA_MASHUP': 35
    },
    'vocal_relief': {
        'ECHO_FREEZE': 55, 'ECHO_THROW': 55, 'ACAPELLA_MASHUP': 65,
        'BASS_SWAP': -35, 'SLOW_BURN': -25
    },
    'pre_peak_build': {
        'ECHO_FREEZE': 45, 'BASS_SWAP': 25, 'SLOW_BURN': -20
    },
    'natural_exit': {
        'SLOW_BURN': 50, 'FILTER_SWEEP': 40, 'BASS_SWAP': 30, 'ECHO_FREEZE': -20
    },
    'breather': {
        'SLOW_BURN': 55, 'FILTER_SWEEP': 45, 'ECHO_FREEZE': -40, 'ECHO_THROW': -40
    },
    'cool_down': {
        'SLOW_BURN': 50, 'FILTER_SWEEP': 40, 'ECHO_FREEZE': -30, 'ECHO_THROW': -30
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# VALID ENERGIES
# ═══════════════════════════════════════════════════════════════════════════
VALID_ENERGIES = {"High", "Low/Chill", "Medium"}

def clamp01(v: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))