"""
set_state.py — Neuro-DJ Vibe & Fatigue Tracker (Workstream 10)
══════════════════════════════════════════════════════════════
A DJ set is a journey, not just a series of good transitions. 
If you play 10 high-energy vocal tracks in a row, the crowd gets exhausted.

The SetStateModel keeps a rolling memory of the "Vibe". 
It tracks:
  - Time since the last breather (Low/Chill track)
  - Vocal fatigue (too many singers back-to-back)
  - Energy momentum (are we climbing, peaking, or cruising?)
  - Novelty vs. Hypnotic pocket (changing genres vs. holding a groove)
  
It provides a `snapshot` to the DecisionCore, and a method to dynamically 
penalize exhausting techniques (like BASS_SWAP) if the crowd needs air, 
or boost them if the set is getting boring.
"""

import time
from dataclasses import dataclass

@dataclass
class SetSnapshot:
    set_phase: str
    energy_rolling: float
    vocal_density_fatigue: float
    time_since_breather: float
    time_since_hard_drop: float
    novelty_rate: float
    mix_count: int
    recommend_breather: bool
    recommend_intensity: bool

class SetStateModel:
    def __init__(self):
        self.start_time = time.time()
        self.mix_count = 0
        
        # Rolling stats
        self.energy_rolling = 0.5  # 0.0 to 1.0
        self.vocal_history = []    # List of booleans (last 5 tracks)
        self.genre_history = []    # List of strings (last 5 tracks)
        
        # Timestamps
        self.last_breather_time = self.start_time
        self.last_hard_drop_time = self.start_time
        
        self.events = []  # Log of transition events

    def get_snapshot(self) -> SetSnapshot:
        now = time.time()
        
        time_since_breather = now - self.last_breather_time
        time_since_hard_drop = now - self.last_hard_drop_time
        
        vocal_fatigue = sum(self.vocal_history) / max(len(self.vocal_history), 1)
        
        # Novelty: 1.0 means every track was a different genre, 0.0 means all same
        if len(self.genre_history) > 1:
            changes = sum(1 for i in range(1, len(self.genre_history)) if self.genre_history[i] != self.genre_history[i-1])
            novelty = changes / (len(self.genre_history) - 1)
        else:
            novelty = 0.5

        # Determine Set Phase
        elapsed_min = (now - self.start_time) / 60.0
        if elapsed_min < 15: phase = "Warmup"
        elif elapsed_min < 45: phase = "Building"
        elif elapsed_min < 90: phase = "Peak"
        else: phase = "Closing"

        # Triggers for intervention
        req_breather = (time_since_breather > 180 and self.energy_rolling > 0.75) or vocal_fatigue > 0.8
        req_intensity = (time_since_hard_drop > 300 and self.energy_rolling < 0.5 and self.mix_count > 3)

        return SetSnapshot(
            set_phase=phase,
            energy_rolling=self.energy_rolling,
            vocal_density_fatigue=vocal_fatigue,
            time_since_breather=time_since_breather,
            time_since_hard_drop=time_since_hard_drop,
            novelty_rate=novelty,
            mix_count=self.mix_count,
            recommend_breather=req_breather,
            recommend_intensity=req_intensity
        )

    def update_transition(self, ta: dict, tb: dict, tech_name: str):
        now = time.time()
        self.mix_count += 1
        
        # Update Energy
        eb = tb.get('energy', 'High')
        energy_val = 0.85 if eb == "High" else 0.25
        self.energy_rolling = (self.energy_rolling * 0.7) + (energy_val * 0.3)
        
        # Update Breather / Drops
        if eb == "Low/Chill" or tech_name == "SLOW_BURN":
            self.last_breather_time = now
            
        if eb == "High" and tech_name in ("BASS_SWAP", "ECHO_FREEZE", "PIANO_HANDOFF"):
            self.last_hard_drop_time = now
            
        # Update Vocals
        b_has_vocal = tb.get('stems', {}).get('has_vocals', False)
        self.vocal_history.append(b_has_vocal)
        if len(self.vocal_history) > 5:
            self.vocal_history.pop(0)
            
        # Update Genre Novelty
        self.genre_history.append(tb.get('genre', 'open'))
        if len(self.genre_history) > 5:
            self.genre_history.pop(0)

    def apply_to_scoring(self, scores: dict, snapshot: SetSnapshot) -> dict:
        """
        Mutates the raw technique scores from select_technique based on macro set fatigue.
        Called directly by the DecisionCore.
        """
        modified = dict(scores)
        
        # If the crowd desperately needs a breather, penalize aggressive techniques
        if snapshot.recommend_breather:
            modified["ECHO_FREEZE"] -= 30
            modified["ECHO_THROW"] -= 20
            modified["BASS_SWAP"] -= 15
            modified["SLOW_BURN"] += 40
            modified["FILTER_SWEEP"] += 25
            
        # If the set is dragging and needs energy
        if snapshot.recommend_intensity:
            modified["ECHO_FREEZE"] += 35
            modified["BASS_SWAP"] += 25
            modified["PIANO_HANDOFF"] += 20
            modified["SLOW_BURN"] -= 40
            
        # If vocal fatigue is high, heavily penalize techniques that layer audio
        if snapshot.vocal_density_fatigue >= 0.8:
            modified["SLOW_BURN"] -= 30
            modified["BASS_SWAP"] -= 20
            modified["ECHO_FREEZE"] += 25  # Freeze clears the outgoing vocal instantly, avoiding mud
            
        return modified

    def summary(self) -> str:
        snap = self.get_snapshot()
        lines = [
            f"📈 Set State [Phase: {snap.set_phase} | Mixes: {snap.mix_count}]",
            f"   ├─ Energy: {snap.energy_rolling*100:.0f}%",
            f"   ├─ Vocal Fatigue: {snap.vocal_density_fatigue*100:.0f}%",
            f"   ├─ Time since breather: {snap.time_since_breather/60:.1f}m",
            f"   └─ Time since hard drop: {snap.time_since_hard_drop/60:.1f}m"
        ]
        if snap.recommend_breather:
            lines.append("   ⚠️  CROWD FATIGUE DETECTED: Recommending a breather.")
        if snap.recommend_intensity:
            lines.append("   💥 ENERGY LULL DETECTED: Recommending a hard drop.")
            
        return "\n".join(lines)