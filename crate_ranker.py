"""
crate_ranker.py — Neuro-DJ Pre-Selection Sieve (Workstream 1)
═════════════════════════════════════════════════════════════
The DecisionCore (Phase B searcher) is highly computationally expensive,
running full cross-correlations on 8-beat phrase windows. We cannot 
run that on 1,000 tracks every time a mix is needed.

The CrateRanker is the lightweight heuristic "Sieve". 
It takes the globally viable crate (tracks that physically CAN mix) 
and scores them on pure musical logic:
  1. Harmonic Proximity (Camelot wheel distance)
  2. Tempo Friction (How much stretching is required?)
  3. Energy Arc (Does this track fit the current trajectory?)
  4. Genre Cohesion (Amapiano prefers Amapiano)

It returns the top `N` (default 8) candidates to the DecisionCore 
for deep, cycle-heavy analysis.
"""

from typing import List, Dict, Any
import os

class CrateRanker:
    def __init__(self, brain: Any):
        """
        Takes a reference to the DJBrain to reuse its foundational 
        Camelot and BPM scoring logic without duplicating code.
        """
        self.brain = brain
        self.played_files = set()

    def record_selection(self, track: dict):
        """Records a track as played so we don't pick it again."""
        filename = track.get('filename')
        if filename:
            self.played_files.add(filename)

    def select_candidates(
        self, 
        current_track: dict, 
        crate: List[dict], 
        master_bpm: float, 
        n: int = 8, 
        energy_history: List[str] = None
    ) -> List[dict]:
        """
        Ranks the entire viable crate and returns the top `n` tracks.
        """
        if not crate:
            return []

        scored_candidates = []
        is_amp_current = (current_track.get('genre') == 'amapiano')
        curr_key = current_track.get('key', 'C')
        curr_energy = current_track.get('energy', 'High')

        for track in crate:
            # Skip tracks we've already played
            if track.get('filename') in self.played_files:
                continue

            score = self._heuristic_score(
                current_track=current_track,
                candidate=track,
                master_bpm=master_bpm,
                is_amp_current=is_amp_current,
                curr_key=curr_key,
                curr_energy=curr_energy,
                energy_history=energy_history or []
            )
            
            scored_candidates.append((score, track))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Extract just the track dictionaries for the top N
        top_n = [track for score, track in scored_candidates[:n]]
        
        # Diagnostic logging
        print(f"   🌪️  CrateRanker sifted {len(crate)} viable tracks down to Top {len(top_n)}:")
        for i, (score, track) in enumerate(scored_candidates[:min(n, 3)]):
            name = os.path.basename(track.get('filename', '?'))[:30]
            print(f"       {i+1}. [{score:>3.0f} pts] {name} ({track.get('bpm', 0):.1f} | {track.get('key', '?')} | {track.get('energy', '?')})")

        return top_n

    def _heuristic_score(
        self, 
        current_track: dict, 
        candidate: dict, 
        master_bpm: float,
        is_amp_current: bool,
        curr_key: str,
        curr_energy: str,
        energy_history: List[str]
    ) -> float:
        """
        Calculates a fast, lightweight compatibility score.
        Max score is theoretically around 100-110.
        """
        score = 0.0
        
        cand_bpm = candidate.get('bpm', 112.0)
        cand_key = candidate.get('key', 'C')
        cand_energy = candidate.get('energy', 'High')
        cand_genre = candidate.get('genre', 'open')

        # ── 1. Tempo Friction (Max ~35 pts) ──
        # Re-use brain's logic, but penalize distance from the *master* BPM, 
        # not just the current track's original BPM.
        is_amp_cand = (cand_genre == 'amapiano')
        score += self.brain.bpm_score(master_bpm, cand_bpm, is_amp_current or is_amp_cand)

        # ── 2. Harmonic Proximity (Max 40 pts) ──
        key_score, key_label = self.brain.camelot_score(curr_key, cand_key)
        score += key_score

        # ── 3. Genre Cohesion (Max 30 pts) ──
        score += self.brain.genre_score(current_track, candidate)

        # ── 4. Energy Flow Logic (Max ~20 pts) ──
        # This is a lighter version of the IntentEngine. 
        # We just want to prevent jarring, illogical energy jumps unless needed.
        if curr_energy == cand_energy:
            score += 10.0  # Safe bet to maintain the vibe
            
            # Penalty for flatlining (3 tracks of the same energy in a row)
            if len(energy_history) >= 2 and all(e == curr_energy for e in energy_history[-2:]):
                score -= 15.0 
        else:
            # If energies are different, it's a vibe shift. 
            # We reward a shift if the crowd has been in the same state for a while.
            if len(energy_history) >= 2 and all(e == curr_energy for e in energy_history[-2:]):
                score += 20.0  # Reward breaking the monotony
            elif curr_energy == "High" and cand_energy == "Low/Chill":
                score += 5.0   # Breathers are generally acceptable
            else:
                score -= 5.0   # Random jumping around is slightly penalized

        # ── 5. Vocal Clash Heuristic (Max 10 pts) ──
        # Just checking if both tracks generally have vocals. The DecisionCore 
        # will check exact phrase overlapping later, but this helps push 
        # instrumentals higher up the candidate list if the current track has vocals.
        score += self.brain.vocal_score(current_track, candidate)

        # ── 6. Spectral Match (Max 10 pts) ──
        score += self.brain.spectral_score(current_track, candidate)

        return score