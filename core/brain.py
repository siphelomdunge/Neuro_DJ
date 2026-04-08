"""
DJ Brain — Harmonic compatibility, ML learning, and track scoring.
"""
from __future__ import annotations
import os
import json
import random
import threading
from typing import Dict, Tuple

from .constants import (
    CAMELOT_WHEEL, TECHNIQUE_LIBRARY, TECHNIQUE_MIN_BEATS, clamp01
)


class PersistentLearner:
    """Machine learning system that adapts technique parameters based on feedback."""
    
    CONFIDENCE_FILE = "neuro_confidence.json"
    MEMORY_FILE = "neuro_brain_memory.json"
    
    def __init__(self):
        self.memory: Dict[str, Dict] = {}
        self.confidence: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.load_brain()
        self._load_confidence()
    
    def _default_memory(self) -> Dict[str, Dict]:
        """Generate default recipe memory for all techniques."""
        mem = {}
        for tech_name, tech in TECHNIQUE_LIBRARY.items():
            for pair in ("High->High", "High->Low/Chill", "Low/Chill->High",
                        "Low/Chill->Low/Chill", "Amapiano->Amapiano"):
                key = f"{tech_name}|{pair}"
                mem[key] = dict(tech["defaults"])
        return mem
    
    def load_brain(self):
        """Load ML memory from disk."""
        if os.path.exists(self.MEMORY_FILE):
            with open(self.MEMORY_FILE, 'r') as f:
                loaded = json.load(f)
            self.memory = self._default_memory()
            self.memory.update(loaded)
            
            # Remove retired techniques
            retired = [k for k in self.memory
                      if k.startswith("HARD_CUT|") or k.startswith("STUTTER_DROP|")]
            for k in retired:
                del self.memory[k]
            if retired:
                self.save_brain()
        else:
            self.memory = self._default_memory()
            self.save_brain()
    
    def save_brain(self):
        """Atomically save ML memory to disk."""
        with self._lock:
            snapshot = dict(self.memory)
        
        tmp = self.MEMORY_FILE + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(snapshot, f, indent=2)
            os.replace(tmp, self.MEMORY_FILE)
        except Exception as e:
            print(f"⚠️ Failed to save brain: {e}")
    
    def _load_confidence(self):
        """Load confidence scores from disk."""
        if os.path.exists(self.CONFIDENCE_FILE):
            try:
                with open(self.CONFIDENCE_FILE, 'r') as f:
                    self.confidence = json.load(f)
            except Exception:
                self.confidence = {}
        else:
            self.confidence = {}
    
    def _save_confidence(self):
        """Atomically save confidence scores."""
        tmp = self.CONFIDENCE_FILE + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(self.confidence, f, indent=2)
            os.replace(tmp, self.CONFIDENCE_FILE)
        except Exception:
            pass
    
    def generate_recipe(self, technique_name: str, energy_a: str, energy_b: str,
                       is_amapiano: bool, quiet: bool = False) -> Tuple[str, Dict]:
        """
        Generate technique recipe with learned parameters + randomization.
        
        Returns:
            (memory_key, recipe_dict)
        """
        pair = "Amapiano->Amapiano" if is_amapiano else f"{energy_a}->{energy_b}"
        key = f"{technique_name}|{pair}"
        
        if key not in self.memory:
            key = f"{technique_name}|High->High"
        
        base = self.memory[key]
        tech_id = TECHNIQUE_LIBRARY[technique_name]["id"]
        
        recipe = {
            "technique_name": technique_name,
            "technique_id": tech_id,
            "beats": max(float(TECHNIQUE_MIN_BEATS.get(technique_name, 64.0)),
                        base["beats"] + random.choice([-16, 0, 16])),
            "bass": clamp01(base["bass"] + random.uniform(-0.05, 0.05)),
            "echo": clamp01(base.get("echo", 0) + random.uniform(-0.15, 0.15)),
            "wash": clamp01(base.get("wash", 0) + random.uniform(-0.10, 0.10)),
            "piano_hold": clamp01(base.get("piano_hold", 0) + random.uniform(-0.15, 0.15)),
        }
        
        if technique_name == "ACAPELLA_MASHUP":
            recipe["acapella"] = clamp01(base.get("acapella", 1.0) + random.uniform(-0.10, 0.10))
        else:
            recipe["acapella"] = 0.0
        
        # Technique-specific constraints
        if technique_name == "PIANO_HANDOFF":
            recipe["echo"] = recipe["wash"] = 0.0
            recipe["bass"] = min(recipe["bass"], 0.55)
        elif technique_name in ("BASS_SWAP", "FILTER_SWEEP", "SLOW_BURN"):
            recipe["bass"] = min(recipe["bass"], 0.75)
        
        if technique_name == "FILTER_SWEEP":
            recipe["wash"] = clamp01(recipe["wash"] + 0.3)
        if technique_name == "ECHO_THROW":
            recipe["echo"] = 1.0
        
        if not quiet:
            print(f"\n🧪 [{TECHNIQUE_LIBRARY[technique_name]['label']}]  {pair}")
            print(f"   Beats:{recipe['beats']:.0f}  Bass@{recipe['bass']*100:.0f}%  "
                  f"Echo:{recipe['echo']:.2f}  Wash:{recipe['wash']:.2f}  "
                  f"PianoHold:{recipe['piano_hold']:.2f}")
        
        return key, recipe
    
    def learn_from_feedback(self, rating: int, mem_key: str, recipe: Dict,
                           failure_tags: list = None):
        """Update technique parameters based on user rating."""
        if mem_key not in self.memory:
            return
        
        # Rating to blend factor
        if rating <= 2:
            blend = -0.50
        elif rating <= 3:
            blend = -0.30
        elif rating <= 4:
            blend = -0.12
        elif rating <= 5:
            blend = 0.0
        elif rating <= 6:
            blend = 0.25
        elif rating <= 7:
            blend = 0.40
        elif rating <= 8:
            blend = 0.65
        elif rating <= 9:
            blend = 0.80
        else:
            blend = 1.0
        
        if blend == 0.0 and not failure_tags:
            return
        
        # Dampen learning for high-confidence memories
        conf = self.confidence.get(mem_key, 0)
        if conf > 10:
            dampening = max(0.3, 1.0 - (conf - 10) * 0.05)
            blend *= dampening
        
        baseline = self.memory[mem_key]
        updated = dict(baseline)
        
        # Blend learned values
        if blend != 0.0:
            for fld in ['beats', 'bass', 'echo', 'wash', 'piano_hold', 'acapella']:
                if fld not in recipe:
                    continue
                base_val = baseline.get(fld, 0.0)
                new_val = base_val + blend * (recipe[fld] - base_val)
                
                if fld == 'beats':
                    new_val = max(16.0, min(192.0, new_val))
                else:
                    new_val = clamp01(new_val)
                
                updated[fld] = round(new_val, 4)
        
        # Apply failure tag corrections
        if failure_tags:
            tags = set(failure_tags)
            
            if 'bass_fight' in tags and 'energy_dip' not in tags:
                updated['bass'] = max(0.0, updated.get('bass', 0.5) - 0.15)
            elif 'energy_dip' in tags and 'bass_fight' not in tags:
                updated['bass'] = min(1.0, updated.get('bass', 0.5) + 0.10)
            
            if 'too_abrupt' in tags and 'outgoing_too_long' not in tags:
                updated['beats'] = min(192.0, updated.get('beats', 64.0) + 16.0)
            elif 'outgoing_too_long' in tags and 'too_abrupt' not in tags:
                updated['beats'] = max(32.0, updated.get('beats', 96.0) - 16.0)
            elif 'no_payoff' in tags and 'too_abrupt' not in tags:
                updated['beats'] = max(32.0, updated.get('beats', 96.0) - 16.0)
            
            if 'vocal_clash' in tags:
                updated['echo'] = min(1.0, updated.get('echo', 0.0) + 0.20)
        
        with self._lock:
            self.confidence[mem_key] = conf + 1
            self.memory[mem_key] = updated
        
        self.save_brain()
        self._save_confidence()


class DJBrain:
    """Core harmonic and compatibility scoring engine."""
    
    def __init__(self):
        self.ml = PersistentLearner()
    
    def camelot_score(self, key_a: str, key_b: str) -> Tuple[int, str]:
        """
        Score harmonic compatibility using Camelot wheel.
        
        Returns:
            (score, label) where label in {exact, energy_boost, adjacent, two_step, clash, unknown}
        """
        ca, cb = CAMELOT_WHEEL.get(key_a), CAMELOT_WHEEL.get(key_b)
        if not ca or not cb:
            return 10, "unknown"
        
        if ca[0] == cb[0] and ca[1] == cb[1]:
            return 40, "exact"
        if ca[0] == cb[0] and ca[1] != cb[1]:
            return 20, "energy_boost"
        
        diff = min(abs(ca[0] - cb[0]), 12 - abs(ca[0] - cb[0]))
        if diff == 1:
            return 25, "adjacent"
        if diff == 2:
            return 10, "two_step"
        
        return -30, "clash"
    
    def key_compat(self, key_a: str, key_b: str) -> str:
        """Simplified compatibility: exact, compatible, or clash."""
        _, label = self.camelot_score(key_a, key_b)
        if label in ("exact", "energy_boost"):
            return "exact"
        elif label == "adjacent":
            return "compatible"
        else:
            return "clash"
    
    def bpm_score(self, bpm_a: float, bpm_b: float, is_amapiano: bool) -> int:
        """Score BPM compatibility (stricter for amapiano)."""
        diff = abs(bpm_b - bpm_a)
        
        if is_amapiano:
            if diff <= 1.0:
                return 35
            elif diff <= 2.0:
                return 15
            elif diff <= 3.0:
                return 0
            else:
                return -120
        else:
            if diff <= 1.0:
                return 35
            elif diff <= 3.0:
                return 20
            elif diff <= 6.0:
                return 5
            elif diff <= 8.0:
                return -10
            else:
                return -50
    
    def vocal_score(self, ta: dict, tb: dict) -> int:
        """Penalize vocal-on-vocal overlap."""
        av = ta.get('stems', {}).get('has_vocals', False)
        bv = tb.get('stems', {}).get('has_vocals', False)
        
        if av and bv:
            return -20
        elif not av and not bv:
            return 10
        else:
            return 0
    
    def spectral_score(self, ta: dict, tb: dict) -> int:
        """Reward matching energy profiles."""
        return 10 if ta.get('energy', 'High') == tb.get('energy', 'High') else -5
    
    def genre_score(self, ta: dict, tb: dict) -> int:
        """Amapiano-specific genre scoring."""
        ga, gb = ta.get('genre', 'open'), tb.get('genre', 'open')
        
        if ga == gb == 'amapiano':
            return 30
        if ga == 'amapiano':
            return -25
        if gb == 'amapiano':
            return -15
        
        return 0