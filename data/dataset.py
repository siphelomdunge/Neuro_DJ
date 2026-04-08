"""
Transition dataset logging and analysis.
"""
from __future__ import annotations
import os
import json
import time
import threading
from typing import List, Dict, Optional


class TransitionDataset:
    """Persistent logging of all transitions with ratings."""
    
    LOG_FILE = "transition_log.json"
    FAILURE_TAGS = [
        "bass_fight", "vocal_clash", "energy_dip", "too_abrupt", "no_payoff",
        "phrase_mismatch", "tonal_mismatch", "groove_drift",
        "intro_too_empty", "outgoing_too_long"
    ]
    MAX_RECORDS = 1000
    
    def __init__(self):
        self.records: List[Dict] = []
        self._lock = threading.Lock()
        self._load_records()
    
    def _load_records(self):
        """Load existing records from disk."""
        if os.path.exists(self.LOG_FILE):
            try:
                with open(self.LOG_FILE) as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load transition log: {e}")
    
    def _save(self):
        """Atomically save records to disk."""
        with self._lock:
            snapshot = list(self.records)
        
        tmp = self.LOG_FILE + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(snapshot, f, indent=2)
            os.replace(tmp, self.LOG_FILE)
        except Exception as e:
            print(f"⚠️ Failed to save transition log: {e}")
    
    def log_transition_async(self, rec_id: str, ta: dict, tb: dict, plan: dict,
                            ctx: dict, recipe: dict, overlap_score: dict,
                            assumptions: dict = None):
        """
        Log a transition asynchronously (thread-safe).
        
        Args:
            rec_id: Unique transition ID
            ta: Track A metadata
            tb: Track B metadata
            plan: Transition plan dict
            ctx: Transition context
            recipe: Recipe parameters
            overlap_score: Overlap scoring results
            assumptions: Planning assumptions
        """
        record = {
            "id": rec_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "track_a": os.path.basename(ta.get('filename', '?')),
            "track_b": os.path.basename(tb.get('filename', '?')),
            "bpm": plan.get('_master_bpm', 0),
            "key_a": ta.get('key', '?'),
            "key_b": tb.get('key', '?'),
            "technique": plan['tech_name'],
            "acapella_method": plan.get('acapella_method', 'none'),
            "recipe": {k: v for k, v in recipe.items()
                      if k not in ('technique_name', 'technique_id')},
            "mix_trigger": round(plan['mix_trigger'], 2),
            "b_start_w": round(plan['b_start_w'], 2),
            "trans_dur": round(plan['trans_dur'], 2),
            "a_exit_section": ctx.get('a_exit_section', '?'),
            "a_exit_function": ctx.get('a_exit_phrase_func') or '?',
            "a_exit_density": round(ctx.get('a_exit_density', 0), 3),
            "a_exit_mixability": round(ctx.get('a_exit_mixability', 0), 3),
            "a_exit_bass": round(ctx.get('a_exit_bass', 0), 3),
            "a_exit_vocal": round(ctx.get('a_exit_vocal', 0), 3),
            "a_exit_tension": round(ctx.get('a_exit_tension', 0), 3),
            "b_entry_section": ctx.get('b_entry_section', '?'),
            "vocal_clash": ctx.get('vocal_clash', False),
            "dance_moment": ctx.get('dance_moment', '?'),
            "overlap_score": (round(overlap_score.get('total', 0), 3)
                            if isinstance(overlap_score, dict) else 0.0),
            "assumptions": assumptions or {},
            "rating": None,
            "failure_tags": [],
            "adaptation_actions": [],
        }
        
        with self._lock:
            self.records.append(record)
            if len(self.records) > self.MAX_RECORDS:
                self.records = self.records[-self.MAX_RECORDS:]
        
        self._save()
    
    def update_rating(self, rec_id: str, rating: int, failure_tags: List[str]):
        """
        Update a transition record with user rating.
        
        Args:
            rec_id: Transition ID
            rating: Rating 1-10
            failure_tags: List of failure mode tags
        """
        with self._lock:
            for rec in self.records:
                if rec['id'] == rec_id:
                    rec['rating'] = rating
                    rec['failure_tags'] = failure_tags
                    break
        
        self._save()
    
    def log_adaptation(self, rec_id: str, actions: List[str]):
        """
        Log adaptation actions taken during transition.
        
        Args:
            rec_id: Transition ID
            actions: List of action descriptions
        """
        if not actions:
            return
        
        with self._lock:
            for rec in self.records:
                if rec['id'] == rec_id:
                    rec['adaptation_actions'] = actions
                    break
        
        self._save()
    
    def get_overlap_score(self, tx_id: str) -> float:
        """
        Get overlap score for a transition.
        
        Args:
            tx_id: Transition ID
        
        Returns:
            Overlap score (0.0-1.0)
        """
        with self._lock:
            for rec in self.records:
                if rec['id'] == tx_id:
                    return rec.get('overlap_score', 0.0)
        return 0.0
    
    def summary(self) -> str:
        """Generate dataset summary statistics."""
        with self._lock:
            rated = [r for r in self.records if r.get('rating') is not None]
            total = len(self.records)
        
        if not rated:
            return "No rated transitions yet."
        
        avg = sum(r['rating'] for r in rated) / len(rated)
        return f"📊 Transition Dataset: {total} total, {len(rated)} rated (Avg: {avg:.2f}/10)"