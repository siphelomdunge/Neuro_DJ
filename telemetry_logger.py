"""
telemetry_logger.py — Neuro-DJ Evaluation Harness
"""

import json
import os
import time
from datetime import datetime
import threading


class TransitionTelemetry:
    def __init__(self, tx_id: str, plan: dict, track_a_path: str,
                 track_b_path: str, log_dir="logs/transitions"):
        self.log_dir = log_dir
        self.tx_id = tx_id  # canonical ID shared with TransitionDataset
        self.start_time = datetime.now().isoformat()
        self._lock = threading.Lock()

        recipe = plan.get('recipe', {})
        self.metadata = {
            "track_a": os.path.basename(track_a_path),
            "track_b": os.path.basename(track_b_path),
            "technique_id": recipe.get('technique_id', 0),
            "technique_name": recipe.get('technique_name', 'UNKNOWN'),
            "planned_duration_beats": recipe.get('beats', 32),
            "mix_trigger_sec": plan.get('mix_trigger', 0.0)
        }

        ctx = plan.get('ctx', {})
        self.phrase_attribution = {
            "a_exit_section": ctx.get('a_exit_section', 'unknown'),
            "a_exit_mixability": ctx.get('a_exit_mixability', 0.0),
            "b_entry_section": ctx.get('b_entry_section', 'unknown'),
            "dance_moment": ctx.get('dance_moment', 'unknown')
        }

        self.timeline_10hz = []
        self.adaptation_log = []
        self._tick_counter = 0

    def log_tick(self, beat_elapsed: float, state):
        with self._lock:
            self._tick_counter += 1
            self.timeline_10hz.append({
                "tick": self._tick_counter,
                "beat": round(beat_elapsed, 2),
                "a_low": round(state.a_low, 3),
                "b_low": round(state.b_low, 3),
                "combined_low_congestion": round(state.combined_low_congestion, 3),
                "beat_offset": round(state.beat_offset, 3),
                "drift_trend": round(state.drift_trend, 3),
                "vocal_clash_active": state.vocal_clash_active,
                "b_groove_established": state.b_groove_established,
                "a_space_opened": state.a_space_opened,
                "readiness": round(state.readiness_score, 3)
            })

    def log_action(self, beat_elapsed: float, action_name: str, state):
        with self._lock:
            self.adaptation_log.append({
                "beat": round(beat_elapsed, 2),
                "action": action_name,
                "trigger_state": {
                    "combined_low_congestion": round(
                        state.combined_low_congestion, 3),
                    "beat_offset": round(state.beat_offset, 3),
                    "readiness": round(state.readiness_score, 3)
                }
            })

    def finalize_and_save(self, rating: int, failure_tags: list):
        with self._lock:
            record = {
                "transition_id": self.tx_id,
                "timestamp": self.start_time,
                "metadata": dict(self.metadata),
                "phrase_attribution": dict(self.phrase_attribution),
                "evaluation": {
                    "rating": rating,
                    "failure_tags": list(failure_tags)
                },
                "adaptation_log": list(self.adaptation_log),
                "timeline_10hz": list(self.timeline_10hz)
            }

        os.makedirs(self.log_dir, exist_ok=True)
        file_path = os.path.join(self.log_dir, f"{self.tx_id}.json")
        tmp_path = file_path + ".tmp"

        try:
            with open(tmp_path, 'w') as f:
                json.dump(record, f, indent=2)
            os.replace(tmp_path, file_path)
            print(f"   💾 Telemetry saved: {self.tx_id}.json")
        except Exception as e:
            print(f"   ⚠️ Failed to save telemetry: {e}")