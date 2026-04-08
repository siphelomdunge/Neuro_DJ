#!/usr/bin/env python3
"""
Neuro-DJ V49.4 PRODUCTION
Main orchestrator and entry point.

CRITICAL FIXES:
- Fixed executor.trigger_transition() call (missing parentheses causing method ref instead of call)
- Added heartbeat diagnostic on transition failure
- Enhanced emergency recovery pathway
- Improved transition monitoring logs
"""
from __future__ import annotations
import sys
import os
import json
import time
import threading
import warnings
import concurrent.futures
import gc
import copy

import librosa
import numpy as np
import soundfile as sf
import neuro_core

from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication

# Core modules
from core.constants import (
    DEFAULT_TRACK_DURATION, TOTAL_SAFETY_MARGIN, IMMEDIATE_TECHNIQUES,
    MAX_WAIT_SECONDS, EXECUTION_RESERVE_SEC, TECHNIQUE_LIBRARY
)
from core.brain import DJBrain
from core.search import PhraseCandidateSearch
from core.decision_core import DecisionCore, CandidateOption, EvaluatedOption, DecisionRecord, Intent
from core.context import build_transition_context
from core.technique_selector import select_technique

# Execution modules
from execution.planner import TransitionPlanner, RunwayExhausted
from execution.executor import TransitionExecutor
from execution.warp import TrackWarper

# Data modules
from data.dataset import TransitionDataset

# GUI modules
from gui.window import NeuroDJWindow

# Optional modules
try:
    from crate_ranker import CrateRanker
    CRATE_RANKER_AVAILABLE = True
except ImportError:
    CRATE_RANKER_AVAILABLE = False
    print("⚠️  crate_ranker.py not found — crate ranking disabled")

try:
    from set_state import SetStateModel
    SET_STATE_AVAILABLE = True
except ImportError:
    SET_STATE_AVAILABLE = False
    print("⚠️  set_state.py not found — set state tracking disabled")

try:
    from telemetry_logger import TransitionTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    print("⚠️  telemetry_logger.py not found — telemetry disabled")


class NeuroDJ(QObject):
    
    request_rating = pyqtSignal(str, object, object)
    
    def __init__(self, library_json: str):
        super().__init__()
        
        self.mixer = neuro_core.NeuroMixer()
        self.brain = DJBrain()
        self.dataset = TransitionDataset()
        self.searcher = PhraseCandidateSearch()
        self.warper = TrackWarper()
        
        self._analysis_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._learn_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        self._set_state_lock = threading.Lock()
        self._tx_lock = threading.Lock()
        
        self.track_a_dur = DEFAULT_TRACK_DURATION
        self.track_b_dur = DEFAULT_TRACK_DURATION
        self.track_a_name = "Loading..."
        self.track_b_name = "—"
        self.master_bpm = 0
        self._master_bpm_precise = 0.0
        self.current_mix_trigger = 0.0
        self.current_technique = "—"
        self.mix_count = 0
        self.energy_history = []
        
        self._pending_tx_id = None
        self._load_failures = set()
        
        self._total_pause_time = 0.0
        self._pause_start_time = 0.0
        self.is_paused = False
        
        self.target_set_duration = 3600.0
        
        with open(library_json, 'r') as f:
            self.crate = json.load(f)
        
        for t in self.crate:
            if '_duration' not in t:
                fn = t.get('filename', '')
                t['_duration'] = (librosa.get_duration(path=fn)
                                 if os.path.exists(fn) else DEFAULT_TRACK_DURATION)
        
        if SET_STATE_AVAILABLE:
            self.set_state = SetStateModel()
        else:
            self.set_state = None
        
        if CRATE_RANKER_AVAILABLE:
            self.crate_ranker = CrateRanker(self.brain)
        else:
            self.crate_ranker = None
        
        self.decision_core = DecisionCore(
            brain=self.brain,
            searcher=self.searcher,
            technique_selector=select_technique,
            context_builder=build_transition_context,
            set_state_model=self.set_state,
            now_fn=time.time
        )
        self.decision_core.dj = self
        
        self.planner = TransitionPlanner(
            mixer=self.mixer,
            warper=self.warper,
            brain=self.brain,
            searcher=self.searcher,
            context_builder=build_transition_context
        )
        
        self.executor = TransitionExecutor(
            mixer=self.mixer,
            master_bpm_fn=lambda: self._master_bpm_precise,
            is_paused_fn=lambda: self.is_paused,
            sleep_pausable_fn=self._sleep_pausable
        )
    
    def _sleep_pausable(self, seconds: float):
        end_time = time.time() + seconds
        
        while time.time() < end_time:
            if self.is_paused:
                time.sleep(0.1)
                end_time += 0.1
                continue
            time.sleep(max(0.0, min(0.1, end_time - time.time())))
    
    def _is_viable(self, curr: dict, cand: dict) -> bool:
        is_amp = (curr.get('genre') == 'amapiano' or cand.get('genre') == 'amapiano')
        bpm_limit = 4.0 if is_amp else 8.0
        
        if abs(cand.get('bpm', 112.0) - curr.get('bpm', 112.0)) > bpm_limit:
            return False
        
        from core.constants import CAMELOT_WHEEL
        ca = CAMELOT_WHEEL.get(curr.get('key', 'C'))
        cb = CAMELOT_WHEEL.get(cand.get('key', 'C'))
        
        if ca and cb:
            diff = abs(ca[0] - cb[0])
            if min(diff, 12 - diff) >= 4:
                return False
        
        if not cand.get('best_entry_phrases'):
            has_intro = any(s.get('label') in ('intro', 'breakdown')
                          for s in cand.get('structure_map', []))
            if not has_intro:
                spb = 60.0 / max(cand.get('bpm', 112.0), 60.0)
                rough_intro_beats = (cand.get('_duration', 300.0) * 0.15) / spb
                if rough_intro_beats < 16:
                    return False
        
        return True
    
    def _load_opener(self) -> dict | None:
        while self.crate:
            valid = [t for t in self.crate if t.get('filename') not in self._load_failures]
            if not valid:
                print("🛑 No loadable tracks in crate.")
                return None
            
            opener = min(valid, key=lambda t: (
                t.get('bpm', 112.0) +
                (5 if t.get('stems', {}).get('has_vocals', False) else 0)
            ))
            
            self.crate.remove(opener)
            ta = opener
            ta['stretch_ratio'] = 1.0
            filename = ta.get('filename', '')
            
            if not os.path.exists(filename):
                continue
            
            try:
                self.track_a_dur = librosa.get_duration(path=filename)
            except Exception:
                self.track_a_dur = DEFAULT_TRACK_DURATION
            
            try:
                self.track_a_name = os.path.basename(filename)
                load_path = os.path.abspath(filename)
                
                if not filename.lower().endswith('.wav'):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y, sr = librosa.load(filename, sr=44100, mono=False)
                    
                    tmp_a = os.path.abspath(f"temp_deck_a_{time.time():.0f}.wav")
                    self.warper._temp_files.append(tmp_a)
                    
                    if y.ndim == 1:
                        y = np.array([y, y])
                    
                    sf.write(tmp_a, y.T.astype(np.float32), 44100)
                    load_path = tmp_a
                
                self.mixer.load_deck("A", load_path)
                self.mixer.play("A")
                
                self._master_bpm_precise = float(ta.get('bpm', 112.0)) or 112.0
                self.master_bpm = round(self._master_bpm_precise)
                
                print(f"\n🎵 NOW PLAYING: {self.track_a_name} "
                      f"({self._master_bpm_precise:.1f} BPM, {self.track_a_dur:.0f}s)")
                
                self.energy_history.append(ta.get('energy', 'High'))
                return ta
            
            except Exception as e:
                print(f"⚠️  Failed to load opener: {e}")
                self._load_failures.add(filename)
                self.crate.append(ta)
                continue
        
        return None
    
    def start_set(self):
        
        if not self.crate:
            return
        
        ta = self._load_opener()
        if ta is None:
            return
        
        set_start_time = time.time()
        attempted_tracks = set()
        
        while self.crate:
            
            self.crate = [t for t in self.crate
                         if t.get('filename') not in self._load_failures]
            if not self.crate:
                print("🛑 Crate exhausted or all remaining tracks corrupted.")
                break
            
            print(f"\n{'='*60}")
            print(f"🔄 MIX #{self.mix_count + 1} — Starting decision cycle")
            print(f"   Crate size: {len(self.crate)}")
            print(f"   Track A: {self.track_a_name}")
            print(f"   Position: {self.mixer.get_position('A'):.1f}s / {self.track_a_dur:.1f}s")
            print(f"{'='*60}")
            
            spb = 60.0 / max(self._master_bpm_precise, 60.0)
            elapsed = (time.time() - set_start_time) - self._total_pause_time
            set_dur_remaining = max(0.0, self.target_set_duration - elapsed)
            
            print(f"⚙️  DecisionCore evaluating options...")
            
            viable_crate = [t for t in self.crate if self._is_viable(ta, t)]
            if not viable_crate:
                print("   ⚠️  No viable candidates — using least-bad fallback")
                viable_crate = sorted(
                    self.crate,
                    key=lambda c: abs(c.get('bpm', 112) - ta.get('bpm', 112))
                )[:3]
            else:
                vetoed = len(self.crate) - len(viable_crate)
                if vetoed:
                    print(f"   🔍 Pre-filter: {vetoed} vetoed, {len(viable_crate)} viable")
            
            current_energy_snapshot = list(self.energy_history) + [ta.get('energy', 'High')]
            
            if self.crate_ranker:
                try:
                    candidate_pool_full = self.crate_ranker.select_candidates(
                        current_track=ta,
                        crate=viable_crate,
                        master_bpm=self._master_bpm_precise,
                        n=8,
                        energy_history=current_energy_snapshot
                    )
                    
                    candidate_pool = candidate_pool_full[:3]
                    
                    if candidate_pool:
                        top_names = [os.path.basename(t.get('filename', '?'))[:30]
                                    for t in candidate_pool]
                        print(f"   🎯 CrateRanker Top 3: {', '.join(top_names)}")
                
                except Exception as e:
                    print(f"   ⚠️ CrateRanker failed: {e}")
                    candidate_pool = viable_crate[:3]
            else:
                candidate_pool = viable_crate[:3]
            
            candidate_pool = [t for t in candidate_pool
                             if t.get('filename') not in attempted_tracks]
            
            if not candidate_pool:
                print(f"   ⚠️ Empty candidate pool — using raw viable crate")
                candidate_pool = [t for t in viable_crate[:3]
                                 if t.get('filename') not in attempted_tracks]
            
            if not candidate_pool:
                print("   💀 No candidates at all. Ending set.")
                break
            
            print(f"   📦 Candidate pool passed to DecisionCore: {len(candidate_pool)} tracks")
            
            with self.dataset._lock:
                trans_history = copy.deepcopy(self.dataset.records[-5:])
            
            print(f"   🧠 Submitting decision to analysis pool...")
            decision_future = self._analysis_pool.submit(
                self.decision_core.decide_next_action,
                current_track=ta,
                candidate_tracks=candidate_pool,
                current_pos_a=self.mixer.get_position("A"),
                track_a_dur=self.track_a_dur,
                master_bpm=self._master_bpm_precise,
                spb=spb,
                mix_count=self.mix_count,
                energy_history=current_energy_snapshot,
                transition_history=trans_history,
                set_duration_remaining=set_dur_remaining,
            )
            
            decision = None
            wait_start = time.time()
            wait_deadline = wait_start + MAX_WAIT_SECONDS
            pause_extension = 0.0
            
            print(f"   ⏳ Waiting for decision (max {MAX_WAIT_SECONDS}s)...")
            
            while decision is None:
                now = time.time()
                elapsed_wait = now - wait_start
                
                if self.is_paused:
                    time.sleep(0.1)
                    pause_extension += 0.1
                    from core.constants import MAX_PAUSE_EXTENSION
                    if pause_extension < MAX_PAUSE_EXTENSION:
                        wait_deadline += 0.1
                    else:
                        print("⚠️ Pause exceeded 30 min — forcing decision timeout")
                    continue
                
                if now > wait_deadline:
                    print(f"💀 DECISION TIMEOUT after {elapsed_wait:.1f}s — forcing emergency mix")
                    break
                
                if int(elapsed_wait) % 5 == 0 and elapsed_wait > 0 and int(elapsed_wait) != int(elapsed_wait - 0.1):
                    print(f"   ⏳ Still waiting for decision... {elapsed_wait:.0f}s elapsed")
                
                try:
                    decision = decision_future.result(timeout=1.0)
                except concurrent.futures.TimeoutError:
                    pass
                except Exception as e:
                    print(f"❌ Decision thread crashed: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            if decision is not None and decision.chosen is not None:
                n_trans = sum(1 for a in decision.alternatives
                             if a.option.option_type == "transition" and a.draft_plan is not None)
                n_holds = sum(1 for a in decision.alternatives if a.option.option_type == "hold")
                print(f"   📊 Evaluated: {n_trans} transitions, {n_holds} holds, "
                      f"best={decision.chosen.option.label()} "
                      f"score={decision.chosen.final_score:.3f}")
            else:
                print(f"   💀 Decision returned None or has no chosen option!")
            
            if (decision is None or decision.chosen is None or
                decision.chosen.option.option_type == "hold"):
                
                time_left = self.track_a_dur - self.mixer.get_position("A")
                force_threshold = max(60.0, 48.0 * spb + 20.0)
                
                if time_left < force_threshold:
                    print(f"   ⚠️  Only {time_left:.0f}s left. Forcing transition!")
                    
                    if decision and decision.alternatives:
                        n_avail = sum(1 for a in decision.alternatives
                                     if a.option.option_type == "transition" and a.draft_plan)
                        print(f"   🚨 FORCE: {n_avail} transition alternatives available")
                    
                    forced_opt = None
                    if decision and decision.alternatives:
                        forced_opt = next(
                            (alt for alt in decision.alternatives
                             if alt.option.option_type == "transition" and alt.draft_plan is not None),
                            None
                        )
                    
                    if forced_opt:
                        decision = DecisionRecord(
                            intent=decision.intent,
                            chosen=forced_opt,
                            alternatives=decision.alternatives,
                            timestamp=decision.timestamp
                        )
                    else:
                        print("   🆘 EMERGENCY: No valid options left. Forcing immediate fallback mix!")
                        emergency_tb = next(
                            (t for t in candidate_pool if t.get('filename') not in attempted_tracks),
                            next((t for t in self.crate if t.get('filename') not in attempted_tracks), None)
                        )
                        
                        if emergency_tb is None:
                            print("💀 All tracks exhausted in emergency loop. Ending set.")
                            break
                        
                        attempted_tracks.add(emergency_tb.get('filename'))
                        
                        dummy_opt = CandidateOption("transition", emergency_tb, 0.0)
                        dummy_eval = EvaluatedOption(
                            option=dummy_opt, final_score=0.0, intent_fit=0.0,
                            transition_opportunity=0.0, global_compat=0.0,
                            robustness=0.0, execution_risk=1.0, future_flex=0.0,
                            draft_plan=None
                        )
                        decision = DecisionRecord(
                            intent=Intent("recovery", 0, 0, 0, "low", "low", [], [], 1.0),
                            chosen=dummy_eval,
                            alternatives=[],
                            timestamp=time.time()
                        )
                else:
                    delay = (decision.chosen.option.delay_beats
                            if (decision and decision.chosen) else 32.0)
                    max_sleep = max(2.0, (time_left - force_threshold) * 0.3)
                    actual_sleep = min(delay * spb * 0.5, max_sleep)
                    print(f"   ⏳ Hold: sleeping {actual_sleep:.1f}s ({time_left:.0f}s remaining)")
                    self._sleep_pausable(actual_sleep)
                    continue
            
            print("🧠 DECISION LOCKED:")
            print(f"   Goal: {decision.intent.goal}")
            print(f"   Action: {decision.chosen.option.label()}")
            
            chosen_track = decision.chosen.option.track_b
            if chosen_track:
                if chosen_track in candidate_pool:
                    rank = candidate_pool.index(chosen_track) + 1
                    print(f"   ✅ Selected CrateRanker Top-{rank} recommendation")
                else:
                    print(f"   🚨 WARNING: Selected track bypassed the candidate pool!")
            
            for r in decision.chosen.reasoning[:5]:
                print(f"   - {r}")
            
            if len(decision.alternatives) > 1:
                print(f"   📊 Decision Scores:")
                for alt in decision.alternatives[:6]:
                    if alt.option.track_b and alt.option.track_b in candidate_pool:
                        track_name = os.path.basename(alt.option.track_b.get('filename', '?'))[:30]
                        print(f"      {alt.final_score:.3f} — {track_name} @ {alt.option.delay_beats}b")
            
            tb = decision.chosen.option.track_b
            if tb in self.crate:
                self.crate.remove(tb)
            self.track_b_name = os.path.basename(tb.get('filename', '?'))
            
            print(f"   🔧 Finalizing transition plan...")
            try:
                plan = self.planner.finalize_plan(
                    decision.chosen, ta, spb, self.track_a_dur, self._master_bpm_precise
                )
                self.track_b_dur = plan.track_b_dur
            except RunwayExhausted as e:
                print(f"   ⚠️ Runway exhausted, skipping track: {e}")
                self._load_failures.add(tb.get('filename'))
                attempted_tracks.add(tb.get('filename'))
                continue
            except Exception as e:
                print(f"   💀 Plan finalization crashed: {e}")
                import traceback
                traceback.print_exc()
                self._load_failures.add(tb.get('filename'))
                attempted_tracks.add(tb.get('filename'))
                continue
            
            self.current_mix_trigger = plan.mix_trigger
            self.current_technique = TECHNIQUE_LIBRARY[plan.tech_name]['label']
            
            trans_end_time = plan.mix_trigger + plan.trans_dur
            
            if trans_end_time > self.track_a_dur - 1.0:
                overshoot = trans_end_time - (self.track_a_dur - 1.0)
                print(f"\n{'=' * 60}")
                print(f"🚨 PRE-FLIGHT EOF VALIDATION FAILED")
                print(f"{'=' * 60}")
                print(f"Track A EOF:        {self.track_a_dur:.1f}s")
                print(f"Mix trigger:        {plan.mix_trigger:.1f}s")
                print(f"Transition dur:     {plan.trans_dur:.1f}s")
                print(f"Planned end:        {trans_end_time:.1f}s")
                print(f"Overshoot:          {overshoot:.1f}s ❌")
                print(f"{'=' * 60}\n")
                
                available_runway = max(0.0, self.track_a_dur - plan.mix_trigger - 1.5)
                
                if available_runway < 4.0:
                    print(f"   💀 PANIC: Cannot fit minimum 4s transition (have {available_runway:.1f}s). Aborting mix!")
                    self._load_failures.add(tb.get('filename'))
                    attempted_tracks.add(tb.get('filename'))
                    self.crate.append(tb)
                    continue
                
                safe_trans_dur = available_runway
                plan.update_duration(safe_trans_dur, spb)
                
                if safe_trans_dur < 16.0 * spb:
                    plan.tech_name = "ECHO_THROW"
                    plan.recipe['technique_name'] = "ECHO_THROW"
                    plan.recipe['technique_id'] = 3
                
                print(f"   ✅ Emergency shrink: transition now {safe_trans_dur:.1f}s")
                print(f"   ✅ Will end at {plan.mix_trigger + safe_trans_dur:.1f}s "
                      f"(EOF at {self.track_a_dur:.1f}s)\n")
            
            print(f"   ⏳ Waiting for trigger point ({plan.mix_trigger:.1f}s)...")
            self.executor.wait_for_trigger(plan, self.track_a_dur)
            
            print(f"   💿 Loading Deck B: {plan.fnb}")
            try:
                self.mixer.load_deck("B", plan.fnb)
                print(f"   ✅ Deck B loaded successfully")
            except Exception as e:
                print(f"💀 Failed to load Deck B ({plan.fnb}): {e}")
                import traceback
                traceback.print_exc()
                self.current_mix_trigger = 0.0
                self.current_technique = "—"
                self.track_b_name = "—"
                self._load_failures.add(tb.get('filename'))
                attempted_tracks.add(tb.get('filename'))
                self.crate.append(tb)
                continue
            
            pos_after_warp = self.mixer.get_position("A")
            if pos_after_warp >= plan.mix_trigger - 1.0:
                gap = pos_after_warp - plan.mix_trigger
                print(f"   ⚠️ CPU Lag detected: missed planned trigger by {gap:.1f}s")
                
                plan.mix_trigger = pos_after_warp + 2.0
                remaining = self.track_a_dur - plan.mix_trigger - EXECUTION_RESERVE_SEC
                
                if remaining < 16.0 * spb:
                    print(f"   ⚠️ Track nearly over — falling back to safe Echo Out")
                    plan.tech_name = "ECHO_THROW"
                    
                    _, emergency_recipe = self.brain.ml.generate_recipe(
                        "ECHO_THROW", ta.get('energy', 'High'), tb.get('energy', 'High'),
                        ta.get('genre') == 'amapiano', quiet=True
                    )
                    emergency_recipe['beats'] = max(4.0, remaining) / spb
                    plan.recipe = emergency_recipe
                    plan.trans_dur = max(4.0, remaining)
                else:
                    print(f"   🔄 Shifting trigger to {plan.mix_trigger:.1f}s")
                
                absolute_max_trans_dur = self.track_a_dur - plan.mix_trigger - 1.0
                if plan.trans_dur > absolute_max_trans_dur:
                    print(f"   🚨 EOF BOUNDARY VIOLATION: Clamping to {absolute_max_trans_dur:.1f}s")
                    plan.update_duration(max(2.0, absolute_max_trans_dur), spb)
            
            if self.crate_ranker:
                self.crate_ranker.record_selection(tb)
            
            self.mixer.seek("B", plan.b_start_w)
            
            print(f"   🎯 Final precision wait to trigger point...")
            stall_start = None
            last_pos = self.mixer.get_position("A")
            
            while True:
                pos = self.mixer.get_position("A")
                tl = plan.mix_trigger - pos
                if tl <= 0:
                    break
                
                if self.is_paused:
                    time.sleep(0.1)
                    stall_start = None
                    last_pos = self.mixer.get_position("A")
                    continue
                
                if pos <= last_pos:
                    if stall_start is None:
                        stall_start = time.time()
                    elif time.time() - stall_start > 5.0:
                        print(f"   ⚠️  Audio stalled — forcing trigger now")
                        break
                else:
                    stall_start = None
                last_pos = pos
                
                if tl > 0.05:
                    time.sleep(min(tl * 0.5, 0.1))
                elif tl > 0.005:
                    time.sleep(0.001)
                else:
                    deadline = time.perf_counter() + tl
                    while time.perf_counter() < deadline:
                        pass
            
            tx_id = f"tx_{int(time.time())}_{self.mix_count}"
            with self._tx_lock:
                self._pending_tx_id = tx_id
            
            pending_record = {
                "tx_id": tx_id,
                "ta": ta, "tb": tb,
                "plan": plan.to_dict(),
                "ctx": plan.ctx,
                "recipe": plan.recipe,
                "overlap_score": plan.overlap_score,
                "assumptions": plan.assumptions,
            }
            
            self.executor.schedule_eq_events(ta, tb, plan, spb)
            
            current_telemetry = None
            if TELEMETRY_AVAILABLE:
                current_telemetry = TransitionTelemetry(
                    tx_id, plan.to_dict(), ta.get('filename', '?'), tb.get('filename', '?')
                )
            
            print(f"   🎬 TRIGGERING TRANSITION NOW")
            print(f"      Technique: {plan.tech_name}")
            print(f"      Duration: {plan.trans_dur:.1f}s")
            print(f"      Beats: {plan.recipe['beats']:.0f}")
            
            # ✅ CRITICAL FIX: Call the method with parentheses!
            try:
                handoff_forced = self.executor.trigger_transition(plan, current_telemetry)
                print(f"   ✅ Transition execution complete (forced={handoff_forced})")
            except Exception as e:
                print(f"💀 Transition execution crashed: {e}")
                import traceback
                traceback.print_exc()
                
                # ✅ Enhanced diagnostics
                print(f"\n🔍 MIXER DIAGNOSTICS:")
                print(f"   Heartbeat: {self.mixer.get_callback_heartbeat()}")
                print(f"   Lock failures: {self.mixer.get_lock_failure_count()}")
                print(f"   Is transitioning: {self.mixer.is_transitioning()}")
                print(f"   Volume A: {self.mixer.get_volume('A'):.3f}")
                print(f"   Volume B: {self.mixer.get_volume('B'):.3f}")
                
                handoff_forced = False
            
            self._learn_pool.submit(
                self.dataset.log_transition_async,
                pending_record["tx_id"],
                pending_record["ta"],
                pending_record["tb"],
                pending_record["plan"],
                pending_record["ctx"],
                pending_record["recipe"],
                pending_record["overlap_score"],
                pending_record["assumptions"]
            )
            
            self.executor.post_transition_sleep(plan.tech_name)
            time.sleep(0.2)
            
            try:
                self.mixer.set_acapella_mode("A", 0.0)
                self.mixer.set_acapella_mode("B", 0.0)
            except Exception:
                pass
            
            print(f"   🔀 Swapping decks...")
            self.mixer.swap_decks()
            
            try:
                self.mixer.set_acapella_mode("A", 0.0)
            except Exception:
                pass
            
            new_bpm = float(tb.get('bpm', self._master_bpm_precise))
            if new_bpm > 0:
                if abs(new_bpm - self._master_bpm_precise) < 1.0:
                    self._master_bpm_precise = new_bpm
                else:
                    bpm_delta = new_bpm - self._master_bpm_precise
                    capped_delta = max(-2.0, min(2.0, bpm_delta))
                    self._master_bpm_precise += 0.3 * capped_delta
                
                if self.mix_count % 10 == 0:
                    self._master_bpm_precise = round(self._master_bpm_precise)
                    print(f"   🎯 BPM recalibrated to {self._master_bpm_precise:.1f}")
                
                self.master_bpm = round(self._master_bpm_precise)
            
            tb_energy = tb.get('energy', 'High')
            self.track_a_dur = plan.track_b_dur
            self.track_a_name = self.track_b_name
            self.track_b_name = "—"
            self.current_mix_trigger = 0.0
            self.current_technique = "—"
            self.mix_count += 1
            attempted_tracks.clear()
            
            self.energy_history.append(tb_energy)
            if len(self.energy_history) > 100:
                self.energy_history = self.energy_history[-100:]
            
            if self.set_state:
                def _update_with_lock():
                    with self._set_state_lock:
                        self.set_state.update_transition(ta, tb, plan.tech_name)
                self._learn_pool.submit(_update_with_lock)
            
            print(f"   📊 Scheduling rating panel (800ms delay)...")
            
            _mem_key = plan.mem_key
            _recipe = dict(plan.recipe)
            _telemetry = current_telemetry
            
            QTimer.singleShot(800, lambda: self.request_rating.emit(
                _mem_key, _recipe, _telemetry
            ))
            
            ta = tb
            self.warper.cleanup_temp_files(keep_last=True)
            gc.collect()
            
            print(f"✅ Mix #{self.mix_count} complete\n")
        
        print(f"\n🏁 Setlist complete. '{self.track_a_name}' playing out.")
        self.brain.ml.save_brain()
        self.warper.cleanup_temp_files(keep_last=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py master_library.json [set_duration_seconds]")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    dj = NeuroDJ(sys.argv[1])
    dj.target_set_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
    win = NeuroDJWindow(dj)
    win.show()
    threading.Thread(target=dj.start_set, daemon=True).start()
    sys.exit(app.exec_())