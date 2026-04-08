"""
Main Qt GUI window for Neuro-DJ.
"""
from __future__ import annotations
import sys
import time
import threading

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QTextEdit, QFrame, QPushButton, QSlider, QCheckBox
)
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QTextCursor

from data.dataset import TransitionDataset
from .stream import Stream


class NeuroDJWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, dj):
        """
        Initialize GUI window.
        
        Args:
            dj: NeuroDJ orchestrator instance
        """
        super().__init__()
        self.dj = dj
        self._old_stdout = sys.stdout
        self._rating_lock = threading.Lock()  # ✅ FIX #3: Dedicated rating lock
        self._msg_lock_expiry = 0.0
        
        self.setWindowTitle("Neuro-DJ — Dynamic Technique Intelligence")
        self.setGeometry(100, 100, 1080, 880)
        self.setStyleSheet("background:#121212;color:#FFF;")
        
        # Build UI
        self._build_ui()
        
        # Redirect stdout
        sys.stdout = Stream(newText=self._append_console)
        
        # Connect signals
        self.dj.request_rating.connect(self._show_rating)
        
        # State
        self.pending_rating = None
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
    
    def _build_ui(self):
        """Build the complete UI layout."""
        root = QVBoxLayout()
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(root)
        
        # Header
        header = QLabel("🧠 NEURO-DJ MAINSTAGE")
        header.setFont(QFont("Arial", 22, QFont.Bold))
        header.setStyleSheet("color:#00FFCC;padding:4px 0;")
        root.addWidget(header)
        
        # Deck rows
        deck_row = QHBoxLayout()
        self._build_deck(deck_row, "A", "DECK A", "#FF0055")
        self._build_deck(deck_row, "B", "DECK B", "#0088FF")
        root.addLayout(deck_row)
        
        # Info bar
        info_frame = self._build_info_bar()
        root.addWidget(info_frame)
        
        # Mix progress
        self.prog_mix = QProgressBar()
        self.prog_mix.setRange(0, 100)
        self.prog_mix.setValue(0)
        self.prog_mix.setTextVisible(False)
        self.prog_mix.setFixedHeight(6)
        self.prog_mix.setStyleSheet(
            "QProgressBar{background:#1A1A1A;border-radius:3px;}"
            "QProgressBar::chunk{background:#FF5500;border-radius:3px;}"
        )
        root.addWidget(self.prog_mix)
        
        # Rating panel
        self.rlhf = self._build_rating_panel()
        root.addWidget(self.rlhf)
        self.rlhf.hide()
        
        # Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFocusPolicy(Qt.NoFocus)
        self.console.setStyleSheet(
            "background:#000;color:#00FF00;font-family:Consolas;font-size:11px;"
        )
        root.addWidget(self.console)
    
    def _build_deck(self, layout: QHBoxLayout, deck: str, label_text: str, color: str):
        """Build a single deck display."""
        frame = QFrame()
        frame.setStyleSheet("background:#1E1E1E;border-radius:8px;padding:8px;")
        deck_layout = QVBoxLayout(frame)
        
        lbl_d = QLabel(label_text)
        lbl_n = QLabel("—")
        lbl_d.setFont(QFont("Arial", 13, QFont.Bold))
        lbl_n.setStyleSheet("color:#777;font-size:11px;")
        lbl_n.setWordWrap(True)
        
        plot = pg.PlotWidget()
        plot.setBackground('#1E1E1E')
        plot.setYRange(-1, 1)
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.setFixedHeight(110)
        curve = plot.plot(pen=pg.mkPen(color, width=2))
        
        prog = QProgressBar()
        prog.setStyleSheet(f"QProgressBar::chunk{{background:{color};}}")
        prog.setTextVisible(False)
        prog.setFixedHeight(5)
        
        for w in (lbl_d, lbl_n, plot, prog):
            deck_layout.addWidget(w)
        
        layout.addWidget(frame)
        
        if deck == "A":
            self.lbl_da, self.lbl_na = lbl_d, lbl_n
            self.plot_a, self.curve_a, self.prog_a = plot, curve, prog
        else:
            self.lbl_db, self.lbl_nb = lbl_d, lbl_n
            self.plot_b, self.curve_b, self.prog_b = plot, curve, prog
    
    def _build_info_bar(self) -> QFrame:
        """Build the info bar with BPM, technique, countdown, and controls."""
        inf = QFrame()
        inf.setStyleSheet(
            "background:#1A1A1A;border:1px solid #2A2A2A;border-radius:6px;padding:6px;"
        )
        ir = QHBoxLayout(inf)
        ir.setSpacing(14)
        
        # BPM
        self.lbl_bpm = QLabel("♩ — BPM")
        self.lbl_bpm.setFont(QFont("Consolas", 11, QFont.Bold))
        self.lbl_bpm.setStyleSheet("color:#FFAA00;")
        ir.addWidget(self.lbl_bpm)
        ir.addWidget(self._sep())
        
        # Technique
        self.lbl_tech = QLabel("⬜ —")
        self.lbl_tech.setFont(QFont("Consolas", 11, QFont.Bold))
        self.lbl_tech.setStyleSheet("color:#AA88FF;")
        ir.addWidget(self.lbl_tech)
        ir.addWidget(self._sep())
        
        # Countdown
        self.lbl_cd = QLabel("⏱ Calculating...")
        self.lbl_cd.setFont(QFont("Consolas", 11, QFont.Bold))
        self.lbl_cd.setStyleSheet("color:#00FFCC;")
        ir.addWidget(self.lbl_cd)
        ir.addStretch()
        
        # Play/Pause button
        self.btn_play_pause = QPushButton("⏸ PAUSE")
        self.btn_play_pause.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_play_pause.setStyleSheet(
            "QPushButton{background:#333;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#555;}"
        )
        self.btn_play_pause.clicked.connect(self.toggle_pause)
        ir.addWidget(self.btn_play_pause)
        
        # Skip button
        self.btn_skip = QPushButton("⏩  SKIP TO MIX  (−30s)")
        self.btn_skip.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_skip.setStyleSheet(
            "QPushButton{background:#FF5500;color:#FFF;border-radius:5px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#FF7733;}"
            "QPushButton:disabled{background:#2A2A2A;color:#555;}"
        )
        self.btn_skip.setEnabled(False)
        self.btn_skip.clicked.connect(self.skip_to_mix)
        ir.addWidget(self.btn_skip)
        
        return inf
    
    def _build_rating_panel(self) -> QFrame:
        """Build the RLHF rating panel."""
        rlhf = QFrame()
        rlhf.setStyleSheet(
            "background:#1A1A2E;border:2px solid #00FFCC;border-radius:8px;padding:8px;"
        )
        rl_root = QVBoxLayout(rlhf)
        
        # Rating slider row
        row1 = QHBoxLayout()
        self.lbl_rate = QLabel("🤖 Rate transition (1–10):")
        self.lbl_rate.setFont(QFont("Arial", 11, QFont.Bold))
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setValue(5)
        self.slider.setFocusPolicy(Qt.NoFocus)
        
        self.lbl_sv = QLabel("5")
        self.lbl_sv.setFont(QFont("Consolas", 14, QFont.Bold))
        self.lbl_sv.setStyleSheet("color:#00FFCC;min-width:22px;")
        self.slider.valueChanged.connect(lambda v: self.lbl_sv.setText(str(v)))
        
        self.btn_train = QPushButton("TRAIN MODEL")
        self.btn_train.setStyleSheet(
            "background:#00FFCC;color:#000;font-weight:bold;padding:7px 14px;"
        )
        self.btn_train.clicked.connect(self.submit_rating)
        
        for w in (self.lbl_rate, self.slider, self.lbl_sv, self.btn_train):
            row1.addWidget(w)
        rl_root.addLayout(row1)
        
        # Tags
        tags_label = QLabel("Tag failure modes (optional):")
        tags_label.setFont(QFont("Arial", 9))
        tags_label.setStyleSheet("color:#AAA;")
        rl_root.addWidget(tags_label)
        
        tags_row1, tags_row2 = QHBoxLayout(), QHBoxLayout()
        self._tag_checks = {}
        
        for i, tag in enumerate(TransitionDataset.FAILURE_TAGS):
            cb = QCheckBox(tag.replace('_', ' '))
            cb.setStyleSheet("color:#CCC;font-size:10px;")
            self._tag_checks[tag] = cb
            (tags_row1 if i < 5 else tags_row2).addWidget(cb)
        
        rl_root.addLayout(tags_row1)
        rl_root.addLayout(tags_row2)
        
        return rlhf
    
    def _sep(self) -> QLabel:
        """Create a vertical separator."""
        s = QLabel("│")
        s.setStyleSheet("color:#333;")
        return s
    
    def _append_console(self, text: str):
        """Append text to console (thread-safe)."""
        c = self.console.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText(text)
        
        # Trim old lines
        if self.console.document().blockCount() > 2000:
            c.movePosition(QTextCursor.Start)
            c.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 500)
            c.removeSelectedText()
            c.movePosition(QTextCursor.End)
        
        self.console.setTextCursor(c)
        self.console.ensureCursorVisible()
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.dj.is_paused = not self.dj.is_paused
        
        if self.dj.is_paused:
            self.dj._pause_start_time = time.time()
            self.dj.mixer.pause("A")
            self.dj.mixer.pause("B")
            self.btn_play_pause.setText("▶ RESUME")
            self.btn_play_pause.setStyleSheet(
                "QPushButton{background:#00AA00;color:#FFF;border-radius:5px;"
                "padding:6px 16px;font-weight:bold;}QPushButton:hover{background:#00CC00;}"
            )
        else:
            if self.dj._pause_start_time > 0:
                self.dj._total_pause_time += (time.time() - self.dj._pause_start_time)
                self.dj._pause_start_time = 0.0
            self.dj.mixer.play("A")
            if self.dj.mixer.is_transitioning():
                self.dj.mixer.play("B")
            self.btn_play_pause.setText("⏸ PAUSE")
            self.btn_play_pause.setStyleSheet(
                "QPushButton{background:#333;color:#FFF;border-radius:5px;"
                "padding:6px 16px;font-weight:bold;}QPushButton:hover{background:#555;}"
            )
    
    def skip_to_mix(self):
        """Skip to 30 seconds before mix trigger."""
        mt = getattr(self.dj, 'current_mix_trigger', 0.0)
        if mt > 30:
            self.dj.mixer.seek("A", mt - 30)
            print(f"\n⏩ SKIP → {mt-30:.1f}s")
    
    def keyPressEvent(self, ev):
        """Handle keyboard shortcuts."""
        if ev.key() == Qt.Key_Up:
            self.skip_to_mix()
        elif ev.key() == Qt.Key_Space:
            self.toggle_pause()
        else:
            super().keyPressEvent(ev)
    
    def closeEvent(self, ev):
        """Handle window close."""
        sys.stdout = self._old_stdout
        print("\n🛑 Saving brain and shutting down threads...")
        self.dj.brain.ml.save_brain()
        
        self.dj._analysis_pool.shutdown(wait=False)
        self.dj._learn_pool.shutdown(wait=False)
        time.sleep(2.0)
        
        self.dj.warper.cleanup_temp_files(keep_last=False)
        ev.accept()
    
    def _tick(self):
        """Update GUI at 20 Hz."""
        # Update waveforms
        try:
            buf_a = self.dj.mixer.get_visual_buffer("A")
            buf_b = self.dj.mixer.get_visual_buffer("B")
            
            if buf_a is not None and len(buf_a) > 0:
                self.curve_a.setData(buf_a)
            else:
                self.curve_a.setData([0] * 512)
            
            if buf_b is not None and len(buf_b) > 0:
                self.curve_b.setData(buf_b)
            else:
                self.curve_b.setData([0] * 512)
        except Exception as e:
            if time.time() % 5 < 0.05:
                print(f"⚠️ Visual buffer error: {e}")
        
        # Update progress bars
        pa = self.dj.mixer.get_position("A")
        pb = self.dj.mixer.get_position("B")
        da = max(1.0, float(self.dj.track_a_dur or 1.0))  # ✅ FIX #16: Safe division
        db = max(1.0, float(self.dj.track_b_dur or 1.0))
        
        self.prog_a.setValue(min(int(pa / da * 100), 100))
        self.prog_b.setValue(min(int(pb / db * 100), 100))
        
        # Update track names
        self.lbl_na.setText(self.dj.track_a_name)
        self.lbl_nb.setText(self.dj.track_b_name)
        
        # Update BPM
        if self.dj.master_bpm:
            self.lbl_bpm.setText(f"♩ {self.dj.master_bpm} BPM")
        
        # Update technique
        self.lbl_tech.setText(getattr(self.dj, 'current_technique', '—'))
        
        # Update deck labels and countdown
        is_trans = self.dj.mixer.is_transitioning()
        mt = getattr(self.dj, 'current_mix_trigger', 0.0)
        can_update = time.time() > self._msg_lock_expiry
        
        if is_trans:
            self.lbl_da.setText("DECK A:  🌊 MIXING OUT")
            self.lbl_db.setText("DECK B:  🌊 MIXING IN")
            self.lbl_da.setStyleSheet("color:#FFAA00;")
            self.lbl_db.setStyleSheet("color:#00FFCC;")
            self.prog_mix.setValue(100)
            self.btn_skip.setEnabled(False)
            if can_update:
                self.lbl_cd.setText("🔥 MIXING LIVE")
                self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
        elif pa > 0.1:
            self.lbl_da.setText("DECK A:  ▶ LIVE")
            self.lbl_db.setText("DECK B:  Cued")
            self.lbl_da.setStyleSheet("color:#FF0055;")
            self.lbl_db.setStyleSheet("color:#555;")
        
        if not is_trans and mt > 0.5:
            ttm = mt - pa
            if ttm > 0:
                m, s = int(ttm // 60), int(ttm % 60)
                self.btn_skip.setEnabled(ttm >= 30)
                if mt > 0.0:
                    self.prog_mix.setValue(min(int(max(0, mt - ttm) / mt * 100), 99))
                if can_update:
                    self.lbl_cd.setText(
                        f"⏱ Next Mix: {m}m {s:02d}s" if m else f"⏱ Next Mix: {s}s"
                    )
                    self.lbl_cd.setStyleSheet(
                        "color:#FF0055;" if ttm < 30 else
                        "color:#FFAA00;" if ttm < 90 else
                        "color:#00FFCC;"
                    )
            else:
                self.prog_mix.setValue(100)
                self.btn_skip.setEnabled(False)
                if can_update:
                    self.lbl_cd.setText("🔥 Dropping Now!")
                    self.lbl_cd.setStyleSheet("color:#FF0055;font-weight:bold;")
        elif not is_trans and mt <= 0.5:
            self.prog_mix.setValue(0)
            self.btn_skip.setEnabled(False)
            if can_update:
                self.lbl_cd.setText("⏱ Calculating next mix...")
                self.lbl_cd.setStyleSheet("color:#444;")
    
    def _show_rating(self, mem_key: str, recipe: dict, telemetry):
        """
        Show rating panel (called from worker thread).
        
        ✅ FIX #3: Thread-safe with dedicated lock
        """
        with self._rating_lock:
            tx_id = self.dj._pending_tx_id
            
            # Drop previous rating if not submitted
            if self.pending_rating is not None:
                print(f"⚠️ Dropping previous rating request: {self.pending_rating[2]}")
            
            self.pending_rating = (mem_key, recipe, tx_id, telemetry)
        
        # Update UI (must be on main thread)
        self.slider.setValue(5)
        self.lbl_sv.setText("5")
        
        score = self.dj.dataset.get_overlap_score(tx_id)
        score_str = f"  [quality={score:.2f}]" if score > 0 else ""
        self.lbl_rate.setText(
            f"🤖 Rate [{recipe.get('technique_name','?')}]{score_str} (1–10):"
        )
        
        for cb in self._tag_checks.values():
            cb.setChecked(False)
        
        self.rlhf.show()
    
    def submit_rating(self):
        """Submit user rating and train model."""
        with self._rating_lock:
            if self.pending_rating is None:
                return
            
            mk, rec, tx_id, telemetry = self.pending_rating
            rating = self.slider.value()
            tags = [tag for tag, cb in self._tag_checks.items() if cb.isChecked()]
            
            # Clear pending
            self.pending_rating = None
        
        # Submit to dataset
        if tx_id:
            self.dj._learn_pool.submit(self.dj.dataset.update_rating, tx_id, rating, tags)
        
        # Train ML model
        self.dj._learn_pool.submit(
            self.dj.brain.ml.learn_from_feedback, rating, mk, rec, tags
        )
        
        # Finalize telemetry
        if telemetry:
            self.dj._learn_pool.submit(telemetry.finalize_and_save, rating, tags)
        
        # Feedback message
        tech = rec.get('technique_name', '?')
        if rating >= 8:
            msg = f"✅ Brain updated — {tech} improved"
        elif rating <= 4:
            msg = f"📉 Slight nudge away from {tech}"
        elif 6 <= rating <= 7:
            msg = f"👍 Small blend toward {tech}"
        else:
            msg = f"⚖️  No change for {tech} (mediocre)"
        
        if tags:
            msg += f"  | {len(tags)} tag(s)"
        
        self.lbl_cd.setText(msg)
        self.lbl_cd.setStyleSheet("color:#FFAA00;font-weight:bold;")
        self._msg_lock_expiry = time.time() + 4.0
        
        # Print summary periodically
        rated_count = sum(1 for r in self.dj.dataset.records
                         if r.get('rating') is not None) + 1
        if rated_count % 5 == 0:
            print(f"\n{self.dj.dataset.summary()}\n")
        
        self.rlhf.hide()