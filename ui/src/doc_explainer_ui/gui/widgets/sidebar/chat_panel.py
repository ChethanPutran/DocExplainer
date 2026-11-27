"""The Explanation tab as a conversation.

Messages stream in as the tutor writes them, with a status line ("Understanding your request", "Writing the
reply", ...). The reader can type (or speak) a question at the bottom, pick up an earlier session of the document,
keep notes and have the conversation summarised. With the speaker on, the reply is read aloud and its text appears
sentence by sentence AS each sentence is spoken, so the screen is a live transcript of the voice.

This is a view: it holds no service logic. The window manager feeds it (set_sessions, begin_reply, append_delta,
finish_reply, ...) and listens to its signals.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{1,}")
AUTOSEND_PAUSE_MS = 1500            # silence after the last spoken phrase before an auto-send


def split_sentences(text: str):
    """(complete sentences each with its trailing whitespace, the unfinished rest)."""
    parts, last = [], 0
    for m in _SENTENCE_END.finditer(text):
        parts.append(text[last:m.end()])
        last = m.end()
    return parts, text[last:]


def _label(text: str = "", style: str = "", markdown: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if markdown:
        label.setTextFormat(Qt.TextFormat.MarkdownText)
    if style:
        label.setStyleSheet(style)
    return label


class MessageBubble(QFrame):
    """One message. The assistant's grows while it streams and gets follow-up questions underneath when it is done."""
    follow_up_clicked = Signal(str)
    save_requested = Signal(str)

    def __init__(self, role: str, text: str = ""):
        super().__init__()
        self.role, self._text = role, ""
        user = role == "user"
        self.setStyleSheet(f"MessageBubble {{ background: {'#e8f0fe' if user else '#f4f4f4'}; border-radius: 8px; }}")
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(10, 6, 10, 6)
        who = _label("You" if user else "Tutor", "color: #666; font-size: 8pt; font-weight: bold;")
        self.body = _label("", markdown=not user)
        self._box.addWidget(who)
        self._box.addWidget(self.body)
        self.extras = QVBoxLayout()
        self._box.addLayout(self.extras)
        self.set_text(text)

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self.body.setText(text if text else ("" if self.role == "user" else "…"))

    def finish(self, follow_ups: List[str], known: List[str], learning: List[str], saveable: bool = True) -> None:
        """Everything that comes after the text: what the reader knows/learns, follow-up questions, save-to-notes."""
        self._clear_extras()
        notes = []
        if known:
            notes.append(f"Known: {', '.join(known)}")
        if learning:
            notes.append(f"Learning: {', '.join(learning)}")
        if notes:
            self.extras.addWidget(_label("   ".join(notes), "color: #888; font-size: 8pt;"))
        for q in follow_ups:
            b = QPushButton(q)
            b.setStyleSheet("QPushButton { text-align: left; padding: 5px; border: 1px solid #ccd; border-radius: 6px; background: white; }"
                            "QPushButton:hover { background: #eef; }")
            b.clicked.connect(lambda _=False, question=q: self.follow_up_clicked.emit(question))
            self.extras.addWidget(b)
        if saveable and self._text.strip():
            save = QToolButton()
            save.setText("📝 Save to notes")
            save.setStyleSheet("QToolButton { border: none; color: #557; font-size: 8pt; }")
            save.clicked.connect(lambda: self.save_requested.emit(self._text))
            self.extras.addWidget(save, alignment=Qt.AlignmentFlag.AlignLeft)

    def _clear_extras(self) -> None:
        while self.extras.count():
            item = self.extras.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def follow_up_buttons(self) -> List[QPushButton]:
        return [b for b in self.findChildren(QPushButton)]


class _InputBox(QPlainTextEdit):
    """Enter sends, Shift+Enter makes a new line."""
    submitted = Signal()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class ChatPanel(QWidget):
    send_requested = Signal(str)
    stop_requested = Signal()
    new_session_requested = Signal()
    session_selected = Signal(str)
    rename_session_requested = Signal(str, str)
    delete_session_requested = Signal(str)
    note_add_requested = Signal(str)
    note_edit_requested = Signal(str, str)
    note_delete_requested = Signal(str)
    summarize_requested = Signal()
    mic_toggled = Signal(bool)
    speaker_toggled = Signal(bool)

    def __init__(self, parent=None, speaker=None):
        super().__init__(parent)
        self.speaker = speaker                     # SentenceSpeaker-like: enqueue/stop/speaking + sentence_started/idle signals
        self.session_id = ""
        self._sessions: List[dict] = []
        self._busy = False
        self._reply: Optional[MessageBubble] = None
        self._buffer = ""                          # received, not yet turned into sentences
        self._awaiting: List[str] = []             # sentences handed to the speaker, not yet revealed
        self._shown = ""
        self._full = ""
        self._done_payload: Optional[dict] = None
        self._speaking_reply = False
        self._autosend_timer = QTimer(self)
        self._autosend_timer.setSingleShot(True)
        self._autosend_timer.timeout.connect(self._send)
        self._build()
        if speaker is not None:
            speaker.sentence_started.connect(self._on_sentence_started)
            speaker.idle.connect(self._on_speaker_idle)
            if hasattr(speaker, "unavailable"):
                speaker.unavailable.connect(self._on_speaker_unavailable)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.session_combo = QComboBox()
        self.session_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.session_combo.setMinimumContentsLength(12)
        self.session_combo.setToolTip("Conversations about this document")
        self.new_button = QToolButton(); self.new_button.setText("＋"); self.new_button.setToolTip("Start a new session")
        self.rename_button = QToolButton(); self.rename_button.setText("✎"); self.rename_button.setToolTip("Rename this session")
        self.delete_button = QToolButton(); self.delete_button.setText("🗑"); self.delete_button.setToolTip("Delete this session")
        self.notes_button = QToolButton(); self.notes_button.setText("📝 Notes"); self.notes_button.setCheckable(True)
        self.notes_button.setToolTip("Notes and summaries of this session")
        bar.addWidget(self.session_combo, 1)
        for b in (self.new_button, self.rename_button, self.delete_button, self.notes_button):
            bar.addWidget(b)
        root.addLayout(bar)

        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.pages.addWidget(self._conversation_page())
        self.pages.addWidget(self._notes_page())

        self.status_label = _label("", "color: #557; font-style: italic; font-size: 9pt;")
        root.addWidget(self.status_label)

        row = QHBoxLayout()
        self.input = _InputBox()
        self.input.setPlaceholderText("Ask anything about this document...  (Enter to send, Shift+Enter for a new line)")
        self.input.setFixedHeight(64)
        side = QVBoxLayout()
        self.mic_button = QToolButton(); self.mic_button.setText("🎤"); self.mic_button.setCheckable(True)
        self.mic_button.setToolTip("Speak your question: the words appear here so you can edit them, then send")
        self.speaker_button = QToolButton(); self.speaker_button.setText("🔊"); self.speaker_button.setCheckable(True)
        self.speaker_button.setToolTip("Read the answer aloud (the text appears as it is spoken)")
        self.send_button = QPushButton("Send")
        side.addWidget(self.mic_button); side.addWidget(self.speaker_button)
        row.addWidget(self.input, 1)
        row.addLayout(side)
        row.addWidget(self.send_button)
        root.addLayout(row)
        self.autosend = QCheckBox("Auto-send when I pause speaking")
        self.autosend.setToolTip("After you stop talking for a moment, the question is sent without clicking Send")
        root.addWidget(self.autosend)

        self.session_combo.activated.connect(lambda i: self._select(i))
        self.new_button.clicked.connect(self.new_session_requested.emit)
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.notes_button.toggled.connect(lambda on: self.pages.setCurrentIndex(1 if on else 0))
        self.send_button.clicked.connect(self._on_send_clicked)
        self.input.submitted.connect(self._on_send_clicked)
        self.mic_button.toggled.connect(self.mic_toggled.emit)
        self.speaker_button.toggled.connect(self._on_speaker_toggled)
        self._refresh_enabled()

    def _conversation_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        self.thread = QVBoxLayout(holder)
        self.thread.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.empty_label = _label("Ask a question below, or select text in the document and right-click.", "color: #999; padding: 16px;")
        self.thread.addWidget(self.empty_label)
        scroll.setWidget(holder)
        self.scroll = scroll
        return scroll

    def _notes_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        top = QHBoxLayout()
        self.summarize_button = QPushButton("✨ Summarize this session")
        top.addWidget(self.summarize_button)
        box.addLayout(top)
        self.note_input = QPlainTextEdit()
        self.note_input.setPlaceholderText("Write a note about this document...")
        self.note_input.setFixedHeight(60)
        self.add_note_button = QPushButton("Add note")
        box.addWidget(self.note_input)
        box.addWidget(self.add_note_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        self.notes_layout = QVBoxLayout(holder)
        self.notes_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(holder)
        box.addWidget(scroll, 1)
        self.summarize_button.clicked.connect(self.summarize_requested.emit)
        self.add_note_button.clicked.connect(self._add_note)
        return page

    # ------------------------------------------------------------------ sessions
    def set_sessions(self, sessions: List[dict], current_id: str) -> None:
        self._sessions = sessions
        self.session_id = current_id
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for s in sessions:
            self.session_combo.addItem(f"{s['title']}  ({s.get('messages', 0) // 2})", s["id"])
        index = next((i for i, s in enumerate(sessions) if s["id"] == current_id), -1)
        self.session_combo.setCurrentIndex(index)
        self.session_combo.blockSignals(False)
        self._refresh_enabled()

    def _select(self, index: int) -> None:
        sid = self.session_combo.itemData(index)
        if sid and sid != self.session_id:
            self.session_selected.emit(sid)

    def _rename(self) -> None:
        if not self.session_id:
            return
        current = next((s["title"] for s in self._sessions if s["id"] == self.session_id), "")
        title, ok = QInputDialog.getText(self, "Rename session", "Name:", text=current)
        if ok and title.strip():
            self.rename_session_requested.emit(self.session_id, title.strip())

    def _delete(self) -> None:
        if not self.session_id:
            return
        if QMessageBox.question(self, "Delete session", "Delete this conversation and its notes?") == QMessageBox.StandardButton.Yes:
            self.delete_session_requested.emit(self.session_id)

    # ------------------------------------------------------------------ the conversation
    def show_history(self, messages: List[dict], notes: List[dict]) -> None:
        """Replace what is on screen with a stored session."""
        self._abort_reply()
        self._clear_thread()
        for m in messages:
            bubble = self._add_bubble(m["role"], m["text"])
            if m["role"] == "assistant":
                meta = m.get("meta") or {}
                bubble.finish(meta.get("follow_ups", []), [], [])
        self.set_notes(notes)
        self._scroll_down()

    def clear(self) -> None:
        self._abort_reply()
        self._clear_thread()
        self.set_notes([])
        self.set_sessions([], "")
        self.set_status("")

    def _clear_thread(self) -> None:
        while self.thread.count():
            item = self.thread.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty_label = _label("Ask a question below, or select text in the document and right-click.", "color: #999; padding: 16px;")
        self.thread.addWidget(self.empty_label)
        self._reply = None

    def _add_bubble(self, role: str, text: str) -> MessageBubble:
        if self.empty_label is not None and self.empty_label.parent() is not None:
            self.thread.removeWidget(self.empty_label)
            self.empty_label.deleteLater()
            self.empty_label = None
        bubble = MessageBubble(role, text)
        bubble.follow_up_clicked.connect(self._send_text)
        bubble.save_requested.connect(self.note_add_requested.emit)
        self.thread.addWidget(bubble)
        self._scroll_down()
        return bubble

    def messages(self) -> List[MessageBubble]:
        return [self.thread.itemAt(i).widget() for i in range(self.thread.count())
                if isinstance(self.thread.itemAt(i).widget(), MessageBubble)]

    def _scroll_down(self) -> None:
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def add_user_message(self, text: str) -> None:
        self._add_bubble("user", text)

    # ------------------------------------------------------------------ a reply, as it is written
    @property
    def speaking_enabled(self) -> bool:
        return self.speaker_button.isChecked() and self.speaker is not None

    def begin_reply(self) -> None:
        self._abort_reply()
        self._reply = self._add_bubble("assistant", "")
        self._buffer, self._awaiting, self._shown, self._full = "", [], "", ""
        self._done_payload = None
        self._speaking_reply = self.speaking_enabled
        self.set_busy(True)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def append_delta(self, text: str) -> None:
        if self._reply is None:
            return
        self._full += text
        if not self._speaking_reply:
            self._shown += text
            self._reply.set_text(self._shown)
            self._scroll_down()
            return
        self._buffer += text
        sentences, self._buffer = split_sentences(self._buffer)
        for s in sentences:
            self._speak(s)

    def _speak(self, sentence: str) -> None:
        if not sentence.strip():
            self._shown += sentence                  # whitespace only: nothing to say, nothing to wait for
            return
        self._awaiting.append(sentence)
        self.speaker.enqueue(sentence)

    def finish_reply(self, payload: dict) -> None:
        """The complete answer arrived. What remains to be spoken is spoken; then the follow-ups appear."""
        if self._reply is None:
            return
        self._done_payload = payload
        if self._speaking_reply:
            if self._buffer:
                tail, self._buffer = self._buffer, ""
                self._speak(tail)
            if self._awaiting:
                self.set_status("Speaking...")
                return
        self._finalize()

    def _finalize(self) -> None:
        reply, payload = self._reply, self._done_payload
        if reply is None or payload is None:
            return
        text = payload.get("text") or self._full
        reply.set_text(text)                           # the stored, complete text replaces the progressive one
        exp = payload.get("explanation") or {}
        reply.finish(exp.get("follow_up_questions", []), exp.get("known_concepts_used", []), exp.get("unknown_concepts_explained", []))
        self._reply, self._done_payload, self._speaking_reply = None, None, False
        self.set_status("")
        self.set_busy(False)
        self._scroll_down()

    def stop_reply(self) -> None:
        """The reader pressed Stop: keep what is on screen, stop the voice, and be ready for the next question."""
        reply = self._reply
        if reply is not None:
            text = self._shown or self._full
            reply.set_text((text + "\n\n" if text else "") + "_(stopped)_")
        if self.speaker is not None and self._speaking_reply:
            self.speaker.stop()
        self._reply, self._buffer, self._awaiting, self._done_payload, self._speaking_reply = None, "", [], None, False
        self.set_status("")
        self.set_busy(False)

    def fail_reply(self, message: str) -> None:
        reply = self._reply
        if reply is not None:
            shown = self._shown or ""
            reply.set_text((shown + "\n\n" if shown else "") + f"⚠ {message}")
        self._abort_reply()
        self.set_status("")
        self.set_busy(False)

    def _abort_reply(self) -> None:
        if self.speaker is not None and self._speaking_reply:
            self.speaker.stop()
        self._reply = None
        self._buffer, self._awaiting, self._done_payload, self._speaking_reply = "", [], None, False

    # ---- speech keeps the text in step
    def _on_sentence_started(self, sentence: str) -> None:
        if self._reply is None or not self._speaking_reply or not self._awaiting:
            return
        shown = self._awaiting.pop(0)
        self._shown += shown
        self._reply.set_text(self._shown)
        self._scroll_down()

    def _on_speaker_idle(self) -> None:
        if self._speaking_reply and self._done_payload is not None and not self._awaiting:
            self._finalize()

    def _on_speaker_unavailable(self, why: str) -> None:
        """No voice: show the rest at once, and say so."""
        self.speaker_button.blockSignals(True)
        self.speaker_button.setChecked(False)
        self.speaker_button.blockSignals(False)
        self.speaker_button.setToolTip(f"Speech is not available on this computer ({why})")
        self.speaker_button.setEnabled(False)
        self._reveal_everything()

    def _on_speaker_toggled(self, on: bool) -> None:
        self.speaker_toggled.emit(on)
        if not on:
            if self.speaker is not None:
                self.speaker.stop()
            self._reveal_everything()

    def _reveal_everything(self) -> None:
        """Speaker turned off (or broke) mid-reply: stop reading and show what is waiting."""
        if self._reply is None or not self._speaking_reply:
            return
        self._speaking_reply = False
        self._shown += "".join(self._awaiting) + self._buffer
        self._awaiting, self._buffer = [], ""
        self._reply.set_text(self._shown)
        if self._done_payload is not None:
            self._finalize()

    # ------------------------------------------------------------------ input
    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_button.setText("Stop" if busy else "Send")
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self.session_combo.setEnabled(not self._busy)
        self.new_button.setEnabled(not self._busy)
        self.delete_button.setEnabled(bool(self.session_id) and not self._busy)
        self.rename_button.setEnabled(bool(self.session_id) and not self._busy)
        self.speaker_button.setVisible(self.speaker is not None)

    def set_enabled_for_document(self, ready: bool) -> None:
        self.input.setEnabled(ready)
        self.send_button.setEnabled(ready)
        self.mic_button.setEnabled(ready)

    def _on_send_clicked(self) -> None:
        if self._busy:
            self.stop_requested.emit()
            return
        self._send()

    def _send(self) -> None:
        self._autosend_timer.stop()
        if self._busy:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.send_requested.emit(text)

    def _send_text(self, text: str) -> None:
        if not self._busy:
            self.send_requested.emit(text)

    # ---- the microphone feeds the input box
    def add_transcription(self, text: str) -> None:
        """Words from the microphone are put in the input box (never sent behind the reader's back, unless auto-send is on)."""
        if self.speaker is not None and self.speaker.speaking:
            return                                      # the tutor's own voice must not become a question
        text = text.strip()
        if not text:
            return
        current = self.input.toPlainText().rstrip()
        self.input.setPlainText((current + " " + text).strip())
        self.input.moveCursor(self.input.textCursor().MoveOperation.End)
        if self.autosend.isChecked():
            self._autosend_timer.start(AUTOSEND_PAUSE_MS)      # sends once no further phrase arrives

    def set_listening(self, on: bool) -> None:
        self.mic_button.blockSignals(True)
        self.mic_button.setChecked(on)
        self.mic_button.blockSignals(False)
        self.mic_button.setText("🔴" if on else "🎤")

    # ------------------------------------------------------------------ notes
    def set_notes(self, notes: List[dict]) -> None:
        while self.notes_layout.count():
            item = self.notes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not notes:
            self.notes_layout.addWidget(_label("No notes yet. Write one above, save a reply with “Save to notes”, "
                                               "or summarize the session.", "color: #999; padding: 12px;"))
        for n in notes:
            self.notes_layout.addWidget(self._note_card(n))
        self.notes_button.setText(f"📝 Notes ({len(notes)})" if notes else "📝 Notes")

    def _note_card(self, n: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet("QFrame { background: #fffbe6; border-radius: 6px; }" if n["kind"] == "note"
                           else "QFrame { background: #e9f7ef; border-radius: 6px; }")
        box = QVBoxLayout(card)
        box.addWidget(_label("Summary" if n["kind"] == "summary" else "Note", "color: #777; font-size: 8pt; font-weight: bold;"))
        box.addWidget(_label(n["text"], markdown=True))
        row = QHBoxLayout()
        edit, delete = QToolButton(), QToolButton()
        edit.setText("Edit"); delete.setText("Delete")
        edit.clicked.connect(lambda _=False, note=n: self._edit_note(note))
        delete.clicked.connect(lambda _=False, nid=n["id"]: self.note_delete_requested.emit(nid))
        row.addStretch(); row.addWidget(edit); row.addWidget(delete)
        box.addLayout(row)
        return card

    def _edit_note(self, n: dict) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Edit note", "Note:", n["text"])
        if ok and text.strip():
            self.note_edit_requested.emit(n["id"], text.strip())

    def _add_note(self) -> None:
        text = self.note_input.toPlainText().strip()
        if text:
            self.note_input.clear()
            self.note_add_requested.emit(text)

    def note_cards(self) -> List[QFrame]:
        return [self.notes_layout.itemAt(i).widget() for i in range(self.notes_layout.count())
                if isinstance(self.notes_layout.itemAt(i).widget(), QFrame)]
