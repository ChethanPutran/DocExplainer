"""Settings dialog. It edits the UI's own settings and writes them to ``ui_config.yaml``: the one source (with
the packaged defaults and ``DOC_EXPLAINER_UI_*`` environment variables) that is read at start-up. The language
model and the knowledge graph belong to the server (``llm_config.yaml`` / ``backend_config.yaml``)."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from doc_explainer_ui.config import UIConfig


def _page(*rows) -> QWidget:
    page, form = QWidget(), QFormLayout()
    page.setLayout(form)
    for label, widget in rows:
        form.addRow(label, widget)
    return page


def _spin(lo, hi, value, cls=QSpinBox, **kw):
    w = cls()
    w.setRange(lo, hi)
    for k, v in kw.items():
        getattr(w, k)(v)
    w.setValue(value)
    return w


class SettingsWindow(QDialog):
    def __init__(self, config: UIConfig, theme_manager, parent=None):
        super().__init__(parent)
        self.config, self.theme_manager = config, theme_manager
        self.setWindowTitle("Settings")
        self.setMinimumSize(520, 420)
        c = config

        self.theme = QComboBox()
        self.theme.addItems(sorted(set(theme_manager.get_theme_names()) | {c.theme.name}))
        self.theme.setCurrentText(c.theme.name)
        self.font_size = _spin(6, 32, c.theme.font_size)
        self.sidebar_visible = QCheckBox("Show the tutor sidebar at start-up")
        self.sidebar_visible.setChecked(c.sidebar.visible)
        self.default_zoom = _spin(0.25, 5.0, c.documents.default_zoom, QDoubleSpinBox, setSingleStep=0.1)
        self.max_recent = _spin(0, 100, c.documents.max_recent_files)
        self.open_last = QCheckBox("Reopen the last documents on start-up")
        self.open_last.setChecked(c.startup.open_last_docs)

        self.voice_enabled = QCheckBox("Enable voice input")
        self.voice_enabled.setChecked(c.voice.enabled)
        self.whisper = QComboBox()
        self.whisper.addItems(sorted({"tiny", "base", "small", "medium", c.voice.whisper_model}))
        self.whisper.setCurrentText(c.voice.whisper_model)
        self.preload = QCheckBox("Load the speech model at start-up")
        self.preload.setChecked(c.voice.preload_model)
        self.idle_unload = _spin(0, 86400, int(c.voice.model_idle_unload_s))
        self.idle_unload.setSuffix(" s  (0 = keep it)")
        self.tts_enabled = QCheckBox("Read answers aloud")
        self.tts_enabled.setChecked(c.voice.output_enabled)
        self.tts_rate = _spin(50, 400, c.voice.output_rate)
        self.tts_volume = _spin(0.0, 1.0, c.voice.output_volume, QDoubleSpinBox, setSingleStep=0.1)

        self.graph_mode = QComboBox()
        for value, text in (("always", "Always build it"), ("ask", "Ask me for each new file"), ("never", "Never build it")):
            self.graph_mode.addItem(text, value)
        self.graph_mode.setCurrentIndex(max(0, self.graph_mode.findData(c.knowledge.build_graph)))
        note = QLabel("The knowledge graph maps the concepts in a document. It powers explanations pitched to what you know,\n"
                      "quizzes, and tracking of what you have learned. Without it you can still read the document and ask\n"
                      "questions. Building takes minutes and uses the language model.\n\n"
                      "A single file can be changed too: choose \"Ask\" to decide per file when you open it, or use\n"
                      "Library > Build knowledge graph to add one later. (The server can also switch graphs off for\n"
                      "everyone: knowledge_graph.enabled in backend_config.yaml.)")
        note.setWordWrap(True)

        self.server_url = QLineEdit(c.server.url)
        self.protocol = QComboBox()
        self.protocol.addItems(["rest", "rpc"])
        self.protocol.setCurrentText(c.server.protocol)
        self.timeout = _spin(1, 3600, int(c.server.timeout_s))
        self.user_id = QLineEdit(c.server.user_id)
        self.user_id.setPlaceholderText("empty = the server's default user")

        tabs = QTabWidget()
        tabs.addTab(_page(("Theme", self.theme), ("Font size", self.font_size), ("", self.sidebar_visible),
                          ("Default zoom", self.default_zoom), ("Recent files", self.max_recent), ("", self.open_last)), "General")
        tabs.addTab(_page(("", self.voice_enabled), ("Speech model", self.whisper), ("", self.preload),
                          ("Unload when idle", self.idle_unload), ("", self.tts_enabled),
                          ("Speech rate", self.tts_rate), ("Volume", self.tts_volume)), "Voice")
        tabs.addTab(_page(("Knowledge graph", self.graph_mode), ("", note)), "Learning")
        tabs.addTab(_page(("Server URL", self.server_url), ("Protocol", self.protocol), ("Timeout (s)", self.timeout),
                          ("User id", self.user_id),
                          ("", QLabel("The language model and knowledge graph are set on the server\n"
                                      "(llm_config.yaml, backend_config.yaml, or DOC_EXPLAINER_* variables).\n"
                                      "Server changes apply after restarting the UI."))), "Server")
        buttons = QHBoxLayout()
        buttons.addStretch()
        for text, slot in (("Reset to defaults", self._reset), ("Cancel", self.reject), ("Save", self._save)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        root = QVBoxLayout(self)
        root.addWidget(tabs)
        root.addLayout(buttons)
        self.theme.currentTextChanged.connect(self.theme_manager.set_theme)       # live preview

    def _collect(self) -> None:
        c = self.config
        c.theme.name, c.theme.font_size = self.theme.currentText(), self.font_size.value()
        c.sidebar.visible = self.sidebar_visible.isChecked()
        c.documents.default_zoom, c.documents.max_recent_files = self.default_zoom.value(), self.max_recent.value()
        c.startup.open_last_docs = self.open_last.isChecked()
        c.voice.enabled, c.voice.whisper_model = self.voice_enabled.isChecked(), self.whisper.currentText()
        c.voice.preload_model, c.voice.model_idle_unload_s = self.preload.isChecked(), float(self.idle_unload.value())
        c.voice.output_enabled, c.voice.output_rate = self.tts_enabled.isChecked(), self.tts_rate.value()
        c.voice.output_volume = self.tts_volume.value()
        c.knowledge.build_graph = self.graph_mode.currentData()
        c.server.url, c.server.protocol = self.server_url.text().strip(), self.protocol.currentText()
        c.server.timeout_s, c.server.user_id = float(self.timeout.value()), self.user_id.text().strip()

    def _save(self) -> None:
        self._collect()
        try:
            self.config.save()
        except OSError as e:
            QMessageBox.warning(self, "Settings", f"Could not save the settings: {e}")
            return
        self.accept()

    def _reset(self) -> None:
        if QMessageBox.question(self, "Reset settings", "Reset all UI settings to their defaults?") != QMessageBox.StandardButton.Yes:
            return
        fresh = UIConfig()
        for name in UIConfig.__dataclass_fields__:
            setattr(self.config, name, getattr(fresh, name))
        try:
            self.config.save()
        except OSError:
            pass
        self.accept()
