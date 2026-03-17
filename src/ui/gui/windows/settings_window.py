from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QTabWidget, QWidget, QGroupBox,
                               QComboBox, QSpinBox, QCheckBox, QDoubleSpinBox,
                               QLineEdit, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont


class SettingsWindow(QDialog):
    """Settings dialog"""
    
    def __init__(self, config, theme_manager, parent=None):
        super().__init__(parent)
        self.config = config
        self.theme_manager = theme_manager
        self.settings = QSettings("DocExplainer", "Settings")
        
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup settings dialog UI"""
        layout = QVBoxLayout()
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # General tab
        self.general_tab = self._create_general_tab()
        self.tabs.addTab(self.general_tab, "General")
        
        # Appearance tab
        self.appearance_tab = self._create_appearance_tab()
        self.tabs.addTab(self.appearance_tab, "Appearance")
        
        # Voice tab
        self.voice_tab = self._create_voice_tab()
        self.tabs.addTab(self.voice_tab, "Voice")
        
        # Documents tab
        self.documents_tab = self._create_documents_tab()
        self.tabs.addTab(self.documents_tab, "Documents")
        
        # LLM tab
        self.llm_tab = self._create_llm_tab()
        self.tabs.addTab(self.llm_tab, "AI Model")
        
        # Advanced tab
        self.advanced_tab = self._create_advanced_tab()
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
        layout.addWidget(self.tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_settings)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_settings)
        button_layout.addWidget(apply_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._save_and_close)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_general_tab(self) -> QWidget:
        """Create general settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Language
        lang_group = QGroupBox("Language")
        lang_layout = QVBoxLayout()
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Spanish", "French", "German", "Chinese"])
        lang_layout.addWidget(self.lang_combo)
        
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)
        
        # Startup
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout()
        
        self.open_last_docs = QCheckBox("Open last documents on startup")
        startup_layout.addWidget(self.open_last_docs)
        
        self.check_updates = QCheckBox("Check for updates on startup")
        startup_layout.addWidget(self.check_updates)
        
        startup_group.setLayout(startup_layout)
        layout.addWidget(startup_group)
        
        # Telemetry
        telemetry_group = QGroupBox("Usage Data")
        telemetry_layout = QVBoxLayout()
        
        self.send_stats = QCheckBox("Send anonymous usage statistics")
        telemetry_layout.addWidget(self.send_stats)
        
        telemetry_group.setLayout(telemetry_layout)
        layout.addWidget(telemetry_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _create_appearance_tab(self) -> QWidget:
        """Create appearance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Theme
        theme_group = QGroupBox("Theme")
        theme_layout = QVBoxLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.theme_manager.get_theme_names())
        self.theme_combo.currentTextChanged.connect(self._preview_theme)
        theme_layout.addWidget(self.theme_combo)
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Font
        font_group = QGroupBox("Font")
        font_layout = QVBoxLayout()
        
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("Font Size:"))
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setValue(self.config.font_size)
        font_size_layout.addWidget(self.font_size)
        font_size_layout.addStretch()
        font_layout.addLayout(font_size_layout)
        
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # Layout
        layout_group = QGroupBox("Layout")
        layout_layout = QVBoxLayout()
        
        self.sidebar_visible = QCheckBox("Show sidebar by default")
        self.sidebar_visible.setChecked(self.config.sidebar_visible)
        layout_layout.addWidget(self.sidebar_visible)
        
        layout_group.setLayout(layout_layout)
        layout.addWidget(layout_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _create_voice_tab(self) -> QWidget:
        """Create voice settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Voice input
        input_group = QGroupBox("Voice Input")
        input_layout = QVBoxLayout()
        
        self.voice_enabled = QCheckBox("Enable voice input")
        self.voice_enabled.setChecked(self.config.voice_enabled)
        input_layout.addWidget(self.voice_enabled)
        
        input_device_layout = QHBoxLayout()
        input_device_layout.addWidget(QLabel("Input Device:"))
        self.voice_device = QComboBox()
        self.voice_device.addItems(["Default", "Microphone", "Headset"])
        input_device_layout.addWidget(self.voice_device)
        input_layout.addLayout(input_device_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Voice output
        output_group = QGroupBox("Voice Output")
        output_layout = QVBoxLayout()
        
        self.tts_enabled = QCheckBox("Enable text-to-speech")
        self.tts_enabled.setChecked(self.config.voice_output_enabled)
        output_layout.addWidget(self.tts_enabled)
        
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Speech Rate:"))
        self.tts_rate = QSpinBox()
        self.tts_rate.setRange(50, 300)
        self.tts_rate.setValue(self.config.voice_output_rate)
        rate_layout.addWidget(self.tts_rate)
        output_layout.addLayout(rate_layout)
        
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.tts_volume = QDoubleSpinBox()
        self.tts_volume.setRange(0.1, 1.0)
        self.tts_volume.setSingleStep(0.1)
        self.tts_volume.setValue(self.config.voice_output_volume)
        volume_layout.addWidget(self.tts_volume)
        output_layout.addLayout(volume_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _create_documents_tab(self) -> QWidget:
        """Create document settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Default paths
        paths_group = QGroupBox("Default Paths")
        paths_layout = QVBoxLayout()
        
        docs_path_layout = QHBoxLayout()
        docs_path_layout.addWidget(QLabel("Documents Folder:"))
        self.docs_path = QLineEdit()
        docs_path_layout.addWidget(self.docs_path)
        browse_docs = QPushButton("Browse")
        browse_docs.clicked.connect(self._browse_docs_path)
        docs_path_layout.addWidget(browse_docs)
        paths_layout.addLayout(docs_path_layout)
        
        cache_path_layout = QHBoxLayout()
        cache_path_layout.addWidget(QLabel("Cache Folder:"))
        self.cache_path = QLineEdit()
        cache_path_layout.addWidget(self.cache_path)
        browse_cache = QPushButton("Browse")
        browse_cache.clicked.connect(self._browse_cache_path)
        cache_path_layout.addWidget(browse_cache)
        paths_layout.addLayout(cache_path_layout)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Display
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Default Zoom:"))
        self.default_zoom = QDoubleSpinBox()
        self.default_zoom.setRange(0.5, 3.0)
        self.default_zoom.setSingleStep(0.1)
        self.default_zoom.setValue(self.config.default_zoom)
        zoom_layout.addWidget(self.default_zoom)
        display_layout.addLayout(zoom_layout)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Cache
        cache_group = QGroupBox("Cache")
        cache_layout = QVBoxLayout()
        
        self.cache_enabled = QCheckBox("Enable document caching")
        self.cache_enabled.setChecked(self.config.cache_documents)
        cache_layout.addWidget(self.cache_enabled)
        
        cache_size_layout = QHBoxLayout()
        cache_size_layout.addWidget(QLabel("Max Cache Size (MB):"))
        self.cache_size = QSpinBox()
        self.cache_size.setRange(100, 10000)
        self.cache_size.setValue(self.config.cache_size_mb)
        cache_size_layout.addWidget(self.cache_size)
        cache_layout.addLayout(cache_size_layout)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _create_llm_tab(self) -> QWidget:
        """Create LLM settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("AI Model")
        model_layout = QVBoxLayout()
        
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))
        self.llm_provider = QComboBox()
        self.llm_provider.addItems(["gemini", "openai"])
        provider_layout.addWidget(self.llm_provider)
        model_layout.addLayout(provider_layout)
        
        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(QLabel("Model:"))
        self.llm_model = QComboBox()
        self.llm_model.addItems(["gemini-1.5-flash", "gemini-1.5-pro", "gpt-4", "gpt-3.5-turbo"])
        model_name_layout.addWidget(self.llm_model)
        model_layout.addLayout(model_name_layout)
        
        temperature_layout = QHBoxLayout()
        temperature_layout.addWidget(QLabel("Temperature:"))
        self.llm_temperature = QDoubleSpinBox()
        self.llm_temperature.setRange(0.0, 2.0)
        self.llm_temperature.setSingleStep(0.1)
        self.llm_temperature.setValue(1.0)
        temperature_layout.addWidget(self.llm_temperature)
        model_layout.addLayout(temperature_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # API Keys
        keys_group = QGroupBox("API Keys")
        keys_layout = QVBoxLayout()
        
        gemini_key_layout = QHBoxLayout()
        gemini_key_layout.addWidget(QLabel("Gemini API Key:"))
        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.Password)
        gemini_key_layout.addWidget(self.gemini_key)
        keys_layout.addLayout(gemini_key_layout)
        
        openai_key_layout = QHBoxLayout()
        openai_key_layout.addWidget(QLabel("OpenAI API Key:"))
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.Password)
        openai_key_layout.addWidget(self.openai_key)
        keys_layout.addLayout(openai_key_layout)
        
        keys_group.setLayout(keys_layout)
        layout.addWidget(keys_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Knowledge graph
        kg_group = QGroupBox("Knowledge Graph")
        kg_layout = QVBoxLayout()
        
        self.kg_enabled = QCheckBox("Enable knowledge graph")
        self.kg_enabled.setChecked(True)
        kg_layout.addWidget(self.kg_enabled)
        
        self.kg_auto_build = QCheckBox("Auto-build graph on document load")
        kg_layout.addWidget(self.kg_auto_build)
        
        kg_group.setLayout(kg_layout)
        layout.addWidget(kg_group)
        
        # Memory
        memory_group = QGroupBox("Memory")
        memory_layout = QVBoxLayout()
        
        self.memory_enabled = QCheckBox("Enable user memory")
        self.memory_enabled.setChecked(True)
        memory_layout.addWidget(self.memory_enabled)
        
        self.session_tracking = QCheckBox("Enable session tracking")
        self.session_tracking.setChecked(True)
        memory_layout.addWidget(self.session_tracking)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        # Debug
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout()
        
        self.debug_mode = QCheckBox("Debug mode")
        debug_layout.addWidget(self.debug_mode)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        debug_layout.addWidget(self.log_level)
        
        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _load_settings(self):
        """Load settings from QSettings"""
        # General
        self.lang_combo.setCurrentText(self.settings.value("language", "English"))
        self.open_last_docs.setChecked(self.settings.value("open_last_docs", False, type=bool))
        self.check_updates.setChecked(self.settings.value("check_updates", True, type=bool))
        self.send_stats.setChecked(self.settings.value("send_stats", False, type=bool))
        
        # Appearance
        self.theme_combo.setCurrentText(self.settings.value("theme", "light"))
        self.font_size.setValue(self.settings.value("font_size", 10, type=int))
        self.sidebar_visible.setChecked(self.settings.value("sidebar_visible", True, type=bool))
        
        # Voice
        self.voice_enabled.setChecked(self.settings.value("voice_enabled", True, type=bool))
        self.voice_device.setCurrentText(self.settings.value("voice_device", "Default"))
        self.tts_enabled.setChecked(self.settings.value("tts_enabled", True, type=bool))
        self.tts_rate.setValue(self.settings.value("tts_rate", 150, type=int))
        self.tts_volume.setValue(self.settings.value("tts_volume", 0.9, type=float))
        
        # Documents
        self.docs_path.setText(self.settings.value("docs_path", ""))
        self.cache_path.setText(self.settings.value("cache_path", "cache/"))
        self.default_zoom.setValue(self.settings.value("default_zoom", 1.0, type=float))
        self.cache_enabled.setChecked(self.settings.value("cache_enabled", True, type=bool))
        self.cache_size.setValue(self.settings.value("cache_size", 500, type=int))
        
        # LLM
        self.llm_provider.setCurrentText(self.settings.value("llm_provider", "gemini"))
        self.llm_model.setCurrentText(self.settings.value("llm_model", "gemini-1.5-flash"))
        self.llm_temperature.setValue(self.settings.value("llm_temperature", 1.0, type=float))
        self.gemini_key.setText(self.settings.value("gemini_key", ""))
        self.openai_key.setText(self.settings.value("openai_key", ""))
        
        # Advanced
        self.kg_enabled.setChecked(self.settings.value("kg_enabled", True, type=bool))
        self.kg_auto_build.setChecked(self.settings.value("kg_auto_build", True, type=bool))
        self.memory_enabled.setChecked(self.settings.value("memory_enabled", True, type=bool))
        self.session_tracking.setChecked(self.settings.value("session_tracking", True, type=bool))
        self.debug_mode.setChecked(self.settings.value("debug_mode", False, type=bool))
        self.log_level.setCurrentText(self.settings.value("log_level", "INFO"))
    
    def _save_settings(self):
        """Save settings to QSettings"""
        # General
        self.settings.setValue("language", self.lang_combo.currentText())
        self.settings.setValue("open_last_docs", self.open_last_docs.isChecked())
        self.settings.setValue("check_updates", self.check_updates.isChecked())
        self.settings.setValue("send_stats", self.send_stats.isChecked())
        
        # Appearance
        self.settings.setValue("theme", self.theme_combo.currentText())
        self.settings.setValue("font_size", self.font_size.value())
        self.settings.setValue("sidebar_visible", self.sidebar_visible.isChecked())
        
        # Voice
        self.settings.setValue("voice_enabled", self.voice_enabled.isChecked())
        self.settings.setValue("voice_device", self.voice_device.currentText())
        self.settings.setValue("tts_enabled", self.tts_enabled.isChecked())
        self.settings.setValue("tts_rate", self.tts_rate.value())
        self.settings.setValue("tts_volume", self.tts_volume.value())
        
        # Documents
        self.settings.setValue("docs_path", self.docs_path.text())
        self.settings.setValue("cache_path", self.cache_path.text())
        self.settings.setValue("default_zoom", self.default_zoom.value())
        self.settings.setValue("cache_enabled", self.cache_enabled.isChecked())
        self.settings.setValue("cache_size", self.cache_size.value())
        
        # LLM
        self.settings.setValue("llm_provider", self.llm_provider.currentText())
        self.settings.setValue("llm_model", self.llm_model.currentText())
        self.settings.setValue("llm_temperature", self.llm_temperature.value())
        self.settings.setValue("gemini_key", self.gemini_key.text())
        self.settings.setValue("openai_key", self.openai_key.text())
        
        # Advanced
        self.settings.setValue("kg_enabled", self.kg_enabled.isChecked())
        self.settings.setValue("kg_auto_build", self.kg_auto_build.isChecked())
        self.settings.setValue("memory_enabled", self.memory_enabled.isChecked())
        self.settings.setValue("session_tracking", self.session_tracking.isChecked())
        self.settings.setValue("debug_mode", self.debug_mode.isChecked())
        self.settings.setValue("log_level", self.log_level.currentText())
    
    def _preview_theme(self, theme_name: str):
        """Preview theme without saving"""
        self.theme_manager.set_theme(theme_name)
    
    def _apply_settings(self):
        """Apply settings without closing"""
        self._save_settings()
        
        # Update config
        self.config.theme = self.theme_combo.currentText()
        self.config.font_size = self.font_size.value()
        self.config.sidebar_visible = self.sidebar_visible.isChecked()
        self.config.voice_enabled = self.voice_enabled.isChecked()
        self.config.voice_output_enabled = self.tts_enabled.isChecked()
        self.config.voice_output_rate = self.tts_rate.value()
        self.config.voice_output_volume = self.tts_volume.value()
        self.config.default_zoom = self.default_zoom.value()
        self.config.cache_documents = self.cache_enabled.isChecked()
        self.config.cache_size_mb = self.cache_size.value()
        
        QMessageBox.information(self, "Settings", "Settings applied successfully")
    
    def _save_and_close(self):
        """Save settings and close"""
        self._apply_settings()
        self.accept()
    
    def _reset_settings(self):
        """Reset settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.settings.clear()
            self._load_settings()
            QMessageBox.information(self, "Settings", "Settings reset to defaults")
    
    def _browse_docs_path(self):
        """Browse for documents folder"""
        path = QFileDialog.getExistingDirectory(self, "Select Documents Folder")
        if path:
            self.docs_path.setText(path)
    
    def _browse_cache_path(self):
        """Browse for cache folder"""
        path = QFileDialog.getExistingDirectory(self, "Select Cache Folder")
        if path:
            self.cache_path.setText(path)