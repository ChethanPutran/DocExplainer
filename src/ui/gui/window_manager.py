import sys
from PySide6.QtWidgets import (
    QApplication
)
from src.orchestrator.pipeline import DocExplainerPipeline
from src.ui.gui.main_window import MainWindow

class WindowManager:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.pipeline = DocExplainerPipeline()
        self.main_window = MainWindow(self)
        self.main_window.resize(1200, 800)

        # Connect UI to Backend
        # self.main_window.explain_requested.connect(self.on_explain)
        # self.main_window.summarize_requested.connect(self.on_summarize)
        # self.main_window.ask_requested.connect(self.on_ask)

        self.main_window.resize(1200, 800)
    
    def on_document_registered(self, path: str) -> int:
        doc_id = self.pipeline.register_document(path)
        return doc_id
    
    def on_explain(self, doc_id: int, text: str, section_id: int = 0):
        response = self.pipeline.explain(doc_id=doc_id, selected_text=text, section_id=section_id)
        self.main_window.sidebar.update_explanation(response)
        self.main_window.voice_output.set_text(response.explanation)

    def on_summarize(self, doc_id: int, text: str, section_id: int = 0):
        response = self.pipeline.summarize(doc_id=doc_id, selected_text=text, section_id=section_id)
        self.main_window.sidebar.update_explanation(response.explanation)

    def on_ask(self, doc_id: int, text: str, section_id: int=0):
        response = self.pipeline.answer_question(doc_id=doc_id, question=text, section_id=section_id)
        self.main_window.sidebar.update_explanation(response.explanation)

    def launch_gui(self):
        self.main_window.show()
        sys.exit(self.app.exec())


def launch_gui():
    manager = WindowManager()
    manager.launch_gui()
    
if __name__ == "__main__":
    launch_gui()