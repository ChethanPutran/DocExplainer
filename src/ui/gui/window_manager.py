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

        # Connect the new sidebar signal
        self.main_window.sidebar.question_clicked.connect(self.handle_follow_up)



        # Connect UI to Backend
        # self.main_window.explain_requested.connect(self.on_explain)
        # self.main_window.summarize_requested.connect(self.on_summarize)
        # self.main_window.ask_requested.connect(self.on_ask)

        self.main_window.resize(1200, 800)
    

    def handle_follow_up(self, question: str, section_id):
        """Called when a user clicks a suggested question in the sidebar."""
        # We need the current active document ID. 
        # You can get it from the current tab in the main window.
        current_viewer = self.main_window.tabs.currentWidget()
        if not current_viewer or not current_viewer.doc_id:
            return

        doc_id = int(current_viewer.doc_id)
        
        # Since this is a general follow-up, we might not have a specific page/pos
        # We can pass 0, 0 and let the backend resolve context from the session history
        self.on_follow_up(doc_id=doc_id,question=question,section_id=section_id )

    def on_document_registered(self, path: str) -> int:
        print("Registering the document...")
        doc_id = self.pipeline.register_document(path)
        print("Documnet added!")
        return doc_id
    
    def get_document(self,doc_id):
        return self.pipeline.get_document(doc_id)
    
    def on_explain(self, doc_id: int, text: str,  page:int,
                position:int):
        # 1. Resolve the section_id using backend logic
        section_id = self.pipeline.get_section_id_at(doc_id, page, position)
        response = self.pipeline.explain(doc_id=doc_id, selected_text=text, section_id=section_id)
        self.update_explanation(response,section_id)
        self.main_window.voice_output.set_text(response.explanation)

    def update_explanation(self, explanation, section_id):
        print(explanation, section_id)
        self.main_window.sidebar.update_explanation(explanation, section_id=section_id)

    def on_summarize(self, doc_id: int, text: str, page:int,
                position:int):
        # 1. Resolve the section_id using backend logic
        section_id = self.pipeline.get_section_id_at(doc_id, page, position)
        response = self.pipeline.summarize(doc_id=doc_id, selected_text=text, section_id=section_id)
        self.update_explanation(response,section_id)

    def on_ask(self, doc_id: int, text: str,   page:int,
                position:int):
        # 1. Resolve the section_id using backend logic
        section_id = self.pipeline.get_section_id_at(doc_id, page, position)
        response = self.pipeline.answer_question(doc_id=doc_id, question=text, section_id=section_id)
        self.update_explanation(response,section_id)

    def on_follow_up(self, doc_id: int,question, section_id):
        response = self.pipeline.answer_question(doc_id=doc_id, question=question, section_id=section_id)
        self.update_explanation(response,section_id)

    def launch_gui(self):
        self.main_window.show()
        sys.exit(self.app.exec())


def launch_gui():
    manager = WindowManager()
    manager.launch_gui()
    
if __name__ == "__main__":
    launch_gui()