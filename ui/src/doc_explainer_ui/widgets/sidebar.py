from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser
from PySide6.QtCore import Signal, QUrl, QUrlQuery

class Sidebar(QWidget):
    # Signal sends (question_text, section_id)
    question_clicked = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.header_label = QLabel("<b>AI Tutor Insights</b>")
        self.header_label.setStyleSheet("font-size: 16px; color: #333; margin-bottom: 5px;")
        
        self.explanation_text = QTextBrowser()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setOpenLinks(False)
        self.explanation_text.anchorClicked.connect(self._handle_link)
        self.explanation_text.setStyleSheet("border: none; background-color: transparent;")

        self.layout.addWidget(self.header_label)
        self.layout.addWidget(self.explanation_text)
        self.setLayout(self.layout)

    def update_explanation(self, explanation_obj, section_id: str):
        css = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; line-height: 1.5; color: #2c3e50; }
            .button {
                display: block;
                background-color: #f0f4f8;
                color: #2E86C1;
                padding: 10px;
                margin-bottom: 8px;
                border-radius: 6px;
                text-decoration: none;
                border: 1px solid #d1d9e6;
            }
        </style>
        """
        html = css + f"<div class='explanation'>{explanation_obj.explanation}</div>"
   
        # Inside Sidebar.update_explanation
        if hasattr(explanation_obj, 'resources') and explanation_obj.resources:
            html += "<div class='section-title'>Recommended for You:</div>"
            for res in explanation_obj.resources:
                # Style as a card
                html += f"""
                <div style='background-color: #ffffff; border: 1px solid #d1d9e6; 
                            border-radius: 5px; padding: 10px; margin-bottom: 10px;'>
                    <b style='color: #2c3e50;'>{res.title}</b><br>
                    <i style='font-size: 11px; color: #7f8c8d;'>Type: {res.type} | Level: {res.difficulty}</i><br>
                    <p style='font-size: 12px; margin: 5px 0;'>{res.description}</p>
                    <a href='{res.url}' style='color: #E67E22; font-weight: bold;'>Open Resource →</a>
                </div>
                """
        if explanation_obj.follow_up_questions:
            html += "<hr><div style='font-weight:bold; margin-bottom:10px;'>Follow-up:</div>"
            for q in explanation_obj.follow_up_questions:
                # Encode section_id into the link
                link = f"app://followup?text={q}&section={section_id}"
                html += f"<a class='button' href='{link}'>{q}</a>"

        self.explanation_text.setHtml(html)

    def _handle_link(self, url: QUrl):
        query = QUrlQuery(url.query())
        question = query.queryItemValue("text")
        section = query.queryItemValue("section")
        self.question_clicked.emit(question, section)

