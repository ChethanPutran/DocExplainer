# import fitz
# from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
# from PySide6.QtGui import QPixmap, QImage
# from PySide6.QtCore import Qt, Signal
# from .base_viewer import BaseViewer, ActionType
# from .pdf_page import PDFPageWidget


# class PDFViewer(QWidget, BaseViewer):

#     text_action = Signal(str, str)  # action, selected_text

#     def __init__(self, parent=None):
#         super().__init__(parent)

#         self.scroll = QScrollArea(self)
#         self.scroll.setWidgetResizable(True)

#         self.container = QWidget()
#         self.layout = QVBoxLayout(self.container)
#         self.layout.setAlignment(Qt.AlignTop)

#         self.scroll.setWidget(self.container)

#         main = QVBoxLayout(self)
#         main.addWidget(self.scroll)

#         self.doc = None

#     def load(self, path):
#         self.clear()
#         self.doc = fitz.open(path)

#         scale = 1.5
#         mat = fitz.Matrix(scale, scale)

#         for page in self.doc:
#             pix = page.get_pixmap(matrix=mat)
#             img = QImage(
#                 pix.samples,
#                 pix.width,
#                 pix.height,
#                 pix.stride,
#                 QImage.Format_RGB888
#             )

#             words = page.get_text("words")

#             page_widget = PDFPageWidget(
#                 QPixmap.fromImage(img),
#                 words,
#                 scale
#             )

#             page_widget.text_selected.connect(self._on_page_text_selected)
#             self.layout.addWidget(page_widget)

#             self.layout.addWidget(page_widget)

#     def _on_page_text_selected(self, text):
#         self.text_action.emit(ActionType.SELECT.value, text)



#     def clear(self):
#         while self.layout.count():
#             item = self.layout.takeAt(0)
#             widget = item.widget()
#             if widget:
#                 widget.deleteLater()

import fitz
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal
from .base_viewer import BaseViewer, ActionType
from .pdf_page import PDFPageWidget


class PDFViewer(QWidget, BaseViewer):
    text_action = Signal(str, str)  # action, selected_text

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.container)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll)
        self.setLayout(main_layout)

        self.doc = None

    def load(self, path):
        """Load PDF and create page widgets"""
        self.clear()
        self.doc = fitz.open(path)
        scale = 1.5

        for page in self.doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = QImage(
                pix.samples,
                pix.width,
                pix.height,
                pix.stride,
                QImage.Format_RGB888
            )
            words = page.get_text("words")

            page_widget = PDFPageWidget(
                pixmap=QPixmap.fromImage(img),
                words=words,
                scale=scale
            )

            page_widget.text_action.connect(self._on_page_text_selected)
            self.layout.addWidget(page_widget)

    def _on_page_text_selected(self, action, text):
        """
        action: SELECT / EXPLAIN / SUMMARIZE / ASK
        text: the selected text
        """
        # Directly emit to parent / sidebar
        self.text_action.emit(action, text)


    def clear(self):
        """Clear all page widgets"""
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.doc = None
