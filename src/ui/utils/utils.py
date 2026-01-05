import os
import fitz  # PyMuPDF

def safe_read_text(path: str) -> str:
	"""Read text file safely; returns empty string on error."""
	try:
		with open(path, "r", encoding="utf-8") as f:
			return f.read()
	except Exception:
		return ""

def ensure_dir(path: str):
	"""Create directory if it doesn't exist."""
	os.makedirs(path, exist_ok=True)


def load_document(file_path):
    """
    Load PDF or HTML/Text and return HTML string for QTextBrowser.
    """
    if file_path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            # Convert to simple HTML
            html = f"<pre>{text}</pre>"
            return html
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return "<h2>Error loading PDF</h2>"
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Wrap plain text or HTML as needed
            if content.strip().startswith("<"):
                return content  # Already HTML
            return f"<pre>{content}</pre>"
        except Exception as e:
            print(f"Error loading file: {e}")
            return "<h2>Error loading document</h2>"
