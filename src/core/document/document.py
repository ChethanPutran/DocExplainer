from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Document:
    doc_id: str
    path: str
    raw_text: str
    sections: List[str]
    embeddings: Dict[str, list]
