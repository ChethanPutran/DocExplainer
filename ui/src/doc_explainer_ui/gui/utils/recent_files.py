"""Recently opened files, and which local file each server document came from.

Stored in one small JSON file: ``{"documents": [paths, newest first], "library": {doc_id: path}}``. The second part is
what lets the Library reopen a document: the server keeps the processed data, not a copy a viewer can show.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RecentFiles:
    def __init__(self, file: Path, limit: int = 10):
        self.file, self.limit = Path(file), max(1, int(limit))
        self._paths: List[str] = []
        self._by_doc: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            self._paths = [p for p in data.get("documents", []) if isinstance(p, str)]
            self._by_doc = {k: v for k, v in (data.get("library") or {}).items() if isinstance(v, str)}
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001 - a damaged list must never stop the app
            logger.warning("Could not read the recent files list: %s", e)

    def _save(self) -> None:
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps({"documents": self._paths, "library": self._by_doc}, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not save the recent files list: %s", e)

    # ------------------------------------------------------------------ recent list
    def add(self, path: str) -> None:
        """Newest first; opening a file again moves it to the top instead of adding a copy."""
        path = os.path.abspath(path)
        self._paths = [path] + [p for p in self._paths if p != path]
        self._paths = self._paths[: self.limit]
        self._save()

    def paths(self, existing_only: bool = True) -> List[str]:
        return [p for p in self._paths if not existing_only or os.path.exists(p)]

    def remove(self, path: str) -> None:
        self._paths = [p for p in self._paths if p != path]
        self._save()

    def clear(self) -> None:
        self._paths = []
        self._save()

    # ------------------------------------------------------------------ library links
    def link(self, doc_id: str, path: str) -> None:
        self._by_doc[str(doc_id)] = os.path.abspath(path)
        self._save()

    def path_for(self, doc_id: str) -> Optional[str]:
        path = self._by_doc.get(str(doc_id))
        return path if path and os.path.exists(path) else None

    def linked_paths(self) -> List[str]:
        return list(self._by_doc.values())

    def forget_document(self, doc_id: str) -> None:
        path = self._by_doc.pop(str(doc_id), None)
        if path:
            self._paths = [p for p in self._paths if p != path]
        self._save()
