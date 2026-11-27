"""The real embedder: sentence-transformers on the CPU or a GPU this PyTorch build can really use.

There is deliberately no fallback to fake vectors: if the model cannot load, indexing must fail
loudly. (The code this replaces silently substituted hash vectors, which look like embeddings but
retrieve nothing meaningful.)
"""
from __future__ import annotations

import logging
import os
import threading
import warnings
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def pick_device() -> str:
    """"cuda" only if this PyTorch build has kernels for the installed GPU, else "cpu".

    sentence-transformers chooses CUDA on its own whenever a GPU exists, and crashes on a card the
    wheel was not built for. Override with DOC_EXPLAINER_EMBED_DEVICE=cpu|cuda.
    """
    forced = os.getenv("DOC_EXPLAINER_EMBED_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not torch.cuda.is_available():
                return "cpu"
            major, minor = torch.cuda.get_device_capability()
            archs = torch.cuda.get_arch_list()
        cc = major * 10 + minor
        if f"sm_{cc}" in archs or any(a.startswith("compute_") and int(a[8:]) <= cc for a in archs):
            return "cuda"
        logger.warning("GPU compute capability %d.%d is not supported by this PyTorch build; embedding on the CPU.", major, minor)
    except Exception:
        logger.warning("Could not use the GPU for embeddings; using the CPU.", exc_info=True)
    return "cpu"


def _quiet_progress() -> None:
    """No "Loading weights" progress bar in the log."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


class SentenceTransformerEmbedder:
    """Loads the model once, on first use, and keeps it."""

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 64, check_for_updates: bool = False):
        self.name = model_name
        self._check = check_for_updates
        self._device = device
        self._batch = batch_size
        self._model = None
        self._lock = threading.Lock()
        self._dimension: Optional[int] = None

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                device = self._device or pick_device()
                logger.info("Loading embedding model %s on %s", self.name, device)
                self._model = self._open(SentenceTransformer, device)
                size = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
                self._dimension = int(size())      # (the older name is deprecated in newer sentence-transformers)
        return self._model

    def _open(self, factory, device: str):
        """The cached copy without any network call, unless asked to check for updates or it is not cached yet."""
        _quiet_progress()
        if self._check:
            return factory(self.name, device=device)
        try:
            return factory(self.name, device=device, local_files_only=True)
        except (OSError, ValueError):                 # not downloaded yet: this once, fetch it
            logger.info("Embedding model %s is not on this computer yet; downloading it", self.name)
            return factory(self.name, device=device)

    @property
    def dimension(self) -> int:
        self._load()
        return int(self._dimension)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._load().encode(list(texts), batch_size=self._batch, normalize_embeddings=True,
                                      show_progress_bar=False)
        return [v.tolist() for v in vectors]
