import hashlib
import json


def compute_hash(obj):
    """Compute a stable hash of a Python object."""
    # We'll convert to JSON string for simplicity
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        s = str(obj)
    return hashlib.sha256(s.encode()).hexdigest()