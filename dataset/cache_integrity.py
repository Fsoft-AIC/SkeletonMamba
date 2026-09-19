"""Content checksums shared by preprocessing and runtime dataset readers."""
from __future__ import annotations

import hashlib

import numpy as np


def arrays_hash(arrays):
    """Hash names, shapes, dtypes and exact array bytes, independent of ZIP layout."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_content_checksum(arrays, metadata, path):
    expected = metadata.get('content_sha256')
    if expected is None and metadata.get('synthetic') is True:
        return
    if not isinstance(expected, str) or expected != arrays_hash(arrays):
        raise ValueError(f'{path}: missing or mismatched prepared content checksum; regenerate the cache')
