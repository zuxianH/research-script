"""Checkpoint fingerprints shared by installed solvers and research runners."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value)) + "\n")
    temporary.replace(path)


def fingerprint(data):
    if is_dataclass(data):
        data = asdict(data)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
