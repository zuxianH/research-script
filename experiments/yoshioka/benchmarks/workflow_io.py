"""Small provenance/checkpoint helpers for the material-to-EELS benchmarks."""

from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def environment():
    names = ("numpy", "scipy", "matplotlib", "ase", "lammps", "mpich", "phonopy", "hiphive", "spglib", "symfc", "trainstation", "abtem")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def previous_statuses():
    values = {}
    for path in sorted((ROOT/"benchmark_results").glob("*/results.json")):
        data = json.loads(path.read_text())
        values[path.parent.name] = data.get("status")
    return values


def artifact(path, base, units, description):
    path = Path(path)
    return {"path": str(path.relative_to(base)), "sha256": sha256(path), "units": units, "description": description}


class ResourceLimits:
    def __init__(self, seconds=1800, memory_gib=4, max_refinements=20):
        self.started = time.monotonic()
        self.seconds = float(seconds)
        self.memory_bytes = float(memory_gib)*1024**3
        self.max_refinements = int(max_refinements)
        self.completed = 0

    def check(self, estimated_bytes=0):
        if time.monotonic()-self.started > self.seconds:
            raise TimeoutError("configured wall-time limit reached; completed checkpoints remain reusable")
        if estimated_bytes > self.memory_bytes:
            raise MemoryError("estimated allocation exceeds configured memory limit")
        import psutil
        if psutil.Process().memory_info().rss > self.memory_bytes:
            raise MemoryError("observed resident memory exceeds configured limit; completed checkpoints remain reusable")
        if self.completed >= self.max_refinements:
            raise TimeoutError("configured refinement limit reached")

    def costs(self):
        import resource
        return {"elapsed_seconds": time.monotonic()-self.started,
                "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                "configured_runtime_seconds": self.seconds, "configured_memory_gib": self.memory_bytes/1024**3,
                "completed_refinements": self.completed, "configured_max_refinements": self.max_refinements}
