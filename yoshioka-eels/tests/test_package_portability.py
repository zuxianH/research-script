"""Installation boundaries and stable checkpoint serialization."""

from dataclasses import dataclass
import importlib.abc
import importlib
import json
from pathlib import Path
import pkgutil
import sys
import tempfile
import unittest

import numpy as np

import yoshioka
from yoshioka.lattice_dynamics import primitive_atom_mapping
from yoshioka.provenance import fingerprint, json_write, sha256


class PackagePortabilityTests(unittest.TestCase):
    def test_all_solver_modules_import_without_benchmarks(self):
        class NoBenchmarks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "benchmarks" or fullname.startswith("benchmarks."):
                    raise AssertionError(f"Installed core imported {fullname}")

        loaded = {key: value for key, value in sys.modules.items()
                  if key == "benchmarks" or key.startswith("benchmarks.")}
        for key in loaded:
            del sys.modules[key]
        guard = NoBenchmarks()
        sys.meta_path.insert(0, guard)
        try:
            for module in pkgutil.iter_modules(yoshioka.__path__, "yoshioka."):
                if module.name != "yoshioka.__main__":
                    importlib.reload(importlib.import_module(module.name))
        finally:
            sys.meta_path.remove(guard)
            sys.modules.update(loaded)

    def test_fingerprint_and_checkpoint_bytes_preserve_serialization_contract(self):
        import hashlib

        @dataclass
        class Settings:
            energy_eV: float = 60000.
            count: int = 3

        expected_settings = b'{"count": 3, "energy_eV": 60000.0}'
        self.assertEqual(fingerprint(Settings()), hashlib.sha256(expected_settings).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.json"
            data = {"complex": 2 + 3j, "array": np.array([1., 2.]), "count": np.int64(3)}
            json_write(a, data)
            expected = b'{\n  "complex": "(2+3j)",\n  "array": [\n    1.0,\n    2.0\n  ],\n  "count": 3\n}\n'
            self.assertEqual(a.read_bytes(), expected)
            self.assertEqual(sha256(a), hashlib.sha256(expected).hexdigest())
            self.assertEqual(json.loads(a.read_text())["array"], [1., 2.])
            with self.assertRaises(ValueError):
                json_write(a, {"bad": float("nan")})
            self.assertEqual(a.read_bytes(), expected)

    def test_primitive_mapping_recovers_basis_and_translations(self):
        from types import SimpleNamespace
        model = SimpleNamespace(cell_m=np.diag([2., 3., 4.]) * 1e-10,
                                positions_m=np.array([[0., 0., 0.], [.5, .8, 1.]]) * 1e-10)
        expected_basis = np.array([1, 0, 1])
        shifts = np.array([[1, -2, 0], [0, 0, 1], [-1, 1, -1]])
        positions = model.positions_m[expected_basis] + shifts @ model.cell_m
        basis, translations = primitive_atom_mapping(model, positions)
        np.testing.assert_array_equal(basis, expected_basis)
        np.testing.assert_array_equal(translations, shifts)
        with self.assertRaises(ValueError):
            primitive_atom_mapping(model, positions + .12e-10)


if __name__ == "__main__":
    unittest.main()
