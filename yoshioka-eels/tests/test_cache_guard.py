import tempfile
from pathlib import Path
import unittest

from benchmarks.guarded_run import validate_guard


class CacheGuardTests(unittest.TestCase):
    def test_changed_physics_rejected_and_identical_budget_independent_profile_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'cache_guard.json'
            physics = {'constants': {'hbar': 1.}, 'kernel': 'old'}
            first = validate_guard(target, physics)
            self.assertEqual(validate_guard(target, physics), first)
            with self.assertRaises(ValueError):
                validate_guard(target, {**physics, 'kernel': 'changed'})
            self.assertEqual(validate_guard(target, physics), first)

    def test_unguarded_committed_cache_requires_explicit_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'cache/x').mkdir(parents=True)
            (directory/'cache/x/state.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_guard(directory/'cache_guard.json', {'kernel': 'known'})
            self.assertTrue(validate_guard(directory/'cache_guard.json', {'kernel': 'known'}, adopt=True)['adopted_existing_cache'])
