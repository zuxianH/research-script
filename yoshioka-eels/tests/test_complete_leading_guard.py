import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from benchmarks.complete_leading_eels import setup,run
from benchmarks.complete_leading_guard import audit_inputs,audit_results,audit_electron_potential,sha256
from yoshioka.electron_phonon import LobatoSilicon


class CompleteLeadingGuardTests(unittest.TestCase):
    def test_real_small_material_resume_is_readonly_and_detects_raw_corruption(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            p=Path(td);setup(p,repeats=1)
            run(p,'born',0,batches=1,workers=1,grid=5,steps=17)
            state=p/'pairs/born/batch00/state.json';before=state.read_bytes()
            run(p,'born',0,batches=1,workers=1,grid=5,steps=17)
            self.assertEqual(state.read_bytes(),before)
            self.assertTrue(audit_results(p,'born')['all_checks_passed'])
            raw=p/'pairs/born/batch00/pair000000.npz'
            with np.load(raw) as d:changed={k:d[k] for k in d.files}
            changed['weights']=changed['weights']+1e-8
            np.savez_compressed(raw,**changed)
            with self.assertRaisesRegex(ValueError,'Changed committed result'):audit_results(p,'born')

    def test_live_reference_covariance_guard_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);setup(p,repeats=1);self.assertTrue(audit_inputs(p))
            path=p/'covariance/T300_mesh32.json';d=json.loads(path.read_text())
            d['covariance_m2'][0][0]=1e-24;path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Reference covariance'):audit_inputs(p)

    def test_live_electron_table_guard_rejects_same_environment_table_change(self):
        actual=sha256;table=Path(LobatoSilicon.from_abtem_table().source_path)
        with patch('benchmarks.complete_leading_guard.sha256',side_effect=lambda p:'changed' if Path(p)==table else actual(p)):
            with self.assertRaisesRegex(ValueError,'Changed Lobato'):audit_electron_potential()

    def test_live_config_is_compared_to_committed_contract(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            p=Path(td);setup(p,repeats=1);run(p,'born',0,batches=1,workers=1,grid=5,steps=17)
            path=p/'config.json';d=json.loads(path.read_text());d['padding_angstrom']=13;path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Changed live calculation'):audit_results(p,'born')

    def test_full_adapted_proposal_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            p=Path(td);setup(p,repeats=1)
            mass=np.load(p/'proposal_mass.npy');np.save(p/'adapted_mass.npy',mass)
            run(p,'adapted',0,batches=1,workers=1,grid=5,steps=17)
            mass[0,0]+=1e-9;mass/=mass.sum();np.save(p/'adapted_mass.npy',mass)
            with self.assertRaisesRegex(ValueError,'Changed full sampling-table'):audit_results(p,'adapted')

if __name__=='__main__':unittest.main()
