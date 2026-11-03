"""Independent physical-unit and atom-order checks of the explicit SNAP boundary."""
import os
from pathlib import Path
import unittest
import numpy as np
from scipy.constants import elementary_charge
from yoshioka.snap_forces import SNAPForces,snap_metadata

INPUTS=Path(os.environ.get('YOSHIOKA_SNAP_INPUTS','/home/zuxian/Documents/PhD/Yoshioka'))


@unittest.skipUnless((INPUTS/'SNAPotential.snapcoeff').exists(),'supplied Si SNAP inputs unavailable')
class SnapForceTests(unittest.TestCase):
    def setUp(self):
        from ase.io import read
        self.unit=read(INPUTS/'Si_110_temp300.lmp',format='lammps-data',atom_style='atomic')
        self.unit.set_masses(np.full(len(self.unit),28.085))
        self.coeff=INPUTS/'SNAPotential.snapcoeff';self.param=INPUTS/'SNAPotential.snapparam'

    def test_supplied_model_identity_and_pressure_units(self):
        metadata=snap_metadata(self.coeff,self.param)
        self.assertEqual(metadata['coefficients_sha256'],'04c04279a25fb52893be5ac2ef2ec48d2134168e38dff20234ce85a920e0123e')
        self.assertEqual(metadata['parameters_sha256'],'9a619a0c1e76bf81b8ef9384f6e97ee47b61a643fab02ba16221c73381aee0da')
        self.assertEqual(metadata['pair_cutoff_angstrom'],4.9)
        energies=[];eps=1e-6
        for scale in (1-eps,1,1+eps):
            atom=self.unit.copy();atom.set_cell(atom.cell*scale,scale_atoms=True)
            with SNAPForces(atom,self.coeff,self.param) as calc: values=calc.evaluate()
            energies.append(values['energy_ev'])
            if scale==1:pressure=values['pressure_bar_voigt'][:3].mean()
        derivative=(energies[2]-energies[0])/(2*eps)
        expected=-3*self.unit.get_volume()*1e-30*pressure*1e5/elementary_charge
        self.assertLess(abs(derivative-expected),1e-6)

    def test_force_derivative_translation_and_atom_order(self):
        atoms=self.unit.repeat((2,3,3));atoms.positions[0,0]+=.013
        with SNAPForces(atoms,self.coeff,self.param) as calc:
            reference=calc.evaluate();eps=1e-5
            plus=atoms.positions.copy();minus=plus.copy();plus[0,0]+=eps;minus[0,0]-=eps
            derivative=-(calc.evaluate(plus)['energy_ev']-calc.evaluate(minus)['energy_ev'])/(2*eps)
            translated=calc.evaluate(atoms.positions+np.array([.23,.47,.18]))
        self.assertLess(abs(derivative-reference['forces_ev_per_angstrom'][0,0]),1e-5)
        np.testing.assert_allclose(translated['forces_ev_per_angstrom'],reference['forces_ev_per_angstrom'],atol=1e-10)
        self.assertAlmostEqual(translated['energy_ev'],reference['energy_ev'],places=9)
        order=np.random.default_rng(441).permutation(len(atoms))
        with SNAPForces(atoms[order],self.coeff,self.param) as calc:permuted=calc.evaluate()
        np.testing.assert_allclose(permuted['forces_ev_per_angstrom'],reference['forces_ev_per_angstrom'][order],atol=1e-10)

if __name__=='__main__':unittest.main()
