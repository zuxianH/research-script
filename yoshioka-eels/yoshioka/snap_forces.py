"""Explicit serial Si SNAP forces. Boundary units: Angstrom, eV, eV/A, bar.

The supplied coefficient/parameter files are passed unchanged to LAMMPS.
No MOD, Tersoff, fitted-parameter override, or alternate calculator fallback.
"""
import ctypes
from pathlib import Path
import sys
import numpy as np


def snap_metadata(coefficients, parameters):
    from .provenance import sha256
    coeff = Path(coefficients).resolve()
    param = Path(parameters).resolve()
    lines = [s.split('#')[0].strip() for s in coeff.read_text().splitlines()]
    lines = [s for s in lines if s]
    if lines[0].split()[0] != '1' or lines[1].split()[0] != 'Si':
        raise ValueError('the explicit interface requires a single Si SNAP element')
    values = dict(s.split('#')[0].split() for s in param.read_text().splitlines() if s.split('#')[0].strip())
    return {'pair_style': 'snap', 'element_mapping': ['Si'], 'units': 'metal',
            'coefficients_sha256': sha256(coeff), 'parameters_sha256': sha256(param),
            'supplied_parameters': values, 'element_radius_angstrom': float(lines[1].split()[1]),
            'pair_cutoff_angstrom': 2*float(lines[1].split()[1])*float(values['rcutfac'])}


class SNAPForces:
    """Reusable serial calculator preserving atom IDs and their order."""
    def __init__(self, atoms, coefficients, parameters):
        self.metadata = snap_metadata(coefficients, parameters)
        self.atoms = atoms.copy()
        self.cell = np.asarray(atoms.cell).copy()
        self.lmp = None
        if (not np.all(atoms.numbers == 14) or not np.all(atoms.pbc)
                or not np.allclose(self.cell, np.diag(np.diag(self.cell)))
                or np.any(np.diag(self.cell) <= 0)
                or not np.allclose(atoms.get_masses(), atoms.get_masses()[0])):
            raise ValueError('SNAP interface requires periodic orthorhombic monoisotopic Si')
        runtime = Path(sys.prefix)/'lib/libmpi.so.12'
        if runtime.is_file():
            ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
        try:
            from lammps import lammps
            self.lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
            if not self.lmp.has_style('pair', 'snap'):
                raise RuntimeError('LAMMPS ML-SNAP pair_style snap is required')
            self.lammps_version = self.lmp.version()
            commands = ['units metal', 'atom_style atomic', 'boundary p p p',
                        'atom_modify map array sort 0 0.0',
                        'region box block 0 %.17g 0 %.17g 0 %.17g units box' % tuple(np.diag(self.cell)),
                        'create_box 1 box', 'mass 1 %.17g' % atoms.get_masses()[0]]
            self.lmp.commands_list(commands)
            for position in atoms.positions % np.diag(self.cell):
                self.lmp.command('create_atoms 1 single %.17g %.17g %.17g units box' % tuple(position))
            if self.lmp.get_natoms() != len(atoms):
                raise ValueError('LAMMPS atom count/order construction failed')
            self.lmp.commands_list(['pair_style snap',
                f'pair_coeff * * "{Path(coefficients).resolve()}" "{Path(parameters).resolve()}" Si',
                'neighbor 0.5 bin', 'neigh_modify every 1 delay 0 check yes', 'thermo 0',
                'thermo_style custom pe pxx pyy pzz pxy pxz pyz', 'run 0'])
        except Exception:
            self.close()
            raise
        self.evaluations = 0

    def evaluate(self, positions=None):
        positions = self.atoms.positions if positions is None else np.asarray(positions, dtype=float)
        if positions.shape != self.atoms.positions.shape or not np.all(np.isfinite(positions)):
            raise ValueError('finite N by 3 positions required')
        wrapped = np.ascontiguousarray(positions % np.diag(self.cell), dtype=np.float64)
        self.lmp.scatter_atoms('x', 1, 3, wrapped.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.lmp.command('run 0 pre yes post no')
        self.evaluations += 1
        return {'energy_ev': float(self.lmp.get_thermo('pe')),
                'forces_ev_per_angstrom': self._gather('f'),
                'pressure_bar_voigt': np.array([self.lmp.get_thermo(s) for s in ('pxx','pyy','pzz','pyz','pxz','pxy')])}

    def _gather(self, name):
        return np.ctypeslib.as_array(self.lmp.gather_atoms(name,1,3), shape=(3*len(self.atoms),)).reshape(-1,3).copy()

    def relax_zero_pressure(self):
        """Relax the orthorhombic lengths and all basis coordinates at static 0 bar."""
        self.lmp.commands_list(['min_style cg', 'fix volume all box/relax aniso 0.0 vmax 0.001',
                               'minimize 1e-14 1e-10 1000 10000', 'unfix volume'])
        lo, hi, *_ = self.lmp.extract_box()
        self.cell = np.diag(np.asarray(hi)-np.asarray(lo))
        self.atoms.set_cell(self.cell)
        self.atoms.positions = self._gather('x')-lo
        return self.atoms.copy(), self.evaluate()

    def close(self):
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def central_snap_force_constants(unit, repetitions, displacement_angstrom, coefficients, parameters, limits=None):
    """24 central force evaluations; lattice translations reconstruct other cells."""
    if not np.isfinite(displacement_angstrom) or displacement_angstrom <= 0:
        raise ValueError('positive central displacement required')
    repetitions = tuple(map(int, repetitions))
    supercell = unit.repeat(repetitions)
    cells = np.array(list(np.ndindex(*repetitions)))
    columns = np.empty((len(cells),len(unit),3,len(unit),3))
    with SNAPForces(supercell, coefficients, parameters) as calculator:
        for atom in range(len(unit)):
            for axis in range(3):
                if limits is not None: limits.check()
                plus, minus = supercell.positions.copy(), supercell.positions.copy()
                plus[atom,axis] += displacement_angstrom
                minus[atom,axis] -= displacement_angstrom
                difference = calculator.evaluate(plus)['forces_ev_per_angstrom']-calculator.evaluate(minus)['forces_ev_per_angstrom']
                columns[:,:,:,atom,axis] = (-difference/(2*displacement_angstrom)).reshape(len(cells),len(unit),3)
    return columns, supercell, cells
