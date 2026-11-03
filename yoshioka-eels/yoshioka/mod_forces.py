"""Exact supplied Kumagai MOD forces through LAMMPS tersoff/mod.

LAMMPS metal units are isolated at this interface: angstrom, eV, eV/angstrom,
and bar. No ordinary-Tersoff or alternate calculator fallback is provided.
"""

import ctypes
from pathlib import Path
import sys

import numpy as np


def potential_metadata(path):
    text = Path(path).read_text()
    tokens = " ".join(line.split("#")[0] for line in text.splitlines()).split()
    names = ("beta", "alpha", "h", "eta", "beta_ters", "lambda2", "B", "R", "D", "lambda1", "A", "n", "c1", "c2", "c3", "c4", "c5")
    if tokens[:3] != ["Si", "Si", "Si"] or len(tokens) != len(names)+3 or "Kumagai" not in text:
        raise ValueError("expected the supplied single-element Kumagai modified Tersoff parameter file")
    values = dict(zip(names, map(float, tokens[3:])))
    return {"parameters_metal_units": values, "inner_cutoff_angstrom": values["R"]-values["D"], "pair_cutoff_angstrom": values["R"]+values["D"],
            "metadata_header": text.splitlines()[0], "pair_style": "tersoff/mod"}


def load_lammps():
    # Official wheel needs MPICH's libmpi.so.12. Preload its real library, not
    # an ABI-incompatible symlink to the host's Open MPI libmpi.so.40.
    runtime = Path(sys.prefix) / "lib/libmpi.so.12"
    if runtime.is_file():
        ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    try:
        from lammps import lammps
        return lammps
    except (ImportError, OSError) as exc:
        raise RuntimeError("LAMMPS with pair_style tersoff/mod and its matching MPI runtime is required") from exc


class ModifiedTersoffForces:
    """Reusable serial force calculator, keeping atomic ID/order fixed."""

    def __init__(self, atoms, potential):
        self.potential = Path(potential).resolve()
        if not self.potential.is_file():
            raise FileNotFoundError(self.potential)
        self.atoms = atoms.copy()
        self.cell = np.asarray(atoms.cell).copy()
        if not np.allclose(self.cell, np.diag(np.diag(self.cell))) or not np.all(atoms.pbc) or not np.all(atoms.numbers == 14):
            raise ValueError("MOD force interface requires the periodic orthorhombic Si supercell")
        self.lmp = load_lammps()(cmdargs=["-log", "none", "-screen", "none"])
        if not self.lmp.has_style("pair", "tersoff/mod"):
            self.close()
            raise RuntimeError("installed LAMMPS does not contain pair_style tersoff/mod")
        lengths = np.diag(self.cell)
        commands = ["units metal", "atom_style atomic", "boundary p p p", "atom_modify map array sort 0 0.0",
                    "region box block 0 %.17g 0 %.17g 0 %.17g units box" % tuple(lengths),
                    "create_box 1 box", "mass 1 %.17g" % atoms.get_masses()[0]]
        self.lmp.commands_list(commands)
        for position in atoms.positions:
            self.lmp.command("create_atoms 1 single %.17g %.17g %.17g units box" % tuple(position))
        self.lmp.commands_list(["pair_style tersoff/mod", f'pair_coeff * * "{self.potential}" Si', "neighbor 0.5 bin",
                                "neigh_modify every 1 delay 0 check yes", "thermo 0", "thermo_style custom pe pxx pyy pzz pxy pxz pyz", "run 0"])
        self.evaluations = 0

    def evaluate(self, positions=None):
        positions = self.atoms.positions if positions is None else np.asarray(positions, dtype=np.float64)
        if positions.shape != self.atoms.positions.shape or not np.all(np.isfinite(positions)):
            raise ValueError("positions must be a finite N by 3 array")
        wrapped = np.ascontiguousarray(positions % np.diag(self.cell), dtype=np.float64)
        self.lmp.scatter_atoms("x", 1, 3, wrapped.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.lmp.command("run 0 pre yes post no")
        pointer = self.lmp.gather_atoms("f", 1, 3)
        forces = np.ctypeslib.as_array(pointer, shape=(len(positions) * 3,)).reshape(-1, 3).copy()
        energy = float(self.lmp.get_thermo("pe"))
        stress = np.array([self.lmp.get_thermo(name) for name in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")])
        self.evaluations += 1
        return {"energy_ev": energy, "forces_ev_per_angstrom": forces, "pressure_bar_voigt": stress}

    def close(self):
        if getattr(self, "lmp", None) is not None:
            self.lmp.close()
            self.lmp = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def central_force_constants(unit, repetitions, displacement_angstrom, potential):
    """24 force calls: central differences for 4 basis atoms times 3 axes.

    Exact lattice-translation symmetry reconstructs other displaced cells.
    Return C[R,a,alpha,b,beta]=Phi(Ra,0b), in eV/angstrom^2. Displaced
    structures are never relaxed, and no hiPhive model is used here.
    """
    repetitions = tuple(map(int, repetitions))
    supercell = unit.repeat(repetitions)
    cells = np.array(list(np.ndindex(*repetitions)))
    basis_size = len(unit)
    columns = np.empty((len(cells), basis_size, 3, basis_size, 3))
    with ModifiedTersoffForces(supercell, potential) as calculator:
        for atom in range(basis_size):
            for axis in range(3):
                plus, minus = supercell.positions.copy(), supercell.positions.copy()
                plus[atom, axis] += displacement_angstrom
                minus[atom, axis] -= displacement_angstrom
                forces_plus = calculator.evaluate(plus)["forces_ev_per_angstrom"]
                forces_minus = calculator.evaluate(minus)["forces_ev_per_angstrom"]
                columns[:, :, :, atom, axis] = (-(forces_plus - forces_minus) / (2 * displacement_angstrom)).reshape(len(cells), basis_size, 3)
    return columns, supercell, cells


def complete_force_constants(columns, repetitions):
    """Expand exact lattice translations into a Phonopy-compatible full array."""
    repetitions = tuple(map(int, repetitions))
    cells = np.array(list(np.ndindex(*repetitions)))
    basis_size = columns.shape[1]
    output = np.empty((len(cells), basis_size, 3, len(cells), basis_size, 3))
    for j, cell in enumerate(cells):
        differences = (cells - cell) % repetitions
        indices = np.ravel_multi_index(differences.T, repetitions)
        output[:, :, :, j, :, :] = columns[indices]
    n_atoms = len(cells) * basis_size
    return output.reshape(n_atoms, 3, n_atoms, 3).transpose(0, 2, 1, 3)


def correct_sum_rules(force_constants):
    """Symmetric orthogonal projection removing only rigid translation couplings.

    P H_sym P enforces permutation symmetry and the acoustic sum rule together.
    Raw arrays and the correction norm must be retained by callers.
    """
    n_atoms = len(force_constants)
    matrix = force_constants.transpose(0, 2, 1, 3).reshape(3*n_atoms, 3*n_atoms)
    matrix = (matrix + matrix.T) / 2
    translation = np.tile(np.eye(3), (n_atoms, 1)) / np.sqrt(n_atoms)
    product = matrix @ translation
    corrected = matrix - product @ translation.T - translation @ product.T + translation @ (translation.T @ product) @ translation.T
    return corrected.reshape(n_atoms, 3, n_atoms, 3).transpose(0, 2, 1, 3)


def force_constant_residuals(force_constants):
    norm = float(np.linalg.norm(force_constants))
    return {"permutation": float(np.linalg.norm(force_constants - force_constants.transpose(1, 0, 3, 2)) / norm),
            "translational": float(np.linalg.norm(np.sum(force_constants, axis=1)) / norm)}
