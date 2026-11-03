"""Real-space harmonic force constants, dynamical matrices, and covariances.

Public arrays are SI. Row cell vectors A have reciprocal rows B=2*pi*A^-T.
The atomic-gauge convention matches Phonopy:
D_ab(q)=sum_R Phi(0a,Rb) exp[+i q.(R+tau_b-tau_a)]/sqrt(Ma Mb).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, elementary_charge, hbar
from scipy.linalg import null_space


@dataclass
class RealSpaceForceConstants:
    cell_m: np.ndarray
    positions_m: np.ndarray
    masses_kg: np.ndarray
    i: np.ndarray
    j: np.ndarray
    translations: np.ndarray
    phi_n_per_m: np.ndarray

    @property
    def reciprocal_m_inv(self):
        return 2 * np.pi * np.linalg.inv(self.cell_m).T

    @property
    def relative_vectors_m(self):
        return self.translations @ self.cell_m + self.positions_m[self.j] - self.positions_m[self.i]

    def dynamical(self, q_fractional):
        q = np.asarray(q_fractional, dtype=np.float64)
        scalar = q.ndim == 1
        q = np.atleast_2d(q)
        cartesian = q @ self.reciprocal_m_inv
        phases = np.exp(1j * (cartesian @ self.relative_vectors_m.T))
        n = len(self.positions_m)
        matrices = np.zeros((len(q), n, 3, n, 3), dtype=np.complex128)
        for a in range(n):
            for b in range(n):
                chosen = (self.i == a) & (self.j == b)
                matrices[:, a, :, b, :] = np.einsum("qk,kab->qab", phases[:, chosen], self.phi_n_per_m[chosen]) / np.sqrt(self.masses_kg[a]*self.masses_kg[b])
        matrices = matrices.reshape(len(q), 3*n, 3*n)
        return matrices[0] if scalar else matrices

    def modes(self, q_fractional):
        matrices = self.dynamical(q_fractional)
        values, vectors = np.linalg.eigh(matrices)
        # Signed frequencies preserve genuine instabilities. No clipping.
        omega = np.sign(values) * np.sqrt(np.abs(values))
        return omega, vectors

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})


def real_space_from_supercell(unit, repetitions, force_constants, threshold_ev_per_a2=1e-9):
    """Extract independently unwrapped physical image vectors before interpolation.

    The supercell must be larger than twice the measured harmonic range; an
    ambiguous nonzero interaction on a periodic bisector is rejected.
    """
    from scipy.constants import atomic_mass
    repetitions = np.array(repetitions, dtype=int)
    cells = np.array(list(np.ndindex(*repetitions)))
    n_basis = len(unit)
    supercell_vectors = repetitions[:, None] * np.asarray(unit.cell)
    positions = np.asarray(unit.positions)
    pairs_i, pairs_j, translations, tensors = [], [], [], []
    discarded_squared = 0.0
    maximum_range = 0.0
    for atom in range(n_basis):
        for cell_index, cell in enumerate(cells):
            for other in range(n_basis):
                tensor = force_constants[atom, cell_index*n_basis+other]
                if np.linalg.norm(tensor) < threshold_ev_per_a2:
                    discarded_squared += float(np.sum(tensor**2))
                    continue
                relative = cell @ np.asarray(unit.cell) + positions[other] - positions[atom]
                fractional = relative @ np.linalg.inv(supercell_vectors)
                if np.any(np.isclose(np.abs(fractional - np.rint(fractional)), 0.5, atol=1e-8)):
                    raise ValueError("nonzero force constant has ambiguous supercell image; enlarge the force supercell")
                image = cell - np.rint(fractional).astype(int) * repetitions
                vector = image @ np.asarray(unit.cell) + positions[other] - positions[atom]
                maximum_range = max(maximum_range, float(np.linalg.norm(vector)))
                pairs_i.append(atom); pairs_j.append(other); translations.append(image); tensors.append(tensor)
    model = RealSpaceForceConstants(np.asarray(unit.cell)*1e-10, positions*1e-10, unit.get_masses()*atomic_mass,
                                   np.array(pairs_i), np.array(pairs_j), np.array(translations), np.array(tensors)*elementary_charge/1e-20)
    return model, {"maximum_nonzero_range_angstrom": maximum_range, "block_threshold_ev_per_angstrom2": threshold_ev_per_a2,
                   "discarded_compact_frobenius_ev_per_angstrom2": float(np.sqrt(discarded_squared)), "retained_blocks": len(tensors)}


def finite_periodic_modes(model: RealSpaceForceConstants, repetitions):
    """Full-model real eigenvectors, normalized over every declared atom.

    Only the three known rigid-translation vectors are removed. Negative
    eigenvalues in the complementary physical subspace cause a failure.
    """
    repetitions = np.asarray(repetitions, dtype=int)
    cells = np.array(list(np.ndindex(*repetitions)))
    nb = len(model.positions_m)
    count = len(cells)*nb
    matrix = np.zeros((count, 3, count, 3))
    for cell_index, cell in enumerate(cells):
        for a, b, shift, tensor in zip(model.i, model.j, model.translations, model.phi_n_per_m):
            target = np.ravel_multi_index(tuple((cell + shift) % repetitions), tuple(repetitions))
            matrix[cell_index*nb+a, :, target*nb+b, :] += tensor / np.sqrt(model.masses_kg[a]*model.masses_kg[b])
    matrix = matrix.reshape(3*count, 3*count)
    masses = np.tile(model.masses_kg, len(cells))
    translations = np.repeat(np.sqrt(masses), 3)[:, None] * np.tile(np.eye(3), (count, 1))
    translations /= np.linalg.norm(translations, axis=0)
    hermiticity = float(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix))
    rigid_residual = float(np.linalg.norm(matrix@translations)/np.linalg.norm(matrix))
    complement = null_space(translations.T)
    values, vectors = np.linalg.eigh(complement.T @ matrix @ complement)
    if np.any(values <= 0):
        raise ValueError(f"physical finite-model dynamical eigenvalues are not positive: minimum {values.min():.6e}")
    eigenvectors = (complement @ vectors).reshape(count, 3, -1)
    omega = np.sqrt(values)
    displacement = np.sqrt(hbar/(2*masses[:, None]*omega))[:, None, :] * eigenvectors
    return {"positions_m": ((cells @ model.cell_m)[:, None, :] + model.positions_m[None]).reshape(count, 3),
            "masses_kg": masses, "omega_rad_s": omega, "eigenvectors": eigenvectors, "displacements_m": displacement,
            "hermiticity_residual": hermiticity, "rigid_translation_residual": rigid_residual,
            "translation_vectors": translations, "removed_rigid_mode_count": 3}


def covariance_mesh(model: RealSpaceForceConstants, mesh: int, temperatures=(0, 300), chunk_size=2048):
    """Uniform oriented-cell reciprocal mesh, normalized q weights exactly once.

    Gamma's known three rigid translations are omitted explicitly. All other
    physical acoustic modes, including arbitrarily small nonzero q, are kept.
    """
    points = np.array(list(np.ndindex(mesh, mesh, mesh)), dtype=float) / mesh
    n = len(model.positions_m)
    covariance = np.zeros((len(temperatures), n, 3, 3))
    min_physical = np.inf
    hermiticity = 0.0
    for start in range(0, len(points), chunk_size):
        q = points[start:start+chunk_size]
        dynamical = model.dynamical(q)
        hermiticity = max(hermiticity, float(np.max(np.linalg.norm(dynamical-dynamical.swapaxes(-1, -2).conj(), axis=(-2,-1))/np.linalg.norm(dynamical, axis=(-2,-1)))))
        values, vectors = np.linalg.eigh(dynamical)
        physical = np.ones(values.shape, dtype=bool)
        gamma = np.all(q == 0, axis=1)
        if np.any(gamma):
            # At Gamma, identify translations by overlap, never a frequency band.
            translation = np.tile(np.eye(3), (n, 1)) * np.repeat(np.sqrt(model.masses_kg), 3)[:, None]
            translation /= np.linalg.norm(translation, axis=0)
            indices = np.argsort(np.sum(np.abs(vectors[gamma].swapaxes(-1,-2).conj() @ translation)**2, axis=-1), axis=-1)[:, -3:]
            for row, removed in zip(np.flatnonzero(gamma), indices):
                physical[row, removed] = False
        if np.any(values[physical] <= 0):
            raise ValueError("negative physical phonon eigenvalue encountered; covariance not defined")
        omega = np.zeros_like(values)
        omega[physical] = np.sqrt(values[physical])
        min_physical = min(min_physical, float(np.min(omega[physical])))
        vectors = vectors.reshape(len(q), n, 3, 3*n)
        for index, temperature in enumerate(temperatures):
            occupation_factor = np.ones_like(values)
            if temperature > 0:
                x = hbar*omega[physical]/(Boltzmann*temperature)
                occupation_factor[physical] = 1 + 2*np.exp(-x)/(-np.expm1(-x))
            factors = np.zeros_like(values)
            factors[physical] = occupation_factor[physical]/omega[physical]
            covariance[index] += np.einsum("qiam,qibm,qm->iab", vectors, vectors.conj(), factors).real * hbar/(2*model.masses_kg[:, None, None]*len(points))
    return covariance, {"mesh": [mesh]*3, "q_points": len(points), "q_weight": 1/len(points),
                        "removed_gamma_translations": 3, "minimum_retained_energy_mev": min_physical*hbar/(1e-3*elementary_charge), "hermiticity_residual": hermiticity}


def primitive_atom_mapping(primitive,positions_m):
    basis=[];translations=[]
    for position in positions_m:
        fractions=(position[None]-primitive.positions_m)@np.linalg.inv(primitive.cell_m)
        residual=np.linalg.norm(fractions-np.rint(fractions),axis=1)
        index=int(np.argmin(residual))
        if residual[index]>1e-7:raise ValueError('fixed specimen atom cannot be mapped to primitive basis')
        basis.append(index);translations.append(np.rint(fractions[index]).astype(int))
    return np.array(basis),np.array(translations)
