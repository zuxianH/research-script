"""T=0 covariance from small Bloch dynamical matrices, without a dense bath.

Each displacement already contains sqrt(1/Nq). Its outer product is summed
once. This is exactly the finite periodic bath covariance, not an added bulk
measure or a second Debye-Waller factor.
"""
import numpy as np


def covariance_from_groups(groups,basis_atoms):
    covariance=np.zeros((basis_atoms,3,3),complex);count=0
    for group in groups:
        d=group['displacements_m'][:basis_atoms]
        covariance+=np.einsum('iam,ibm->iab',d,d.conj());count+=d.shape[-1]
    imaginary=float(np.linalg.norm(covariance.imag)/np.linalg.norm(covariance.real))
    if imaginary>1e-10:raise ValueError('complete Bloch bath covariance is not real')
    covariance=covariance.real;common=np.mean(covariance,axis=0)
    site=float(np.max(np.linalg.norm(covariance-common,axis=(-2,-1)))/np.linalg.norm(common))
    off=common-np.diag(np.diag(common));relative_off=float(np.linalg.norm(off)/np.linalg.norm(common))
    if site>1e-9 or relative_off>1e-9:raise ValueError('expected equivalent Si sites and diagonal covariance')
    return np.diag(np.diag(common)),{'site_equivalence_relative_residual':site,'off_diagonal_relative_residual':relative_off,
        'imaginary_relative_residual':imaginary,'physical_mode_count':count,'dense_bath_dynamical_matrix_constructed':False}
