"""Fixed laboratory-polarized optical modes in a commensurate finite bath."""
from dataclasses import dataclass
import numpy as np
from scipy import constants
from .lattice_dynamics import finite_periodic_modes


@dataclass
class FiniteOpticalBath:
    displacement_m: np.ndarray
    omega_rad_s: np.ndarray
    inactive_covariance_m2: np.ndarray
    cell_m: np.ndarray
    positions_m: np.ndarray
    labels: list


def finite_optical_bath(model, repetitions=(1,1,4), temperature_k=10.):
    """Project the highest Gamma optical cluster onto lab x/y; phase atom 0 > 0.

    The full declared periodic bath supplies all 3*N-3 nonrigid modes for covariance.
    Active vectors have unit norm over the actual specimen, not per unit cell.
    Local inactive Gaussian averaging is applied once by the electron operator.
    """
    if temperature_k < 0: raise ValueError('nonnegative temperature required')
    omega, vectors = model.modes(np.zeros(3))
    energies = constants.hbar*omega/(1e-3*constants.e)
    # Identify the optical top cluster by its freshly calculated frequencies.
    group = np.flatnonzero(abs(energies-energies.max()) < .01)
    if len(group) != 3: raise ValueError('expected an isolated three-dimensional top optical Gamma cluster')
    subspace = vectors[:,group]
    nb = len(model.positions_m)
    selected=[];active_omega=[];labels=[]
    dynamic = model.dynamical([0,0,0]).real
    for axis,name in ((0,'a:Gamma-optical-lab-x'),(1,'b:Gamma-optical-lab-y')):
        selector = np.tile(np.eye(3)[axis],nb)
        _,rotation = np.linalg.eigh(subspace.conj().T@(selector[:,None]*subspace))
        vector = subspace@rotation[:,-1]
        vector *= np.exp(-1j*np.angle(vector[axis]))
        if np.linalg.norm(vector.imag) > 1e-9: raise ValueError('Gamma polarization is not real')
        vector=vector.real;vector/=np.linalg.norm(vector)
        frequency = np.sqrt(vector@dynamic@vector)
        residual = np.linalg.norm(dynamic@vector-frequency**2*vector)/frequency**2
        polarization = np.sum(vector.reshape(nb,3)**2,axis=0)
        if polarization[axis]<1-1e-10 or residual>1e-7: raise ValueError('polarization projection is not a transverse normal mode')
        # Round only forbidden components at floating-point noise after the check.
        shaped=vector.reshape(nb,3);shaped[:,np.arange(3)!=axis]=0;shaped/=np.linalg.norm(shaped)
        selected.append(shaped);active_omega.append(frequency)
        labels.append({'id':name,'gamma_cluster_indices':group.tolist(),
                       'lab_polarization_weight':polarization.tolist(),'eigen_residual_relative':float(residual),
                       'phase':'atom 0 selected Cartesian component positive'})
    active_omega=np.array(active_omega)
    cells=int(np.prod(repetitions))
    active_vectors=np.tile(np.stack(selected,axis=-1),(cells,1,1))/np.sqrt(cells)
    masses=np.tile(model.masses_kg,cells)
    displacement=active_vectors*np.sqrt(constants.hbar/(2*masses[:,None,None]*active_omega[None,None,:]))
    full=finite_periodic_modes(model,repetitions)
    x=constants.hbar*full['omega_rad_s']/(constants.k*temperature_k) if temperature_k else np.full_like(full['omega_rad_s'],np.inf)
    factors=1+2*np.exp(-x)/(-np.expm1(-x))
    covariance=np.einsum('iam,ibm,m->iab',full['displacements_m'],full['displacements_m'],factors)
    xa=constants.hbar*active_omega/(constants.k*temperature_k) if temperature_k else np.full(2,np.inf)
    active_factors=1+2*np.exp(-xa)/(-np.expm1(-xa))
    raw_covariance=covariance.copy()
    shared=covariance.reshape(cells,nb,3,3).mean(axis=0)
    covariance=np.tile(shared,(cells,1,1))
    symmetry_error=np.linalg.norm(raw_covariance-covariance)/np.linalg.norm(covariance)
    if symmetry_error>1e-10: raise ValueError('finite covariance violates cell translation symmetry')
    active_cov=np.einsum('iam,ibm,m->iab',displacement,displacement,active_factors)
    inactive=covariance-active_cov
    if np.min(np.linalg.eigvalsh(inactive)) < -1e-12*np.max(abs(covariance)):
        raise ValueError('inactive covariance accounting is not positive')
    gram=np.einsum('iam,ian->mn',active_vectors,active_vectors)
    if np.linalg.norm(gram-np.eye(2))>1e-12: raise ValueError('specimen active modes are not orthonormal')
    bath=FiniteOpticalBath(displacement,active_omega,inactive,
        np.asarray(repetitions)[:,None]*model.cell_m,full['positions_m'],labels)
    evidence=dict(active_vectors=active_vectors,active_covariance_m2=active_cov,full_covariance_m2=covariance,
                  inactive_covariance_m2=inactive,gram=gram,finite_omega_rad_s=full['omega_rad_s'],
                  finite_eigenvectors=full['eigenvectors'],finite_displacements_m=full['displacements_m'],
                  finite_thermal_factors=factors,active_thermal_factors=active_factors,
                  omitted_rigid_translations=3,temperature_k=temperature_k,repetitions=np.array(repetitions),
                  finite_local_covariance_raw_m2=raw_covariance, covariance_translation_relative=symmetry_error,
                  boundary_conditions='periodic commensurate '+ 'x'.join(map(str,repetitions)) +' nuclear bath; transverse periodic electron cell, z vacuum padded',
                  covariance_accounting=f'full {3*len(masses)-3}-mode local quantum covariance minus the two normalized active contributions; inactive average once',
                  two_phonon_basis_note='Individual 20/11/02 populations depend on the chosen basis within an exactly degenerate subspace; retain their sum')
    return bath,evidence
