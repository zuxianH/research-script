"""Diagnose longitudinal coherent sampling using completed 9A amplitudes."""

import json
from pathlib import Path

import numpy as np
from scipy import constants

from yoshioka.free_propagation import electron_wavenumber
from yoshioka.lattice_dynamics import RealSpaceForceConstants, finite_periodic_modes
from .workflow_io import ROOT, json_write, sha256


def main():
    output = ROOT/'benchmark_results/phonon_bath_audit_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT/'benchmark_results/dynamical_si_20261102_v1/results.json'
    source = json.loads(path.read_text())
    model = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    result = {'status':'INCOMPLETE', 'source_results_sha256':sha256(path), 'cases':[],
        'scope':'longitudinal phonon sampling at fixed 52-repeat atom slab, not a change of specimen thickness',
        'projection':'sum_Q A(Q)^dagger P_qz A(Q), with P_qz obtained from mass-weighted eigenvectors; retain coherent basis rotation before taking probabilities',
        'interpretation':'A thin reciprocal interference feature requires qz integration on a scale of order 2pi/L. Coarse discrete bath nodes can overweight its central sample; this audit measures the effect rather than treating a finite-q sample as a continuum point mass.'}
    for nz in (2,4):
        case = next(c for c in source['cases'] if c['status']=='COMPLETE' and c['settings']['thickness_repeats']==52
            and c['settings']['temperature_k']==0 and c['settings']['phonon_z']==nz)
        modes = finite_periodic_modes(model,(2,2,nz))
        m = len(modes['omega_rad_s'])
        vectors = modes['eigenvectors'].reshape(2,2,nz,4,3,m)
        transformed = np.fft.fft(vectors,axis=2)/np.sqrt(nz)
        with np.load(case['artifact']) as data:
            angle = np.linalg.norm(data['qxy_m_inv'],axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000
        amplitudes = np.load(Path(case['checkpoint']).parent/'loss.npy',mmap_mode='r').reshape(m,-1)[:,angle<=20+1e-12]
        projectors = []
        for index in range(nz):
            block = transformed[:,:,index].reshape(-1,m)
            projectors.append(block.conj().T@block)
        decomposition = []
        for numerator in range(nz//2+1):
            indices = [numerator] if numerator in (0,nz//2) else [numerator,nz-numerator]
            projector = sum(projectors[i] for i in indices)
            probability = float(np.einsum('mb,mn,nb->',amplitudes.conj(),projector,amplitudes,optimize=True).real)
            decomposition.append({'absolute_qz_fraction':numerator/nz,'inelastic_weight_20mrad':probability})
        omega = modes['omega_rad_s']
        commutation = max(np.linalg.norm(p*(omega[:,None]-omega[None,:]))/np.linalg.norm(omega) for p in projectors)
        total = float(np.sum(abs(amplitudes)**2))
        result['cases'].append({'phonon_z':nz,'mode_count':m,'actual_thickness_nm':case['thickness_nm'],
            'qz_spacing_m_inv':float(2*np.pi/(nz*model.cell_m[2,2])),
            'slab_interference_first_zero_scale_m_inv':float(2*np.pi/(52*model.cell_m[2,2])),
            'spacing_over_interference_scale':52/nz,'qz_components':decomposition,
            'full_weight_20mrad':total,'projector_partition_error':float(np.linalg.norm(sum(projectors)-np.eye(m))),
            'mode_energy_projector_commutator':float(commutation),
            'decomposed_weight_absolute_error':abs(sum(d['inelastic_weight_20mrad'] for d in decomposition)-total),
            'central_qz_weight_times_bath_z':nz*decomposition[0]['inelastic_weight_20mrad']})
        if result['cases'][-1]['projector_partition_error'] > 1e-10 or commutation > 1e-10 or result['cases'][-1]['decomposed_weight_absolute_error'] > 1e-12:
            raise ValueError('phonon momentum decomposition failed its independent partition/energy checks')
    json_write(output/'results.json',result)


if __name__=='__main__':
    main()
