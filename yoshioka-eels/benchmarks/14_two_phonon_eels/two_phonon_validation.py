"""Independent two-mode Fock propagation and actual-q source controls."""
import argparse
from pathlib import Path

import numpy as np
from scipy import constants, fft
from scipy.linalg import expm

from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.thermal_dyson_words import propagate_words
from .approximate_eels_repeated import dense_thermal_control
from .workflow_io import json_write


def dense_two_mode_control():
    """Independent finite Fock p=2 and dense augmented exponential, vacuum.

    Two distinct nonzero Bloch shifts, varying noncommuting electron/source
    fields, and actual kinetic splitting are identical in both references.
    The dense reference does not call the thermal-word local action.
    """
    grid=TransverseGrid(1e-9,1.3e-9,3,3)
    energy=60000*constants.elementary_charge
    length=1.5e-9; steps=3; dz=length/steps
    hv=constants.hbar*electron_speed(energy)
    omega=np.array([.031,.067])*constants.elementary_charge/constants.hbar
    shifts=np.array([[.37,-.19],[-.23,.41]])*(2*np.pi/np.array([grid.lx_m,grid.ly_m]))
    rng=np.random.default_rng(841310)
    sources=(rng.normal(size=(steps,2,3,3))+1j*rng.normal(size=(steps,2,3,3)))*constants.elementary_charge
    background=rng.normal(size=(steps,3,3))*constants.elementary_charge
    words,actual,_,_=propagate_words(grid,omega,shifts,length,steps,lambda i:sources[i],lambda i:background[i])
    states=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
    lookup={s:i for i,s in enumerate(states)}; nm=len(states)
    raising=[]
    for m in range(2):
        matrix=np.zeros((nm,nm))
        for j,s in enumerate(states):
            final=list(s); final[m]+=1
            if tuple(final) in lookup: matrix[lookup[tuple(final)],j]=np.sqrt(s[m]+1)
        raising.append(matrix)
    state_array=np.array(states)
    energies=state_array@(constants.hbar*omega)
    q=full_transverse_vectors(grid)[None]+(state_array@shifts)[:,None]
    free=np.exp(-.5j*dz*np.sum(q*q,axis=-1).reshape(nm,3,3)/(2*electron_wavenumber(energy)))
    reference=np.zeros((3,nm,3,3),complex)
    reference[0,0]=1/np.sqrt(grid.lx_m*grid.ly_m)
    for z in range(steps):
        reference=fft.ifft2(fft.fft2(reference,axes=(-2,-1))*free[None],axes=(-2,-1))
        diagonal=np.diag(np.repeat(energies,9)+np.tile(background[z].ravel(),nm))
        interaction=np.zeros_like(diagonal,dtype=complex)
        for m in range(2):
            interaction+=np.kron(raising[m],np.diag(sources[z,m].ravel()))
            interaction+=np.kron(raising[m].T,np.diag(sources[z,m].conj().ravel()))
        augmented=np.kron(np.eye(3),diagonal)+np.kron(np.diag(np.ones(2),-1),interaction)
        reference=(expm(-1j*dz/hv*augmented)@reference.ravel()).reshape(3,nm,3,3)
        reference=fft.ifft2(fft.fft2(reference,axes=(-2,-1))*free[None],axes=(-2,-1))
    expected=fft.fft2(reference,axes=(-2,-1))*np.sqrt(grid.lx_m*grid.ly_m)/9
    reconstructed=np.zeros_like(expected)
    for word,amplitude in zip(words,actual):
        bath=np.eye(nm)
        for letter in word:
            bath=bath@(raising[letter] if letter<2 else raising[letter-2].T)
        reconstructed[len(word)]+=bath[:,0,None,None]*amplitude[None]
    error=float(np.linalg.norm(reconstructed-expected))
    maximum=float(np.max(abs(reconstructed-expected)))
    relative=float(np.linalg.norm(reconstructed[2]-expected[2])/np.linalg.norm(expected[2]))
    return dict(complex_vector_error=error,maximum_absolute_error=maximum,
        two_insertion_relative_error=relative,passed=bool(error<1e-10 and maximum<1e-10),
        phases_aligned=False,reference='independent dense p=2 Fock plus order-augmented exponential',
        modes=2,electron_grid=3,steps=3,emission_bloch_shifts_m_inv=shifts.tolist())


def material_controls(output):
    import json
    from yoshioka.bulk_thermal_source import mode_group
    from yoshioka.bloch_source_windowed import source_exit_bloch
    from yoshioka.fourier_depth_fields import complex_depth_field,fourier_depth_provider
    from yoshioka.lattice_dynamics import RealSpaceForceConstants
    from yoshioka.two_phonon import propagate_bulk_pair,pair_nodes
    from .two_phonon_eels import ready_background,PRIMITIVE
    from .workflow_io import ResourceLimits
    config=json.loads((output/'config.json').read_text())
    settings={'temperature_k':300,'grid_unit':49,'slices':385}
    limits=ResourceLimits(600,2)
    positions,reference,length,covariance,grid,background,incident,_=ready_background(config,settings,output,limits)
    model=RealSpaceForceConstants.load(PRIMITIVE)
    q,branch,_=pair_nodes(0,729001)
    scenarios=[('production_node',q[0],branch[0]),
        ('small_q',np.array([[.0001,-.0002,.0003],[.11,-.23,.17]]),np.array([0,4])),
        ('opposite_q',np.array([[.17,-.21,.13],[-.17,.21,-.13]]),np.array([1,2]))]
    rows=[]
    for name,nodes,branches in scenarios:
        base=propagate_bulk_pair(model,positions,reference,covariance,grid,length,385,nodes,branches,[1,1],
            background,output/'validation_scratch',300,limits,save_amplitudes=True)
        scaled=propagate_bulk_pair(model,positions,reference,covariance,grid,length,385,nodes,branches,[1,1],
            background,output/'validation_scratch',300,limits,coupling_scale=.5,save_amplitudes=True)
        words=[()]+[(k,) for k in range(4)]+[(a,b) for a in range(4) for b in range(4)]
        factors=np.array([.5**len(w) for w in words])[:,None,None]
        scaling=float(np.linalg.norm(scaled['word_coefficients']-factors*base['word_coefficients']))
        first_errors=[]
        for side in range(2):
            omega,d,shift,labels=mode_group(model,positions,[grid.lx_m,grid.ly_m],nodes[side],1.,reference)
            at=int(np.flatnonzero(labels==branches[side])[0])
            field=complex_depth_field(positions,covariance,grid,length,385,output/'validation_source.npy',d[:,:,at:at+1],shift)
            provider=fourier_depth_provider(field,grid)
            for sign in (1,-1):
                source=provider if sign==1 else lambda i:provider(i).conj()
                expected=source_exit_bloch(grid,60000*constants.elementary_charge,sign*omega[at:at+1],length,385,
                    source,lambda i:np.asarray(incident[i]),lambda i:np.asarray(background[i,0]),sign*shift)
                actual=base['word_coefficients'][words.index((side if sign==1 else side+2,))].ravel()
                error=float(np.linalg.norm(actual-expected.ravel()))
                relative=error/max(float(np.linalg.norm(expected)),1e-100)
                first_errors.append({'mode':side,'sign':sign,'complex_error':error,'relative_error':relative})
            del provider
            field._mmap.close();(output/'validation_source.npy').unlink()
        g4=float(np.max(abs(scaled['weights']-base['weights']/16))/max(np.max(abs(base['weights']/16)),1e-100))
        rows.append(dict(name=name,q_fractional=nodes,branches=branches,first_order_errors=first_errors,
            g_scaling_complex_vector_error=scaling,g_four_intensity_relative_error=g4,
            all_beam_order2_balance=float(base['all_beam_order2_balance']),
            passed=bool(scaling<1e-10 and g4<1e-10 and max(x['relative_error'] for x in first_errors)<1e-10)))
        np.savez_compressed(output/f'validation_{name}.npz',q_fractional=nodes,branches=branches,
            coefficients=base['word_coefficients'],half_coupling_coefficients=scaled['word_coefficients'],
            word_shifts=base['word_shifts'],word_energies_j=base['word_energies_j'])
    result={'rows':rows,'resources':limits.costs(),'passed':all(r['passed'] for r in rows)}
    json_write(output/'material_source_validation.json',result)
    if not result['passed']:raise RuntimeError('actual-q source validation failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('dense','material'),default='dense')
    args=p.parse_args()
    if args.stage=='material':
        print(material_controls(args.output),flush=True)
        return
    rows={'two_mode_nonzero_Bloch_dense':dense_two_mode_control(),
          'finite_temperature_dense':dense_thermal_control(300)}
    json_write(args.output/'independent_validation.json',rows)
    print(rows,flush=True)
    if not all(row['passed'] for row in rows.values()): raise RuntimeError('dense validation failed')


if __name__=='__main__': main()
