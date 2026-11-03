"""Fixed-specimen bulk BZ quadrature with explicit Bloch channel boundaries.

The primitive two-atom eigenvectors use the Phonopy atomic gauge. A mode on a
uniform Nq mesh has d_i=sqrt(hbar/(2Mi omega)) e_basis(i) exp(iq.r_i)/sqrt(Nq).
The outgoing source has transverse Bloch phase exp(-iq_phonon.L), handled by
the existing numerical source integrator, never by incompatible periodic FFTs.
"""

import json
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.electron_phonon import FirstOrderSourcePropagator,LobatoSilicon,full_space_born,numerical_transition_source,transverse_modes
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.lattice_dynamics import RealSpaceForceConstants,real_space_from_supercell
from .eels_observables import DETECTORS,PEAK_GROUPS,absolute_spectrum,detector_masks
from .workflow_io import ROOT,fingerprint,json_write,sha256


def primitive_material(material_manifest,oriented_model,output):
    from hiphive import ForceConstantPotential
    path=Path(material_manifest)
    if not path.is_absolute():path=ROOT/path
    manifest=json.loads(path.read_text())
    fcp=ForceConstantPotential.read(str(path.parent/manifest['artifacts']['hiphive_potential']['path']))
    primitive=fcp.primitive_structure
    repetitions=(5,5,5)
    force_constants=fcp.get_force_constants(primitive.repeat(repetitions)).get_fc_array(2)
    model,metadata=real_space_from_supercell(primitive,repetitions,force_constants)
    model.save(output/'primitive_force_constants.npz')
    # This is an exact representation change of the fitted bulk model.
    ao=oriented_model.cell_m;ap=model.cell_m
    cosets=np.eye(3)@np.linalg.inv(ao).T@ap.T
    folded=next(q for q in cosets if not np.allclose(q-np.rint(q),0,atol=1e-10))
    primitive_e=sorted((model.modes(np.array([[0,0,0],folded]))[0]*constants.hbar/(1e-3*constants.elementary_charge)).ravel())
    oriented_e=oriented_model.modes([0,0,0])[0]*constants.hbar/(1e-3*constants.elementary_charge)
    error=float(np.max(abs(np.array(primitive_e)-oriented_e)))
    return model,{'primitive_cell_m':ap.tolist(),'primitive_positions_m':model.positions_m.tolist(),'primitive_reciprocal_m_inv':model.reciprocal_m_inv.tolist(),
                  'primitive_to_oriented_folding_energy_error_mev':error,'construction':'exact hiPhive model exported on a primitive supercell, not a new force fit',**metadata}


from yoshioka.lattice_dynamics import primitive_atom_mapping


def mesh_modes(primitive,mesh):
    indices=np.array(list(np.ndindex(mesh,mesh,mesh)),dtype=int)
    centered=indices.copy();centered[centered>mesh/2]-=mesh
    fractional=centered/mesh
    omega,vectors=primitive.modes(fractional)
    mask=np.ones(omega.shape,dtype=bool)
    t=np.tile(np.eye(3),(len(primitive.positions_m),1))*np.repeat(np.sqrt(primitive.masses_kg),3)[:,None]
    t/=np.linalg.norm(t,axis=0)
    overlaps=np.sum(abs(vectors[0].conj().T@t)**2,axis=-1)
    removed=np.argsort(overlaps)[-3:]
    mask[0,removed]=False
    if np.any(omega[mask]<=0):raise ValueError('a physical primitive phonon branch is unstable; no frequency clipping is allowed')
    return indices,fractional,omega,vectors,mask


def displacement_for_q(primitive,positions_m,basis,q_fractional,omega,eigenvectors,q_count):
    qcart=q_fractional@primitive.reciprocal_m_inv
    vectors=eigenvectors.reshape(len(primitive.positions_m),3,-1)[basis]
    factor=np.sqrt(constants.hbar/(2*primitive.masses_kg[basis,None]*omega[None]))/np.sqrt(q_count)
    return vectors*factor[:,None,:]*np.exp(1j*(positions_m@qcart))[:,None,None]


def checkpointed_case(primitive,positions_m,cell_m,electron_repetitions,covariance_m2,settings,output,limits,provenance,max_new_q=None):
    """Resume per-q longitudinal numerical propagation from atomic checkpoints."""
    mesh,n,samples,padding=map(settings.get,('mesh','grid','samples','padding_angstrom'))
    # Estimate eigensystem, retained line arrays, and source-kernel cache before
    # creating a potentially huge q mesh. No dense full-mode channel matrix.
    limits.check(estimated_bytes=mesh**3*4096+16*4*64*samples*16*2+200*1024**2)
    code_names=('yoshioka/electron_phonon.py','benchmarks/bulk_eels.py','benchmarks/eels_observables.py')
    digest=fingerprint({'settings':settings,'provenance':provenance,'code':{name:sha256(ROOT/name) for name in code_names},
                        'covariance_m2':covariance_m2.tolist(),'specimen_positions_m':positions_m.tolist()})[:20]
    directory=output/'cache'/digest;directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'state.json'
    q_indices,q_fractional,omega,vectors,physical=mesh_modes(primitive,mesh)
    q_count=len(q_fractional);branches=omega.shape[1]
    shapes={'energy_mev':(q_count,branches),'weights':(q_count,branches,len(DETECTORS)),'reference_weights':(q_count,branches,len(DETECTORS)),
            'momentum':(q_count,8,8),'reference_momentum':(q_count,8,8),'forward':(q_count,),'reference_forward':(q_count,),
            'half_regions':(q_count,2),'reference_half_regions':(q_count,2),'amplitude_errors':(q_count,),'reference_norms':(q_count,)}
    state=json.loads(state_path.read_text()) if state_path.exists() else {'digest':digest,'completed_q':0,'total_q':q_count,'elapsed_seconds':0,'status':'INCOMPLETE','settings':settings}
    arrays={}
    for name,shape in shapes.items():
        path=directory/f'{name}.npy'
        if path.exists():arrays[name]=np.lib.format.open_memmap(path,mode='r+')
        else:
            arrays[name]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=shape);arrays[name][:]=0
    if state['completed_q']==q_count:return summarize_case(directory,state,arrays,physical,settings),False
    limits.check(estimated_bytes=16*4*64*samples*16*2+200*1024**2)
    basis,_=primitive_atom_mapping(primitive,positions_m)
    grid=TransverseGrid(electron_repetitions[0]*cell_m[0,0],electron_repetitions[1]*cell_m[1,1],n,n)
    qbox=np.array([2*np.pi/grid.lx_m,2*np.pi/grid.ly_m])
    reciprocal_integer=primitive.reciprocal_m_inv[:,:2]/qbox[None]
    if not np.allclose(reciprocal_integer,np.rint(reciprocal_integer),atol=1e-9,rtol=0):raise ValueError('primitive/electron cell relation is not commensurate enough to define exact Bloch sectors')
    reciprocal_integer=np.rint(reciprocal_integer).astype(int)
    positions=positions_m.copy();positions[:,2]+=padding*1e-10
    thickness=electron_repetitions[2]*cell_m[2,2];length=thickness+2*padding*1e-10
    potential=LobatoSilicon.from_abtem_table();energy=60000*constants.elementary_charge
    kernel_cache={};started=time.monotonic();resumed_at=state['completed_q'];stop_reason=None
    try:
        for iq in range(state['completed_q'],q_count):
            limits.check()
            if max_new_q is not None and iq-resumed_at>=max_new_q:
                stop_reason='explicit test/checkpoint batch limit';break
            chosen=physical[iq]
            frequencies=omega[iq,chosen]
            d=displacement_for_q(primitive,positions_m,basis,q_fractional[iq],frequencies,vectors[iq][:,chosen],q_count)
            residue=(-q_indices[iq]@reciprocal_integer)%mesh
            residue[residue>mesh/2]-=mesh
            shift=residue/mesh*qbox
            qxy,indices,angles=transverse_modes(grid,energy,20,shift)
            _,source=numerical_transition_source(potential,positions,d,covariance_m2,qxy,grid,length,samples,kernel_cache)
            amplitudes=FirstOrderSourcePropagator(grid,energy,frequencies,indices,shift).propagate_scan(source,length)
            reference=full_space_born(potential,positions,d,covariance_m2,qxy,frequencies,energy,grid.lx_m*grid.ly_m,length)
            probability=abs(amplitudes)**2;ref_probability=abs(reference)**2
            masks=detector_masks(angles)
            arrays['energy_mev'][iq,chosen]=frequencies*constants.hbar/(1e-3*constants.elementary_charge)
            for j,mask in enumerate(masks.values()):
                arrays['weights'][iq,chosen,j]=np.sum(probability[:,mask],axis=1)
                arrays['reference_weights'][iq,chosen,j]=np.sum(ref_probability[:,mask],axis=1)
            electron_k=electron_wavenumber(energy)
            xy_mrad=np.clip(qxy/electron_k*1000,-20,20)  # only roundoff at the closed detector boundary
            histogram=np.histogram2d(xy_mrad[:,0],xy_mrad[:,1],bins=np.linspace(-20,20,9),weights=np.sum(probability,axis=0))[0]
            ref_histogram=np.histogram2d(xy_mrad[:,0],xy_mrad[:,1],bins=np.linspace(-20,20,9),weights=np.sum(ref_probability,axis=0))[0]
            arrays['momentum'][iq]=histogram;arrays['reference_momentum'][iq]=ref_histogram
            forward=np.linalg.norm(qxy,axis=1)<1e-5
            arrays['forward'][iq]=np.sum(probability[:,forward]);arrays['reference_forward'][iq]=np.sum(ref_probability[:,forward])
            for side,mask in enumerate((qxy[:,0]>=0,qxy[:,0]<0)):
                arrays['half_regions'][iq,side]=np.sum(probability[:,mask]);arrays['reference_half_regions'][iq,side]=np.sum(ref_probability[:,mask])
            arrays['amplitude_errors'][iq]=np.linalg.norm(amplitudes-reference)**2
            arrays['reference_norms'][iq]=np.linalg.norm(reference)**2
            state['completed_q']=iq+1
            if (iq+1)%64==0:
                for array in arrays.values():array.flush()
                state['elapsed_seconds']+=time.monotonic()-started;started=time.monotonic()
                json_write(state_path,state)
    except (TimeoutError,MemoryError) as exc:
        stop_reason=str(exc)
    finally:
        for array in arrays.values():array.flush()
        state['elapsed_seconds']+=time.monotonic()-started
        state['status']='COMPLETE' if state['completed_q']==q_count else 'INCOMPLETE'
        state['stop_reason']=stop_reason;state['resumed_from_q']=resumed_at
        state['settings']=settings
        json_write(state_path,state)
    if state['status']!='COMPLETE':return {'status':'INCOMPLETE','settings':settings,'cache_directory':str(directory.relative_to(output)),'completed_q':state['completed_q'],'total_q':q_count,'stop_reason':stop_reason,'elapsed_seconds':state['elapsed_seconds'],
                                        'partial_inelastic_weight_not_renormalized':float(np.sum(arrays['weights'][:state['completed_q'],:,0]))},True
    limits.completed+=1
    return summarize_case(directory,state,arrays,physical,settings),True


def summarize_case(directory,state,arrays,physical,settings):
    energies=np.asarray(arrays['energy_mev'])[physical]
    weights=np.asarray(arrays['weights'])[physical]
    reference_weights=np.asarray(arrays['reference_weights'])[physical]
    energy_grid=np.arange(-10,100+.025,.05)
    values={};reference_values={};spectra={};peaks={}
    for j,name in enumerate(DETECTORS):
        values[name]=float(np.sum(weights[:,j]));reference_values[name]=float(np.sum(reference_weights[:,j]))
        spectra[name]=absolute_spectrum(energies*1e-3*constants.elementary_charge,weights[:,j],energy_grid,2)
        for group,(low,high) in PEAK_GROUPS.items():
            selected=(energies>=low)&(energies<high)
            values[f'{name}/{group}']=float(np.sum(weights[selected,j]));reference_values[f'{name}/{group}']=float(np.sum(reference_weights[selected,j]))
            window=(energy_grid>=low)&(energy_grid<high)
            peaks[f'{name}/{group}']=float(energy_grid[window][np.argmax(spectra[name][window])]) if np.any(window) and values[f'{name}/{group}']>=1e-18 else None
    histogram=np.sum(arrays['momentum'],axis=0);reference_hist=np.sum(arrays['reference_momentum'],axis=0)
    for ix in range(8):
        for iy in range(8):
            values[f'momentum_pixel_{ix}_{iy}']=float(histogram[ix,iy]);reference_values[f'momentum_pixel_{ix}_{iy}']=float(reference_hist[ix,iy])
    for j,name in enumerate(('positive_qx_half','negative_qx_half')):
        values[name]=float(np.sum(arrays['half_regions'][:,j]));reference_values[name]=float(np.sum(arrays['reference_half_regions'][:,j]))
    values['exact_forward_q_zero']=float(np.sum(arrays['forward']));reference_values['exact_forward_q_zero']=float(np.sum(arrays['reference_forward']))
    path=directory/'summary_arrays.npz'
    np.savez_compressed(path,energies_mev=energies,mode_detector_weights=weights,reference_mode_detector_weights=reference_weights,
                        energy_grid_mev=energy_grid,momentum_histogram=histogram,reference_momentum_histogram=reference_hist,
                        **{f'spectrum_{name}':value for name,value in spectra.items()})
    return {'status':'COMPLETE','settings':settings,'completed_q':state['completed_q'],'q_points':state['total_q'],'physical_mode_count':int(np.sum(physical)),
            'q_weight':1/state['total_q'],'primitive_branches':6,'removed_rigid_translations':3,'elapsed_seconds':state['elapsed_seconds'],
            'resumed_from_q':state.get('resumed_from_q',0),'observables':values,'reference_observables':reference_values,'peak_positions_mev':peaks,
            'complex_amplitude_relative_error_vs_Born':float(np.sqrt(np.sum(arrays['amplitude_errors'])/np.sum(arrays['reference_norms']))),
            'artifact':str(path),'cache_directory':str(directory),'energy_grid_mev':energy_grid.tolist()}
