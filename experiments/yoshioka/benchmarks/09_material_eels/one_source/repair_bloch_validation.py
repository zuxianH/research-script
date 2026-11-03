"""Independent bath-basis, Born-normalization and full-cell propagation checks."""
import json
import numpy as np
from scipy import constants

from yoshioka.bloch_source import bath_groups,source_exit_bloch
from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
from yoshioka.electron_phonon import common_diagonal_covariance,LobatoSilicon,full_space_born,transverse_modes
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.depth_fields import incident_depth
from yoshioka.source_streaming import streamed_source_exit
from .workflow_io import ROOT,json_write


def basis_and_born_check(model,repetitions=(2,2,2)):
    old=finite_periodic_modes(model,repetitions);positions,groups=bath_groups(model,repetitions,2)
    covariance,_=common_diagonal_covariance(old['displacements_m']);mass=old['masses_kg'];N=len(mass);M=3*N-3
    new=[];frequencies=[]
    for group in groups:
        q=group['q_fractional'];parts=[]
        for ix,iy in np.ndindex(*repetitions[:2]):
            translation=np.array([ix,iy,0])@model.cell_m
            parts.append(group['displacements_m']*np.exp(1j*translation@(q@model.reciprocal_m_inv)))
        new.append(np.concatenate(parts,axis=0));frequencies.extend(group['omega_rad_s'])
    d=np.concatenate(new,axis=2);frequencies=np.array(frequencies)
    e=d*np.sqrt(2*mass[:,None,None]*frequencies[None,None]/constants.hbar)
    U=old['eigenvectors'].reshape(3*N,-1).conj().T@e.reshape(3*N,-1)
    gram=float(np.linalg.norm(U.conj().T@U-np.eye(M)))
    energy=float(np.linalg.norm(old['omega_rad_s'][:,None]*U-U*frequencies[None])/np.linalg.norm(frequencies))
    newcov,_=common_diagonal_covariance(d)
    cov_error=float(np.linalg.norm(newcov-covariance)/np.linalg.norm(covariance))
    energy_j=60000*constants.elementary_charge;grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],17,17)
    fullgrid=TransverseGrid(repetitions[0]*grid.lx_m,repetitions[1]*grid.ly_m,17*repetitions[0],17*repetitions[1]);Q,index,angle=transverse_modes(fullgrid,energy_j,20)
    pad=12e-10;length=2*model.cell_m[2,2]+2*pad;posold=old['positions_m'].copy();posold[:,2]+=pad
    potential=LobatoSilicon.from_abtem_table()
    reference=full_space_born(potential,posold,old['displacements_m'],covariance,Q,old['omega_rad_s'],energy_j,fullgrid.lx_m*fullgrid.ly_m,length)
    expected=U.conj().T@reference;actual=np.zeros_like(expected);offset=0
    from yoshioka.dynamical_source import full_transverse_vectors
    base=full_transverse_vectors(grid);positions[:,2]+=pad
    for group in groups:
        q=base+group['bloch_shift_m_inv'];select=np.linalg.norm(q,axis=1)/electron_wavenumber(energy_j)*1000<=20+1e-12;q=q[select]
        nearest=np.argmin(np.linalg.norm(q[:,None]-Q[None],axis=-1),axis=1)
        if np.max(np.linalg.norm(q-Q[nearest],axis=-1))>1e-4:raise ValueError('physical detector momenta differ')
        wave=full_space_born(potential,positions,group['displacements_m'],covariance,q,group['omega_rad_s'],energy_j,grid.lx_m*grid.ly_m,length)
        actual[offset:offset+len(wave),nearest]=wave;offset+=len(wave)
    error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    return {'mode_count':M,'repetitions':repetitions,'bath_basis_unitarity_error':gram,'energy_commutator_relative_error':energy,'covariance_relative_error':cov_error,
            'complex_Born_amplitude_relative_error':error,'maximum_absolute_Born_amplitude_error':float(np.max(abs(actual-expected))),
            'detector_probability_absolute_difference':float(abs(np.sum(abs(actual)**2)-np.sum(abs(expected)**2))),
            'passed':gram<1e-10 and energy<1e-10 and cov_error<1e-10 and error<1e-10,
            'phase_convention':'new eigenvectors=old eigenvectors@U; amplitudes transform asU†; no fitted phase/scale; atomic sum/area cancel under transverse replication'}


def full_cell_propagation_check(folder,repetitions=2,shift_fraction=(-.5,-.5)):
    folder.mkdir(parents=True,exist_ok=True);n=5;steps=33;length=17e-9;energy=60000*constants.elementary_charge
    rep=repetitions;grid=TransverseGrid(1e-9,1.2e-9,n,n);full=TransverseGrid(rep*1e-9,rep*1.2e-9,rep*n,rep*n)
    x,y=np.meshgrid(np.arange(n)*grid.dx_m,np.arange(n)*grid.dy_m)
    z=np.arange(steps);bg=(.3*np.cos(2*np.pi*x/grid.lx_m)+.2*np.sin(2*np.pi*y/grid.ly_m))*constants.elementary_charge
    background=np.broadcast_to(bg,(steps,1,n,n)).copy()*(1+.2*np.cos(2*np.pi*z/steps))[:,None,None,None]
    fullbg=np.tile(background,(1,1,rep,rep))
    np.save(folder/'background_unit.npy',background);np.save(folder/'background_full.npy',fullbg)
    background=np.load(folder/'background_unit.npy',mmap_mode='r');fullbg=np.load(folder/'background_full.npy',mmap_mode='r')
    _,incident,_=incident_depth(grid,energy,background,length,folder/'unit.npy')
    _,incident_full,_=incident_depth(full,energy,fullbg,length,folder/'full.npy')
    coupling=np.array([(.01+.02j)*np.cos(2*np.pi*x/grid.lx_m),(.03-.01j)*np.sin(2*np.pi*y/grid.ly_m)])*constants.elementary_charge
    shift=2*np.pi*np.array(shift_fraction)/np.array([grid.lx_m,grid.ly_m]);omega=np.array([.02,.06])*constants.elementary_charge/constants.hbar
    envelope=np.cos(4*np.pi*z/steps)+.3*np.sin(2*np.pi*z/steps)
    source=lambda i:coupling*envelope[i]
    a=source_exit_bloch(grid,energy,omega,length,steps,source,lambda i:incident[i],lambda i:background[i,0],shift)
    X,Y=np.meshgrid(np.arange(rep*n)*full.dx_m,np.arange(rep*n)*full.dy_m);phase=np.exp(1j*(shift[0]*X+shift[1]*Y))
    fullsource=lambda i:np.tile(source(i),(1,rep,rep))*phase
    b=streamed_source_exit(full,energy,omega,length,steps,fullsource,lambda i:incident_full[i],lambda i:fullbg[i,0])
    h=np.rint(np.fft.fftfreq(n)*n).astype(int);ix=(rep*h+int(round(rep*shift_fraction[0])))%(rep*n);iy=(rep*h+int(round(rep*shift_fraction[1])))%(rep*n)
    extracted=b[:,iy[:,None],ix[None,:]]
    error=float(np.linalg.norm(a-extracted)/np.linalg.norm(a))
    mask=np.ones((rep*n,rep*n),bool);mask[iy[:,None],ix[None,:]]=False
    return {'complex_amplitude_relative_error':error,'other_sector_weight':float(np.sum(abs(b[:,mask])**2)),
            'passed':error<1e-11,'repetitions':rep,'shift_fraction':shift_fraction,'reference':'independent full repeated-cell FFT propagation on a commensurate even grid, with the same complex source and physical momenta; arbitrary depth signs retain cancellations'}


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B9A_bloch_validation';output.mkdir(parents=True,exist_ok=True)
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    a=basis_and_born_check(model);b=full_cell_propagation_check(output)
    c=basis_and_born_check(model,(4,4,2));d=full_cell_propagation_check(output/'quarter',4,(.25,-.25))
    result={'status':'PASS' if all(r['passed'] for r in (a,b,c,d)) else 'FAIL','bath_and_normalization':a,'propagation':b,'quarter_bath_and_normalization':c,'quarter_sector_propagation':d}
    json_write(output/'results.json',result);print(result,flush=True)


if __name__=='__main__':main()
