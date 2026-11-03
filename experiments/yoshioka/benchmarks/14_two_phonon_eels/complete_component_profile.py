"""Separate bounded process profiles; peak RSS includes declared setup."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import constants
from .workflow_io import json_write,sha256
from .complete_leading_eels import DEFAULT,physical_setup,nodes
from .approximate_eels import PRIMITIVE
from .born_quadrature import line_spectra
from .two_phonon_analysis import AXIS
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.bulk_thermal_source import mode_group
from yoshioka.fourier_depth_fields import complex_depth_field,fourier_depth_provider
from yoshioka.depth_fields import incident_depth
from yoshioka.bloch_source_windowed import source_exit_bloch
from yoshioka.thermal_dyson_words import propagate_words
from yoshioka.leading_reference import analytic_phase_blocks,quadrature_phase_blocks
from yoshioka.coupled_channels import channel_basis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT/'bulk_static_v2');p.add_argument('--component',required=True,choices=('phonon','field','elastic','first_order','second_order','nonlinear','spectrum','io'));a=p.parse_args();output=a.output
    config,settings,data=physical_setup(output);positions,reference,length,cov,grid,bg,incident,_=data
    model=RealSpaceForceConstants.load(PRIMITIVE);q,branch,iw=nodes(output,'born',0,config['seed_start'])
    om,d,shift,labels=mode_group(model,positions,[grid.lx_m,grid.ly_m],q[0,0],1.,reference)
    selected=int(np.flatnonzero(labels==branch[0,0])[0]);disp=d[:,:,selected:selected+1]
    work=output/'component_profiles';work.mkdir(exist_ok=True);maps=[];extra={}
    if a.component in ('first_order','second_order'):
        field=complex_depth_field(positions,cov,grid,length,settings['slices'],work/f'{a.component}_source.npy',d,shift);maps.append(field)
        provider=fourier_depth_provider(field,grid)
    wall=time.monotonic();cpu=time.process_time()
    if a.component=='phonon':
        for i in range(64):mode_group(model,positions,[grid.lx_m,grid.ly_m],q[0,0]+i*1e-5,1.,reference)
        extra={'evaluations':64,'includes':'dynamical matrix eigensolve, physical branch filter, mode displacements/Bloch phase'}
    elif a.component=='field':
        field=complex_depth_field(positions,cov,grid,length,settings['slices'],work/'field.npy',disp,shift);maps.append(field)
        extra={'selected_modes':1,'bytes':int(field.nbytes),'separate_quadratic_cost':'See profile.json; evaluated same primary grid/steps'}
    elif a.component=='elastic':
        wave,history,diagnostic=incident_depth(grid,60000*constants.elementary_charge,bg,length,work/'elastic_history.npy');maps.append(history);extra=diagnostic
    elif a.component=='first_order':
        amp=source_exit_bloch(grid,60000*constants.elementary_charge,om,length,settings['slices'],provider,lambda i:np.asarray(incident[i]),lambda i:np.asarray(bg[i,0]),shift)
        extra={'physical_branches':len(om),'amplitude_norm':float(np.linalg.norm(amp)),'scope':'one emission source; all branches in this q; includes incoming cached distorted wave, outgoing diffraction'}
    elif a.component=='second_order':
        words,amp,_,_=propagate_words(grid,om[:2],np.tile(shift,(2,1)),length,settings['slices'],lambda i:provider(i)[:2],lambda i:np.asarray(bg[i,0]))
        extra={'words':len(words),'amplitude_norm':float(np.linalg.norm(amp)),'scope':'sequential linear hierarchy only; complete direct-plus-sequential costs in profile.json'}
    elif a.component=='nonlinear':
        basis=channel_basis(2,4);xi=np.column_stack((np.linspace(-.4,.4,256),np.linspace(.3,-.3,256)))
        exact=analytic_phase_blocks(basis,xi);analytic_seconds=time.monotonic()-wall
        quad=quadrature_phase_blocks(basis,xi,32)
        extra={'retained_cutoff':4,'working_quadrature':32,'evaluation_count':256,'analytic_seconds':analytic_seconds,
               'quadrature_seconds':time.monotonic()-wall-analytic_seconds,'relative_complex_agreement':float(np.linalg.norm(exact-quad)/np.linalg.norm(exact))}
    elif a.component=='spectrum':
        with np.load(output/'profile_pair0.npz') as d:e=d['energies_mev'];w=d['weights']
        for i in range(64):line_spectra(e,w,AXIS)
        extra={'line_lists':64,'four_signed_lines_per_list':True,'detectors':4,'fwhm_mev':2}
    else:
        with np.load(output/'profile_pair0.npz') as d:arrays={k:d[k] for k in d.files}
        path=work/'checkpoint_io.npz';np.savez_compressed(path,**arrays);checksum=sha256(path)
        with np.load(path) as d:
            for key,value in arrays.items():np.testing.assert_array_equal(d[key],value)
        extra={'bytes':path.stat().st_size,'sha256':checksum,'scope':'compressed atomic-pair payload write+SHA256+read+exact round trip; JSON commit costs separately in batch state'}
    result={'component':a.component,'timed_wall_seconds':time.monotonic()-wall,'timed_cpu_seconds':time.process_time()-cpu,
        'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'memory_scope':'Peak of fresh process including immutable cached background and stated source setup; not incremental allocation.','grid':grid.shape,'steps':settings['slices'],'geometry':config['selected_repeats'],'extra':extra,'script_sha256':sha256(__file__)}
    json_write(work/f'{a.component}.json',result);print(result,flush=True)
    for mapped in maps:mapped._mmap.close()
    for name in (f'{a.component}_source.npy','field.npy','elastic_history.npy'):(work/name).unlink(missing_ok=True)

if __name__=='__main__':main()
