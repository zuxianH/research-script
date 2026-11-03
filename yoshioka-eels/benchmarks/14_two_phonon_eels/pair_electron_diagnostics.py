"""Guarded frozen-pair electron diagnostics; no selection can certify the BZ."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.stats import t as student_t
from .workflow_io import ROOT,ResourceLimits,json_write,sha256,environment
from .complete_leading_eels import setup,physical_setup,KERNELS,finite_json
from .approximate_eels import PRIMITIVE,ORIENTED,DETECTORS
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.pair_galerkin import propagate_bulk_galerkin_pair,frequency_indices
from yoshioka.leading_two_phonon import propagate_bulk_leading_pair
from yoshioka.two_phonon import masks_for_shift

BASE=ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1'


def run_case(args):
    output=args.output/args.case;output.mkdir(parents=True,exist_ok=True)
    panel=json.loads(args.panel.read_text());panel=panel.get('pairs',panel.get('panel',panel))
    if isinstance(panel,dict):raise ValueError('panel must contain a pairs list')
    potential=LobatoSilicon.from_abtem_table()
    extra=('yoshioka/free_propagation.py','yoshioka/elastic_propagation.py',
       'yoshioka/detector_cells.py','yoshioka/depth_fields.py',
       'benchmarks/approximate_eels.py','benchmarks/two_phonon_eels.py',
       'benchmarks/two_phonon_analysis.py','benchmarks/born_quadrature.py','benchmarks/workflow_io.py')
    inputs=(PRIMITIVE,ORIENTED,ROOT/'Si.tersoff.mod',
       ROOT/'benchmark_results/approximate_eels_20261102_v1/covariance/T300_mesh32.json',
       ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1/bulk_static_v2/criteria.json')
    contract=dict(guard_version=4,entrypoint_sha256=sha256(__file__),retained=args.retained,product=args.product,steps=args.steps,
       local_substeps=args.substeps,born=args.born,legacy=args.legacy,
       require_dealiased=not args.circular,field_halfwidth=args.field_halfwidth,
       panel_sha256=sha256(args.panel),panel=str(args.panel.resolve()),
       kernels={p:sha256(ROOT/p) for p in (*KERNELS,*extra,'yoshioka/pair_galerkin.py')},
       input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in inputs},detectors=DETECTORS,
       lobato_file_sha256=sha256(potential.source_path),environment=environment(),
       fixed_physics='1.535553nm Si;300K;60keV;static Taylor;four historical masks;2meV FWHM;incident1',
       interpretation='Frozen diagnostic panel; selected nodes do not certify global BZ convergence')
    if args.planar:
        contract.update(representation='analytic regular longitudinal fields plus exact quadratic contact jumps',
            planar_kernels={p:sha256(ROOT/p) for p in ('yoshioka/pair_planar_fields.py','yoshioka/pair_planar_propagation.py')})
        if args.cell_average:
            contract['representation']='analytic exact interval means plus quadratic contact jumps; controlled time ordering'
            contract['planar_kernels']['yoshioka/pair_planar_averages.py']=sha256(ROOT/'yoshioka/pair_planar_averages.py')
    if args.background_only:
        contract['setup_strategy']='omit unused incident trajectory; analytic on demand or same streamed static depth field'
        contract['background_setup_sha256']=sha256(ROOT/'benchmarks/14_two_phonon_eels/pair_background_setup.py')
    # JSON restores tuple constants as lists. Canonicalize before comparing
    # so an unchanged cache resumes, while every physical value stays pinned.
    contract=finite_json(contract)
    statefile=output/'state.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(contract=contract,files={},completed_pairs=0)
    if state['contract']!=contract:raise ValueError('changed case contract; preserve this case and use a new identifier')
    for name,digest in state['files'].items():
        if sha256(output/name)!=digest:raise ValueError('changed completed output')
    setup(output)
    setup_hashes={k:sha256(output/k) for k in ('config.json','criteria.json','covariance/T300_mesh32.json','proposal_mass.npy')}
    if state.get('setup_sha256') is not None and state['setup_sha256']!=setup_hashes:
        raise ValueError('changed local physical setup')
    state['setup_sha256']=setup_hashes
    before=time.monotonic()
    if args.background_only:
        from .pair_background_setup import background_only_setup
        config,settings,data=background_only_setup(output,args.product,args.steps,args.planar)
    else:config,settings,data=physical_setup(output,args.product,args.steps)
    positions,reference,length,cov,product,bg,inc,_=data
    grid=TransverseGrid(product.lx_m,product.ly_m,args.retained,args.retained)
    k=electron_wavenumber(60000*constants.elementary_charge)
    qxy=full_transverse_vectors(grid)
    settings_record=dict(cell_lengths_angstrom=np.array([grid.lx_m,grid.ly_m])*1e10,
       real_product_spacing_angstrom=np.array([product.dx_m,product.dy_m])*1e10,
       reciprocal_spacing_inverse_angstrom=2*np.pi/np.array([grid.lx_m,grid.ly_m])*1e-10,
       retained_halfwidth_integer=(args.retained-1)//2,
       retained_axis_cutoff_inverse_angstrom=np.max(abs(qxy),axis=0)*1e-10,
       retained_axis_cutoff_mrad=np.max(abs(qxy),axis=0)/k*1000,
       depth_step_angstrom=length/args.steps*1e10,
       product_preparation_seconds=time.monotonic()-before,
       fields=('Analytic periodic Lobato regular profiles and exact V2zz contacts' if args.planar else
               'Analytic Lobato Fourier coefficients, full longitudinal displacement and midpoint z FFT'),
       antialiasing=contract['require_dealiased'],incident_normalization=1,
       runner_creation_sha256=sha256(__file__))
    numerical_path=output/'numerical_settings.json'
    if numerical_path.exists():
        if state.get('numerical_settings_sha256')!=sha256(numerical_path):
            raise ValueError('changed numerical settings record')
    else:
        json_write(numerical_path,finite_json(settings_record))
        state['numerical_settings_sha256']=sha256(numerical_path)
    json_write(statefile,state)
    model=RealSpaceForceConstants.load(PRIMITIVE);limits=ResourceLimits(args.seconds,4)
    new=0;status='COMPLETE';reason=None
    try:
        for j,row in enumerate(panel):
            path=output/f'pair{j:03d}.npz'
            if path.name in state['files']:continue
            if args.max_new is not None and new>=args.max_new:
                status='BOUNDED_PREFIX';break
            limits.check()
            source=Path(row.get('source_path',row.get('path','')))
            if not source.is_absolute():source=ROOT/source
            if row.get('source_sha256') and sha256(source)!=row['source_sha256']:raise ValueError('changed frozen pair')
            with np.load(source) as z:q=z['q_fractional'];b=z['branches'];iw=z['inverse_marginals']
            before=time.monotonic()
            if args.legacy:
                if args.retained!=args.product:raise ValueError('legacy requires same wave/product grid')
                result=propagate_bulk_leading_pair(model,positions,reference,cov,grid,length,args.steps,q,b,iw,
                      np.zeros_like(bg) if args.born else bg,output/'scratch',300,limits,save_amplitudes=True,workers=1,reference_kind='static')
            else:
                if args.planar:
                    from yoshioka.pair_planar_propagation import propagate_bulk_planar_pair
                    propagator=propagate_bulk_planar_pair
                else:propagator=propagate_bulk_galerkin_pair
                kw={'cell_average':args.cell_average} if args.planar else {}
                result=propagator(model,positions,reference,cov,grid,product,length,args.steps,q,b,iw,
                      bg,output/'scratch',300,limits,args.substeps,args.born,True,not args.circular,args.field_halfwidth,**kw)
            result.update(q_fractional=q,branches=b,inverse_marginals=iw,source_path=str(source),source_sha256=sha256(source),
                    measured_wall_seconds=time.monotonic()-before,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**result);temp.replace(path)
            state['files'][path.name]=sha256(path);state['completed_pairs']=len(state['files']);new+=1
            json_write(statefile,state);print(args.case,j,float(result['elapsed_seconds']),flush=True)
    except (TimeoutError,MemoryError) as error:status='PARTIAL_RESOURCE_LIMIT';reason=str(error)
    finally:
        if bg is not None:bg._mmap.close()
        if inc is not None:inc._mmap.close()
    json_write(output/f'invocation_{time.time_ns()}.json',dict(status=status,reason=reason,new_pairs=new,
       completed_pairs=state['completed_pairs'],requested_pairs=len(panel),**limits.costs()))
    print(args.case,status,state['completed_pairs'],flush=True)


def compare_case(a,b,output):
    sa=json.loads((a/'state.json').read_text());sb=json.loads((b/'state.json').read_text())
    ca=sa['contract'];cb=sb['contract']
    same_panel=ca['panel_sha256']==cb['panel_sha256']
    n=ca['retained'];m=cb['retained'];common=min(n,m)
    ia=frequency_indices(common,n);ib=frequency_indices(common,m);rows=[];diff=[];fine=[]
    for p in sorted(a.glob('pair[0-9][0-9][0-9].npz')):
        other=b/p.name
        if not other.exists():continue
        if sa['files'].get(p.name)!=sha256(p) or sb['files'].get(p.name)!=sha256(other):
            raise ValueError('changed committed paired array')
        with np.load(p) as x,np.load(other) as y:
            for key in ('source_sha256','q_fractional','branches','inverse_marginals','energies_mev','omega_rad_s','final_shifts'):
                if not np.array_equal(x[key],y[key]):raise ValueError('unpaired '+key)
            na=json.loads((a/'numerical_settings.json').read_text());nb=json.loads((b/'numerical_settings.json').read_text())
            if na['cell_lengths_angstrom']!=nb['cell_lengths_angstrom']:raise ValueError('changed physical cell')
            lx,ly=np.array(na['cell_lengths_angstrom'])*1e-10
            ga=TransverseGrid(lx,ly,n,n);gb=TransverseGrid(lx,ly,m,m)
            fa=np.rint(np.fft.fftfreq(n)*n).astype(int);fb=np.rint(np.fft.fftfreq(m)*m).astype(int)
            labels_a=np.array([(yy,xx) for yy in fa for xx in fa]);labels_b=np.array([(yy,xx) for yy in fb for xx in fb])
            mask_equal=True
            for shift in x['final_shifts']:
                ma=masks_for_shift(ga,shift);mb=masks_for_shift(gb,shift)
                for detector in range(4):
                    if set(map(tuple,labels_a[ma[detector]]))!=set(map(tuple,labels_b[mb[detector]])):
                        mask_equal=False
            components={};total=[]
            for key in ('direct_amplitudes','sequential_amplitudes','total_amplitudes'):
                aa=x[key] if key in x else x['direct_amplitudes']+x['sequential_amplitudes']
                bb=y[key] if key in y else y['direct_amplitudes']+y['sequential_amplitudes']
                ac=aa[...,ia[:,None],ia];bc=bb[...,ib[:,None],ib]
                components[key]=dict(common_absolute_l2=float(np.linalg.norm(bc-ac)),
                    common_relative_l2=float(np.linalg.norm(bc-ac)/max(np.linalg.norm(bc),1e-300)),
                    added_coarse_l2=float(np.sqrt(max(0,np.linalg.norm(aa)**2-np.linalg.norm(ac)**2))),
                    added_fine_l2=float(np.sqrt(max(0,np.linalg.norm(bb)**2-np.linalg.norm(bc)**2))))
                if key=='total_amplitudes':total=[ac,bc]
            d=y['direct_weights'];s=y['sequential_weights'];w=y['weights']
            lossx=np.sum(x['weights']*(x['energies_mev']>0)[:,None],axis=0)
            lossy=np.sum(w*(y['energies_mev']>0)[:,None],axis=0)
            diff.append(lossy-lossx);fine.append(lossy)
            rows.append(dict(pair=p.stem,components=components,
                absolute_intensity_difference=w-x['weights'],relative_intensity_difference=(w-x['weights'])/np.maximum(abs(w),1e-300),
                cancellation_condition=np.sqrt(np.maximum(d+s,0)/np.maximum(w,1e-300)),
                same_physical_detector_labels=mask_equal,
                same_physical_detector_test='executed integer-label comparison; no interpolated phase/scale fit',
                raw_a_sha256=sha256(p),raw_b_sha256=sha256(other)))
    delta=np.array(diff);f=np.array(fine)
    if not rows:raise ValueError('no complete hash-verified matched pair outputs')
    stats=dict(pairs=len(rows),mean_loss_change=delta.mean(axis=0),mean_fine_loss=f.mean(axis=0),
        relative_mean_loss_change=abs(delta.mean(axis=0))/np.maximum(abs(f.mean(axis=0)),1e-300),
        paired_mean_change_halfwidth95=student_t.ppf(.975,len(rows)-1)*delta.std(axis=0,ddof=1)/np.sqrt(len(rows)) if len(rows)>1 else None,
        scope='Descriptive for selected diagnostic nodes; only independently sampled representative replicas permit statistical inference')
    result=finite_json(dict(coarse=str(a),fine=str(b),rows=rows,stats=stats,identical_full_panel=same_panel,
        pairing='Every matched raw source hash, q, branch, inverse weight, energy and Bloch shift verified; unmatched prefixes excluded',
        source_sha256=sha256(__file__)))
    json_write(output,result);print(result['stats'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=BASE/'electron_cases');p.add_argument('--case')
    p.add_argument('--panel',type=Path);p.add_argument('--retained',type=int,default=17);p.add_argument('--product',type=int,default=35)
    p.add_argument('--steps',type=int,default=769);p.add_argument('--substeps',type=int,default=1);p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--max-new',type=int);p.add_argument('--born',action='store_true');p.add_argument('--legacy',action='store_true')
    p.add_argument('--circular',action='store_true');p.add_argument('--field-halfwidth',type=int);p.add_argument('--compare',nargs=2,type=Path);p.add_argument('--comparison-output',type=Path)
    p.add_argument('--planar',action='store_true',help='analytic regular longitudinal fields and exact quadratic contact jumps')
    p.add_argument('--cell-average',action='store_true',help='exact regular interval means; requires --planar')
    p.add_argument('--background-only',action='store_true',help='stream required background; skip unused separate incident trajectory')
    a=p.parse_args()
    if a.planar and (a.legacy or a.circular or a.field_halfwidth is not None):
        p.error('analytic contact propagation requires dealiased full-support Galerkin action')
    if a.cell_average and not a.planar:p.error('cell-average requires planar')
    if a.compare:compare_case(*a.compare,a.comparison_output)
    else:
        if not a.case or not a.panel:p.error('case and panel required')
        run_case(a)


if __name__=='__main__':main()
