"""Cold largest-background and optional bounded source-sector memory pilots."""
import argparse,json,mmap,time,resource
from pathlib import Path
import numpy as np
from .workflow_io import ROOT,json_write,ResourceLimits
from yoshioka.fourier_depth_fields import depth_field
from yoshioka.bloch_source import bath_groups
from yoshioka.bloch_covariance import covariance_from_groups
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import TransverseGrid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-block',action='store_true')
    parser.add_argument('--config',default='benchmarks/configs/repair_9a_fourier_depth_joint_20261102_v2.json')
    parser.add_argument('--label',default='fourier')
    args=parser.parse_args();output=ROOT/'benchmark_results/repair_20261102_v2';started=time.monotonic()
    config=json.loads((ROOT/args.config).read_text())
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    if args.source_block:
        from .repair_9a_fourier_depth import calculate
        settings=config['cases'][0]
        cached_before=any((lambda state:state['contract']['settings']==settings and state.get('background_complete') and state.get('incident_complete'))(json.loads(path.read_text()))
            for path in (ROOT/config['output']).glob('cache/*/state.json'))
        class Pilot(ResourceLimits):
            calls=0
            def check(self,**kwargs):
                self.calls+=1
                if self.calls==3:raise TimeoutError('pilot complete: one newly committed source batch')
                return super().check(**kwargs)
        limits=Pilot(600,2)
        result=calculate(model,config['cases'][0],ROOT/config['output'],limits,config['mode_block_size'])
        row={'result':result,'resource_costs':limits.costs(),'cold_setup_included':not cached_before,
            'scope':'warm independent source batch; compatible background and incident reused' if cached_before else 'cold background, incident propagation and one independent source batch'}
        name=args.label+'_source_pilot.json'
    else:
        settings=config['cases'][0];n=settings['grid_unit'];steps=settings['slices'];repeats=settings['thickness_repeats']
        positions,groups=bath_groups(model,(2,2,2),repeats);positions[:,2]+=12e-10
        cov,_=covariance_from_groups(groups,4);grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],n,n)
        target=output/'cold_background_pilot.npy'
        actual,diag=depth_field(positions,cov,grid,repeats*model.cell_m[2,2]+24e-10,steps,target)
        construction_seconds=time.monotonic()-started
        old_path=next((output/'B9A_windowed_joint').glob('cache/*/background.npy'))
        expected=np.load(old_path,mmap_mode='r');squared_error=squared_norm=0.
        for lo in range(0,steps,16):
            a=np.asarray(actual[lo:lo+16]);b=np.asarray(expected[lo:lo+16])
            squared_error+=float(np.sum(abs(a-b)**2));squared_norm+=float(np.sum(abs(b)**2))
            del a,b
            for mapped in (actual,expected):mapped._mmap.madvise(mmap.MADV_DONTNEED)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        row={'settings':settings,'background_construction_seconds':construction_seconds,'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':peak,'limit_bytes':2*1024**3,'relative_field_error':float(np.sqrt(squared_error/squared_norm)),
            'reference':str(old_path),'diagnostic':diag,'status':'PASS' if peak<2*1024**3 and squared_error/squared_norm<1e-24 else 'FAIL'}
        del actual,expected;target.unlink();Path(str(target)+'.reality.json').unlink(missing_ok=True)
        name=args.label+'_background_pilot.json'
    json_write(output/name,row);print(json.dumps(row,indent=2),flush=True)


if __name__=='__main__':main()
