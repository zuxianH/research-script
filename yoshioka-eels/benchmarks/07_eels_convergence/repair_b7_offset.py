"""Same bulk measure, shifted cubature nodes; independent offset diagnostic."""
import argparse,json
import numpy as np
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .detector_cubature import checkpointed_cells
from .repair_convergence import summarize_arrays,compare_cases
from .workflow_io import ROOT,ResourceLimits,json_write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mesh',type=int,default=64);args=parser.parse_args();mesh=args.mesh
    output=ROOT/('benchmark_results/repair_20261102_v2/B7_offset' if mesh==64 else f'benchmark_results/repair_20261102_v2/B7_offset{mesh}');output.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'benchmark_results/detector_cells_20261102_v1/results.json').read_text())
    spec=json.loads((ROOT/'benchmark_results/si_eels_convergence/results.json').read_text())['model_specification']
    primitive=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')
    limits=ResourceLimits(1800,2,100)
    folder,state,arrays=checkpointed_cells(primitive,spec,mesh,[.5,.5,.5],output,limits)
    result={'status':state['status'],'checkpoint':str(folder/'state.json'),'resource_costs':limits.costs(),
            'interpretation':'continuum BZ quadrature offset; same infinite bulk phonons, fixed finite electron slab and Bloch boundaries; not a changed finite physical crystal'}
    if state['status']=='COMPLETE':
        case,spectra=summarize_arrays(arrays);case.update(settings={'mesh':mesh,'shift':[.5,.5,.5],'route':'Born with exact cell-detector overlaps'},artifact=str(folder/'summary.npz'))
        np.savez_compressed(case['artifact'],energy_grid_mev=np.arange(-10,100.025,.05),**{f'spectrum_{name}':spectra[:,j] for j,name in enumerate(('full_0_20_mrad','central_0_5_mrad','annulus_5_10_mrad','annulus_10_20_mrad'))})
        nominal=next(c for c in old['cases'] if c['settings']['mesh']==mesh)
        result.update(case=case,offset_comparison=compare_cases(nominal,case,output))
    json_write(output/'results.json',result)


if __name__=='__main__':main()
