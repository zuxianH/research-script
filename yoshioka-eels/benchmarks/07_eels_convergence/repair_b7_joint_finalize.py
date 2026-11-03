"""Evaluate the complete independent joint gate; never accept a partial prefix."""
import json
from pathlib import Path
import numpy as np
from .repair_convergence import summarize_arrays,compare_cases
from .workflow_io import ROOT,json_write


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2'
    progress=json.loads((output/'B7_joint/results.json').read_text());path=Path(progress['checkpoint'])
    state=json.loads(path.read_text());result={'status':'INCOMPLETE','completed_q':state['completed_q'],'total_q':state['total_q'],
        'partial_prefix_accepted':False,'rule':'two finest physical phonon comparisons plus complete independent numerical-source joint128³ check at256/4096/12Å; unchanged1% intensity/L1,.2meV peak and1e-4 source-amplitude targets',
        'baseline':'validated continuum Born accelerator; no numerical source discretization in that baseline. The complete joint numerical propagation is an independent route, not a second call to the accelerator.'}
    if state['completed_q']==state['total_q']:
        folder=path.parent;arrays={name:np.load(folder/(name+'.npy'),mmap_mode='r') for name in ('energies','weights','momentum','halves','forward','physical')}
        case,spectra=summarize_arrays(arrays);mesh=state['contract']['parameters']['mesh']
        case.update(settings={'mesh':mesh,'shift':[0,0,0],'route':'independent256/4096/12Å numerical source'},artifact=str(folder/'summary.npz'))
        np.savez_compressed(case['artifact'],energy_grid_mev=np.arange(-10,100.025,.05),**{f'spectrum_{name}':spectra[:,j] for j,name in enumerate(('full_0_20_mrad','central_0_5_mrad','annulus_5_10_mrad','annulus_10_20_mrad'))})
        physical=json.loads((output/'B7/results.json').read_text());fine=physical['cases'][-1];previous=physical['cases'][-2]
        same=compare_cases(fine,case,output);joint=compare_cases(previous,case,output)
        error=float(np.sqrt(state['squared_amplitude_error']/state['reference_squared_norm']))
        passed=physical['physical_quadrature_status']=='PASS' and same['physical_passed'] and joint['physical_passed'] and error<1e-4
        result.update(status='PASS' if passed else 'INCOMPLETE',same_phonon_joint=same,electron_and_phonon_joint=joint,
            global_complex_amplitude_relative_error=error,numerical_source_case=case,partial_prefix_accepted=False)
    json_write(output/'B7_joint_evaluation.json',result);print(result['status'],result['completed_q'],'/',result['total_q'])


if __name__=='__main__':main()
