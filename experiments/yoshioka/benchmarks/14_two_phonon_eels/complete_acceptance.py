"""All-observable statistical gates; a loss integral alone never closes EELS."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from .workflow_io import json_write,sha256
from .complete_leading_eels import DEFAULT,finite_json
from .two_phonon_analysis import WINDOWS
from .approximate_eels import DETECTORS


def evaluate(raw,comparisons,numerical_status='NOT_ESTABLISHED',relative_max=.1,absolute_floor=1e-18):
    windows={}
    for key in WINDOWS:
        row=raw.get(key);gates=[]
        for j,detector in enumerate(DETECTORS):
            m=None if row is None else row['mean'][j]
            h=None if row is None else row['halfwidth95'][j]
            finite=m is not None and h is not None and np.isfinite(m) and np.isfinite(h) and h>=0
            passed=bool(finite and (h<=absolute_floor if abs(m)<=absolute_floor else h/abs(m)<relative_max))
            gates.append({'detector':detector,'mean':m,'halfwidth95':h,'passed':passed,
                'criterion':'absolute declared floor' if finite and abs(m)<=absolute_floor else 'relative95 halfwidth<10percent'})
        windows[key]=gates
    all_windows=all(x['passed'] for rows in windows.values() for x in rows)
    loss=all(x['passed'] for x in windows['all_loss']);gain=all(x['passed'] for x in windows['all_gain'])
    shape=bool(comparisons and len(comparisons)==len(DETECTORS) and all(r['shape_pass'] for r in comparisons))
    features=bool(comparisons and len(comparisons)==len(DETECTORS) and all(r['all_peak_gate_pass'] for r in comparisons))
    numerical=numerical_status=='NUMERICALLY_CONVERGED_DECLARED_TARGET'
    return {'integrated_loss_all_detectors':loss,'integrated_gain_all_detectors':gain,
        'all_energy_windows_all_detectors':all_windows,'shape':shape,'features_identified_and_stable':features,
        'numerical_discretization':numerical_status,'statistical_gates_passed':bool(loss and gain and all_windows and shape and features),
        'declared_target_accepted':bool(loss and gain and all_windows and shape and features and numerical),
        'window_gates':windows,'status':'NOT_ACCEPTED' if not (loss and gain and all_windows and shape and features and numerical) else 'ACCEPTED_DECLARED_SCOPE'}


def paired_ratio(numerator,denominator):
    x=np.asarray(numerator);y=np.asarray(denominator);count=len(x)
    mx=x.mean(axis=0);my=y.mean(axis=0);ratio=mx/my
    centered=x-ratio*y
    sem=centered.std(axis=0,ddof=1)/(np.sqrt(count)*abs(my))
    covariance=np.sum((x-mx)*(y-my),axis=0)/(count-1)
    return {'ratio':ratio,'standard_error':sem,'halfwidth95':student_t.ppf(.975,count-1)*sem,
            'numerator_denominator_batch_covariance':covariance,
            'method':'Paired delta method across independent scramble integrals, retaining covariance; weak denominators remain uncertain.'}


def analyze(output,scheme):
    path=output/f'{scheme}_analysis.json';d=json.loads(path.read_text());latest=d['prefixes'][-1]
    gates=evaluate(latest['raw'],d['comparisons'])
    array=output/f"{scheme}_n{latest['pairs_per_batch']}_analysis.npz"
    with np.load(array) as z:
        ratio=paired_ratio(z['batch_weights_loss'],z['batch_sequential_weights_loss'])
        paired_difference=z['batch_weights_loss']-z['batch_sequential_weights_loss']
        difference={'mean':paired_difference.mean(axis=0),'halfwidth95':student_t.ppf(.975,len(paired_difference)-1)*paired_difference.std(axis=0,ddof=1)/np.sqrt(len(paired_difference))}
    result={'scheme':scheme,'pairs':latest['total_pairs'],'batches':d['batches'],'all_gates':gates,
            'coherent_over_sequential_loss':ratio,'coherent_minus_sequential_loss':difference,
            'analysis_sha256':sha256(path),'batch_arrays_sha256':sha256(array),'source_sha256':sha256(__file__),
            'numerical_scope':'Sampling gates do not establish electron, source, bath or interaction-order convergence.'}
    json_write(output/f'{scheme}_acceptance.json',finite_json(result));print(scheme,{k:v for k,v in gates.items() if k!='window_gates'},flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT/'bulk_static_v2');p.add_argument('--scheme',choices=('born','adapted','stratified'),required=True);a=p.parse_args();analyze(a.output,a.scheme)

if __name__=='__main__':main()
