"""Audit coordinate quadrature at the largest computed reduced Si basis.

Compare the actual interaction action (with identical diagonal terms removed),
and selected nonlinear phase blocks against an independent Laguerre formula.
This diagnoses quadrature error; it is not a material-convergence acceptance.
"""
import argparse,json,time
from pathlib import Path
from math import factorial
import numpy as np
from scipy.special import eval_genlaguerre
from .workflow_io import ROOT,ResourceLimits,json_write
from yoshioka.coordinate_channel_action import CoordinateMaterialAction
from yoshioka.lattice_dynamics import RealSpaceForceConstants

OUT=ROOT/'benchmark_results/repair_20261102_v2/B9B_order_audit'


def exact_block(xi,p):
    block=np.empty((p+1,p+1),complex)
    for m in range(p+1):
        for n in range(p+1):
            lo=min(m,n);hi=max(m,n)
            block[m,n]=np.exp(-.5*xi*xi)*np.sqrt(factorial(lo)/factorial(hi))*(-1j*xi)**(hi-lo)*eval_genlaguerre(lo,hi-lo,xi*xi)
    return block


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--order',type=int,default=24);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);limits=ResourceLimits(300,2)
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    h=CoordinateMaterialAction(model,4,65,257,quadrature_order=args.order,maximum_bytes=1800*1024**2)
    fine=json.loads((OUT.parent/'B9B_fine/results.json').read_text())
    with np.load(fine['artifact']) as a:actual=a['wave']
    rng=np.random.default_rng(57823);random=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape);random/=np.linalg.norm(random)
    indices=[104,113,124,135,145];actions=[]
    diagonal=h.channel_j[:,None,None]+h.kinetic_j[None]
    for index in indices:
        limits.check()
        actions.append(np.array([h.apply(w,index)-(diagonal+h.scalar[index])*w for w in (actual,random)]))
    selected=[];frequencies=np.rint(np.fft.fftfreq(h.pad)*h.pad).astype(int)
    for atom in (0,1):
        for x,y in ((0,0),(1,1),(8,16),(32,32),(64,64)):
            ix=int(np.flatnonzero(frequencies==x)[0]);iy=int(np.flatnonzero(frequencies==y)[0])
            phase=h.phase_x[atom,:,ix]*h.phase_y[atom,:,iy]
            numerical=h.overlap.T@(phase[:,None]*h.overlap)
            q=np.array([2*np.pi*x/model.cell_m[0,0],2*np.pi*y/model.cell_m[1,1],0.])
            xi=q@h.d[atom];blocks=[exact_block(value,4) for value in xi]
            reference=np.ones_like(numerical)
            for m in range(2):reference*=blocks[m][h.basis[:,m,None],h.basis[None,:,m]]
            selected.append({'atom':atom,'reciprocal_difference':[x,y],'xi':xi.tolist(),
                'maximum_absolute_block_error':float(np.max(abs(numerical-reference))),
                'relative_block_error':float(np.linalg.norm(numerical-reference)/max(np.linalg.norm(reference),1e-300))})
    artifact=OUT/f'action_order{args.order}.npz';np.savez_compressed(artifact,actions=np.array(actions),indices=indices)
    row={'order':args.order,'settings':fine['settings'],'artifact':str(artifact),'selected_phase_blocks':selected,
        'resource_costs':limits.costs(),'scope':'largest computed electron beam range; actual exit and normalized random states; diagonal kinetic/channel/scalar removed; sampled actions do not certify full propagation'}
    json_write(OUT/f'order{args.order}.json',row)
    other=OUT/'action_order24.npz'
    if args.order!=24 and other.exists():
        with np.load(other) as a,np.load(artifact) as b:
            delta=b['actions']-a['actions'];reference=b['actions']
            row['order24_to_fine_action_relative_errors']=[float(np.linalg.norm(delta[:,i])/max(np.linalg.norm(reference[:,i]),1e-300)) for i in range(2)]
        json_write(OUT/'comparison.json',row)
    print(json.dumps(row,indent=2))


if __name__=='__main__':main()
