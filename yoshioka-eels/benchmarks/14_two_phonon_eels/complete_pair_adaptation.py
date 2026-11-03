"""Freeze an independently assessed complete-amplitude proposal from old pilot.

No production samples are recycled for design. Each declared detector and
energy window contributes to a marginal score normalized by its pilot mean.
The learned histogram is smoothed periodically and mixed with the old Born
proposal. Uniform/radial support in BornPairProposal is additionally retained.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import t as student_t
from .workflow_io import ROOT,json_write,sha256
from .complete_leading_eels import DEFAULT
from yoshioka.phonon_pair_importance import BornPairProposal


def prepare(pilot,output):
    paths=sorted((pilot/'pairs/born').glob('batch*/pair*.npz'))
    if len(paths)!=64:raise ValueError('frozen independent pilot is exactly8 scrambles x8 pairs')
    energies=[];weights=[];nodes=[];branches=[];provenance={}
    for p in paths:
        with np.load(p) as d:
            energies.append(d['energies_mev']);weights.append(d['weights'])
            nodes.append(d['q_fractional']);branches.append(d['branches'])
        provenance[str(p.relative_to(ROOT))]=sha256(p)
    e=np.array(energies);w=np.array(weights)
    windows=[(5,40),(40,65),(65,90),(90,125),(125,160),(-40,-5),(-65,-40),(-90,-65),(-125,-90),(-160,-125)]
    observables=np.stack([np.sum(w*((e>=lo)&(e<hi))[...,None],axis=1) for lo,hi in windows],axis=1)
    means=observables.mean(axis=0);floor=float(np.max(means))*1e-5
    score=np.sum(observables/np.maximum(means,floor)[None],axis=(1,2))
    mesh=8;hist=np.zeros((mesh,mesh,mesh,6))
    q=np.array(nodes);b=np.array(branches)
    for i in range(len(q)):
        for side in range(2):
            idx=tuple(np.floor(q[i,side]*mesh+.5).astype(int)%mesh)+(int(b[i,side]),)
            hist[idx]+=score[i]/2
    for branch in range(6):hist[:,:,:,branch]=gaussian_filter(hist[:,:,:,branch],sigma=1.,mode='wrap')
    hist/=hist.sum()
    index=np.array(np.unravel_index(np.arange(64**3),(64,)*3)).T
    coarse=np.floor(index*mesh/64+.5).astype(int)%mesh
    mass=hist[tuple(coarse.T)]/((64//mesh)**3)
    old=np.load(output/'proposal_mass.npy');mass=.5*mass+.5*old;mass/=mass.sum()
    target=output/'adapted_mass.npy'
    if target.exists():
        if not np.array_equal(np.load(target),mass):raise ValueError('frozen learned density changed')
    else:np.save(target,mass)
    # Independent known-integral diagnostic across scrambles:∫sum_ba,bb(ba+1)(bb+1)=441.
    proposal=BornPairProposal(mass,64);known=[]
    for seed in range(3829001,3829009):
        q,b,iw=proposal.nodes(14,seed)
        known.append(float(np.mean((1+q[:,0,0]*q[:,1,2])*(b[:,0]+1)*(b[:,1]+1)*iw.prod(axis=1))))
    mean=float(np.mean(known));half=float(student_t.ppf(.975,7)*np.std(known,ddof=1)/np.sqrt(8))
    record=dict(pilot_pairs=64,pilot_files=provenance,learned_mass_sha256=sha256(target),old_mass_sha256=sha256(output/'proposal_mass.npy'),
        score='Equal declared energy-window and detector objectives, each divided by separate pilot mean with1e-5 global floor. Complete coherent F, not sequential-only intensity.',
        windows_mev=windows,coarse_mesh=mesh,periodic_gaussian_sigma_cells=1.,learned_histogram_fraction=.5,
        statistical_safety='Adapted production uses independent seeds2929001..2929008; pilot seeds1929001..1929008. No pilot samples enter its final assessment.',
        kernel_mixture={'uniform':.2,'radial':.4,'learned_Born_table_mixture':.4},
        known_integral=dict(target=441.,independent_batch_values=known,mean=mean,halfwidth95=half,error=mean-441.,passed=abs(mean-441)<max(5*half,.2)),
        source_sha256=sha256(__file__))
    json_write(output/'adaptation.json',record)
    if not record['known_integral']['passed']:raise RuntimeError('proposal known-integral check failed')
    print(record['known_integral'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,default=DEFAULT/'bulk_static');p.add_argument('--output',type=Path,default=DEFAULT/'bulk_static_v2');args=p.parse_args();prepare(args.pilot,args.output)

if __name__=='__main__':main()
