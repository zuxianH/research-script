"""Finite periodic harmonic bath and absolute all-direction Born screening.

The first Gaussian cumulant contraction is used: each atomic vertex carries
exp(-K C_full K/2) (-i K.d_mu). C_full includes every quantum mode once.
This is a single-inelastic-event approximation, not nonlinear coupled dynamics.
"""
import numpy as np
from scipy import constants as c


def thermal_occupation(omega, temperature):
    if temperature < 0: raise ValueError('nonnegative temperature required')
    return np.zeros_like(omega) if temperature == 0 else 1/np.expm1(c.hbar*np.asarray(omega)/(c.k*temperature))


def commensurate_bath(model, repetitions, temperature):
    reps=np.asarray(repetitions,int)
    cells=np.array(list(np.ndindex(*reps)))
    positions=((cells@model.cell_m)[:,None,:]+model.positions_m).reshape(-1,3)
    vectors=[];frequencies=[];labels=[];travel=[];tw=[];tl=[]
    seen=set()
    for h in map(tuple,cells):
        q=np.array(h)/reps;cart=q@model.reciprocal_m_inv
        phase=np.repeat(np.exp(1j*model.positions_m@cart),3)
        dynamic=phase[:,None]*model.dynamical(q)*phase.conj()[None,:]
        partner=tuple((-np.array(h))%reps)
        if h==partner:
            if np.linalg.norm(dynamic.imag)>1e-10*np.linalg.norm(dynamic):raise ValueError('self-conjugate cell gauge is not real')
            values,v=np.linalg.eigh(dynamic.real)
        else:values,v=np.linalg.eigh(dynamic)
        rigid=[]
        if not any(h):
            t=np.tile(np.eye(3),(len(model.positions_m),1))*np.repeat(np.sqrt(model.masses_kg),3)[:,None]
            t/=np.linalg.norm(t,axis=0)
            rigid=np.argsort(np.sum(abs(v.conj().T@t)**2,axis=1))[-3:]
        for j,value in enumerate(values):
            if j in rigid:continue
            if value<=0:raise ValueError('unstable physical mode')
            w=np.sqrt(value)
            e=(np.exp(2j*np.pi*(cells@q))[:,None]*v[:,j][None,:]/np.sqrt(len(cells))).reshape(-1,3)
            label=dict(q_index=list(h),q_fractional=q.tolist(),branch=j,partner=list(partner))
            travel.append(e);tw.append(w);tl.append(label)
            if h in seen:continue
            if h==partner:
                if np.linalg.norm(e.imag)>1e-10:raise ValueError('self-conjugate real mode failed')
                vectors.append(e.real);frequencies.append(w);labels.append(dict(label,standing='real'))
            else:
                for part,name in ((e.real,'cos'),(e.imag,'sin')):
                    vectors.append(np.sqrt(2)*part);frequencies.append(w);labels.append(dict(label,standing=name))
        seen.add(h);seen.add(partner)
    e=np.stack(vectors,axis=-1);w=np.array(frequencies)
    te=np.stack(travel,axis=-1);tw=np.array(tw)
    masses=np.tile(model.masses_kg,len(cells))
    d=e*np.sqrt(c.hbar/(2*masses[:,None,None]*w[None,None,:]))
    td=te*np.sqrt(c.hbar/(2*masses[:,None,None]*tw[None,None,:]))
    factors=2*thermal_occupation(w,temperature)+1
    covariance=np.einsum('iam,ibm,m->iab',d,d,factors)
    return dict(positions_m=positions,cell_m=reps[:,None]*model.cell_m,masses_kg=masses,
        eigenvectors=e,omega_rad_s=w,displacements_m=d,labels=labels,
        traveling_eigenvectors=te,traveling_displacements_m=td,traveling_omega_rad_s=tw,traveling_labels=tl,
        thermal_factors=factors,local_covariance_m2=covariance)


def inactive_covariance(full,displacements,omega,temperature):
    active=np.einsum('iam,ibm,m->iab',displacements,displacements.conj(),1+2*thermal_occupation(omega,temperature)).real
    inactive=np.asarray(full)-active
    scale=np.max(abs(full))
    if np.min(np.linalg.eigvalsh(inactive)) < -1e-10*scale:raise ValueError('active covariance exceeds full covariance')
    return inactive,active


def born_mode(potential,positions,displacement,covariance,qxy,omega,energy,area,length):
    """Infinite-z atomic integral, finite lateral Galerkin exit beams, SI.

    omega may be signed for creation/annihilation. The incident coefficient is
    one, so squared outgoing coefficients are absolute probabilities. No
    detector-wise normalization is performed. Finite vacuum tails need a check.
    """
    from .elastic_propagation import electron_speed
    from .free_propagation import electron_wavenumber
    velocity=electron_speed(energy);k0=electron_wavenumber(energy)
    kappa=np.sum(qxy*qxy,axis=1)/(2*k0)+omega/velocity
    K=np.column_stack((qxy,-kappa))
    kernel=potential.fourier_energy_j_m3(np.sum(K*K,axis=1))
    total=np.zeros(len(K),complex)
    for pos,d,cov in zip(positions,displacement,covariance):
        total+=np.exp(-1j*K@pos-.5*np.einsum('qa,ab,qb->q',K,cov,K))*(K@d.conj())
    return -kernel*total*np.exp(-1j*kappa*length)/(c.hbar*velocity*area)


def frequency_groups(omega,tolerance_mev=.01):
    energy=c.hbar*np.asarray(omega)/(c.e*1e-3)
    groups=[]
    for index in np.argsort(energy):
        if not groups or energy[index]-energy[groups[-1][0]]>tolerance_mev:groups.append([])
        groups[-1].append(int(index))
    return groups
