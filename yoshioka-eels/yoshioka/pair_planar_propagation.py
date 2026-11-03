"""Dealiased leading pair with analytic longitudinal fields and contact jumps.

The undisplaced Lobato atom has a Coulomb cusp. Its longitudinal second
derivative contains a Dirac contact. Midpoint nodes integrate only the regular
part; the direct order-two wave receives its exact finite jump at each atomic
plane, using the continuous elastic wave there. No contact is squared as a
separate probability. This remains the same bare Taylor pair observable.
"""
from pathlib import Path
import time
import numpy as np
from scipy import constants
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .dynamical_source import full_transverse_vectors
from .pair_galerkin import pair_structure,to_product,from_product,frequency_indices
from .bulk_thermal_source import mode_group,occupations
from .leading_two_phonon import leading_pair_trace


def propagate_planar_pair(grid,product_grid,omega,shifts,length,steps,fields,
                          local_substeps=1,limits=None,born=False):
    n,m=grid.nx,product_grid.nx
    frequency_indices(n,m)
    if m<2*n-1:raise ValueError('planar propagation requires exact dealiased products')
    if local_substeps<1 or int(local_substeps)!=local_substeps:raise ValueError('positive integer substeps required')
    words,lookup,energies,word_shifts,final=pair_structure(omega,shifts);w=len(words)
    parents=np.array([lookup[a[1:]] for a in words[1:]])
    letters=np.array([a[0] for a in words[1:]])
    all_e=np.r_[energies,energies[final]];all_s=np.r_[word_shifts,word_shifts[final]]
    Q=full_transverse_vectors(grid)[None]+all_s[:,None]
    kinetic_rate=np.sum(Q*Q,axis=-1).reshape(-1,n,n)/(2*electron_wavenumber(60000*constants.elementary_charge))
    hv=constants.hbar*electron_speed(60000*constants.elementary_charge)
    planes=np.asarray(fields.contact_depths_m)
    if np.any(planes<=0) or np.any(planes>=length):
        raise ValueError('incident/outgoing boundaries must lie outside all atomic contact planes')
    z=np.unique(np.r_[np.linspace(0,length,steps+1),planes])
    c=np.zeros((w+4,n,n),complex);c[0,0,0]=1.
    max_phase=0.;field_seconds=0.;action_seconds=0.;contacts=0
    before=time.monotonic()
    for i,(left,right) in enumerate(zip(z[:-1],z[1:])):
        if limits is not None and i%8==0:limits.check()
        dz=right-left;mid=(left+right)/2
        started=time.monotonic()
        parts=fields.averaged_fields(left,right) if hasattr(fields,'averaged_fields') else fields.fields_at(mid)
        bg=parts['background'];emission=parts['linear'];quadratic=parts['quadratic']
        if born:bg=np.zeros_like(bg)
        field_seconds+=time.monotonic()-started
        source=np.concatenate((emission,emission.conj()))
        scale=-1j*dz/(hv*local_substeps)
        max_phase=max(max_phase,float(np.max(abs(bg))*abs(scale)))
        kinetic=np.exp(-.5j*dz*kinetic_rate)
        def action(a):
            real=to_product(a,m)
            value=(bg[None]+all_e[:,None,None])*real
            value[1:w]+=source[letters]*real[parents]
            value[w:]+=quadratic*real[0]
            return scale*from_product(value,n)
        started=time.monotonic();c*=kinetic
        for j in range(local_substeps):
            k1=action(c);k2=action(c+.5*k1);k3=action(c+.5*k2);k4=action(c+k3)
            c+=(k1+2*k2+2*k3+k4)/6
        c*=kinetic
        # z contains the exact plane values, so no tolerance can merge contacts.
        hit=np.flatnonzero(planes==right)
        for plane in hit:
            contact=fields.quadratic_contact_fourier[int(plane)]
            contact_real=to_product(contact,m)
            c[w:]+=(-1j/hv)*from_product(contact_real*to_product(c[0],m),n)
            contacts+=1
        action_seconds+=time.monotonic()-started
    sequential=np.array([c[lookup[(0 if a==1 else 2,1 if b==1 else 3)]]+
                         c[lookup[(1 if b==1 else 3,0 if a==1 else 2)]] for a,b in ((1,1),(-1,-1),(1,-1),(-1,1))])
    return dict(direct_amplitudes=c[w:],sequential_amplitudes=sequential,total_amplitudes=c[w:]+sequential,
        word_coefficients=c[:w],final_shifts=word_shifts[final],word_shifts=word_shifts,word_energies_j=energies,
        field_evaluation_seconds=field_seconds,local_action_seconds=action_seconds,
        propagation_seconds=time.monotonic()-before,maximum_local_diagonal_phase=max_phase,
        actual_intervals=len(z)-1,contact_planes_applied=contacts,depth_nodes_m=z,
        retained_grid=n,product_grid=m,local_substeps=local_substeps,
        antialiasing='project each H action; exact transfer convolution',
        longitudinal_representation=('exact analytic interval means and V2zz contact jumps; time ordering/splitting still refined' if hasattr(fields,'averaged_fields') else
            'analytic periodic regular fields and exact V2zz Dirac contact jumps'))


def propagate_bulk_planar_pair(model,positions,reference,covariance,grid,product_grid,
                              length,steps,q_pair,branches,inverse_marginals,background,
                              scratch,temperature_k=300,limits=None,local_substeps=1,
                              born=False,save_amplitudes=True,require_dealiased=True,field_halfwidth=None,
                              cell_average=False):
    from .pair_planar_fields import PairPlanarFields
    if np.any(np.asarray(covariance)!=0):raise ValueError('bare static Taylor family required')
    if not require_dealiased or field_halfwidth is not None:
        raise ValueError('planar diagnostic retains the full exact transfer support')
    started=time.monotonic();omega=[];shifts=[];d=[]
    for side in range(2):
        om,displacement,shift,labels=mode_group(model,positions,[grid.lx_m,grid.ly_m],q_pair[side],1.,reference)
        at=int(np.flatnonzero(labels==branches[side])[0]);omega.append(om[at]);shifts.append(shift);d.append(displacement[:,:,at])
    phonon_seconds=time.monotonic()-started
    if cell_average:
        from .pair_planar_averages import AveragedPairPlanarFields
        field_class=AveragedPairPlanarFields
    else:field_class=PairPlanarFields
    fields=field_class(positions,product_grid,length,np.stack(d,axis=-1),np.array(shifts),covariance=covariance)
    r=propagate_planar_pair(grid,product_grid,np.array(omega),np.array(shifts),length,steps,fields,local_substeps,limits,born)
    trace=leading_pair_trace(r['direct_amplitudes'],r['sequential_amplitudes'],np.array(omega),
              occupations(np.array(omega),temperature_k),grid,r['final_shifts'],inverse_marginals)
    trace.update(omega_rad_s=np.array(omega),emission_shifts_m_inv=np.array(shifts),phonon_seconds=phonon_seconds,
        elapsed_seconds=time.monotonic()-started)
    if save_amplitudes:trace.update(r)
    return trace
