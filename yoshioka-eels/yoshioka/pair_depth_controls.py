"""Independent field-table and propagation meshes for contact-separated pairs.

A table represents a *fixed piecewise-constant regular Hamiltonian*. Refining
its propagation mesh does not reevaluate any field. The common propagation
mesh may also be used with different tables, thereby changing representation
without changing integration intervals. Exact atomic-plane jumps remain
separate. Neither exact interval means nor table refinement remove kinetic
splitting, source ordering, or local exponential-action errors.
"""
import time
import numpy as np
from scipy import constants
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .dynamical_source import full_transverse_vectors
from .pair_galerkin import pair_structure,to_product,from_product,frequency_indices
from .two_phonon import SIGNS


PARTS=('background','linear','quadratic')


def contact_aligned_mesh(length,intervals,planes):
    if not np.isfinite(length) or length<=0 or intervals<1 or int(intervals)!=intervals:
        raise ValueError('positive period and integer interval count required')
    planes=np.asarray(planes,float)
    if planes.ndim!=1 or np.any(~np.isfinite(planes)) or np.any(planes<=0) or np.any(planes>=length):
        raise ValueError('contact planes must be strictly inside the propagation interval')
    return np.unique(np.r_[np.linspace(0,length,int(intervals)+1),planes])


def subdivide_mesh(nodes,factor):
    nodes=np.asarray(nodes,float)
    if nodes.ndim!=1 or len(nodes)<2 or np.any(~np.isfinite(nodes)) or np.any(np.diff(nodes)<=0):
        raise ValueError('finite strictly increasing mesh required')
    if int(factor)!=factor or factor<1:raise ValueError('positive integer mesh subdivision required')
    return np.r_[np.concatenate([np.linspace(a,b,int(factor)+1)[:-1] for a,b in zip(nodes[:-1],nodes[1:])]),nodes[-1]]


def pack_regular(parts):
    background=np.asarray(parts['background'])
    return np.concatenate((background[None],np.asarray(parts['linear']),np.asarray(parts['quadratic'])),axis=0)


def coarsen_regular_table(fine_values,fine_nodes,coarse_nodes):
    """Zeroth moments on nested meshes; no delta contact enters the table."""
    fine_nodes=np.asarray(fine_nodes);coarse_nodes=np.asarray(coarse_nodes)
    ix=np.searchsorted(fine_nodes,coarse_nodes)
    if np.any(ix>=len(fine_nodes)) or not np.array_equal(fine_nodes[ix],coarse_nodes):
        raise ValueError('coarse boundaries must be exact members of the fine mesh')
    out=np.empty((len(coarse_nodes)-1,*fine_values.shape[1:]),complex)
    for i,(lo,hi) in enumerate(zip(ix[:-1],ix[1:])):
        weights=np.diff(fine_nodes[lo:hi+1])/(coarse_nodes[i+1]-coarse_nodes[i])
        out[i]=np.tensordot(weights,fine_values[lo:hi],axes=1)
    return out


class FrozenPlanarTable:
    """Fixed regular values and separate integrated contact Fourier arrays."""
    def __init__(self,nodes,regular,contact_depths,contact_fourier):
        self.nodes=np.asarray(nodes,float);self.regular=np.asarray(regular)
        self.contact_depths_m=np.asarray(contact_depths,float)
        self.quadratic_contact_fourier=np.asarray(contact_fourier)
        if self.nodes.ndim!=1 or len(self.nodes)<2 or self.nodes[0]!=0 or np.any(np.diff(self.nodes)<=0):
            raise ValueError('table mesh must increase from zero')
        if self.regular.ndim!=4 or self.regular.shape[:2]!=(len(self.nodes)-1,7) or self.regular.shape[-1]!=self.regular.shape[-2]:
            raise ValueError('regular table shape must be (cells,7,M,M)')
        if self.quadratic_contact_fourier.shape!=(len(self.contact_depths_m),4,*self.regular.shape[-2:]):
            raise ValueError('contact array shape mismatch')
        if np.any(self.contact_depths_m<=0) or np.any(self.contact_depths_m>=self.nodes[-1]) or len(np.unique(self.contact_depths_m))!=len(self.contact_depths_m):
            raise ValueError('distinct interior contacts required')
        if not np.all(np.isin(self.contact_depths_m,self.nodes)):
            raise ValueError('every contact must be an exact table boundary')
        self.nodes.setflags(write=False);self.regular.setflags(write=False)
        self.contact_depths_m.setflags(write=False);self.quadratic_contact_fourier.setflags(write=False)

    def propagation_cells(self,nodes):
        nodes=np.asarray(nodes,float)
        if nodes.ndim!=1 or nodes[0]!=self.nodes[0] or nodes[-1]!=self.nodes[-1] or np.any(np.diff(nodes)<=0):
            raise ValueError('propagation mesh endpoints/order differ from the fixed field interval')
        if not np.all(np.isin(self.nodes,nodes)):
            raise ValueError('propagation must preserve every fixed field discontinuity exactly')
        cell=np.searchsorted(self.nodes,nodes[:-1],side='right')-1
        if np.any(nodes[1:]>self.nodes[cell+1]):raise ValueError('a propagation interval crosses a field discontinuity')
        return cell


def propagate_table_pair(grid,product_grid,omega,shifts,table,propagation_nodes,
                          local_substeps=2,limits=None,start_interval=0,initial_coefficients=None,
                          checkpoint=None):
    """Strang kinetic / projected RK4 local evolution of one fixed table.

    The operation and target hierarchy are the same complete D+S pair used by
    the established contact propagator. User-supplied mesh changes only the
    propagation discretization; the table is never evaluated or interpolated.
    """
    n,m=grid.nx,product_grid.nx;frequency_indices(n,m)
    if grid.shape!=(n,n) or product_grid.shape!=(m,m) or (grid.lx_m,grid.ly_m)!=(product_grid.lx_m,product_grid.ly_m):
        raise ValueError('square retained/product grids with the same physical cell required')
    if m<2*n-1:raise ValueError('full dealiased Galerkin transfers required')
    if local_substeps<1 or int(local_substeps)!=local_substeps:raise ValueError('positive integer local substeps required')
    if table.regular.shape[-2:]!=(m,m):raise ValueError('field table product grid mismatch')
    nodes=np.asarray(propagation_nodes,float);indices=table.propagation_cells(nodes)
    words,lookup,energies,word_shifts,final=pair_structure(omega,shifts);w=len(words)
    parents=np.array([lookup[a[1:]] for a in words[1:]]);letters=np.array([a[0] for a in words[1:]])
    all_e=np.r_[energies,energies[final]];all_s=np.r_[word_shifts,word_shifts[final]]
    Q=full_transverse_vectors(grid)[None]+all_s[:,None]
    kinetic_rate=np.sum(Q*Q,axis=-1).reshape(-1,n,n)/(2*electron_wavenumber(60000*constants.elementary_charge))
    hv=constants.hbar*electron_speed(60000*constants.elementary_charge)
    c=np.zeros((w+4,n,n),complex);c[0,0,0]=1.
    if initial_coefficients is not None:
        if np.asarray(initial_coefficients).shape!=c.shape:raise ValueError('checkpoint hierarchy shape differs')
        c[:]=initial_coefficients
    if not 0<=start_interval<len(nodes):raise ValueError('invalid start interval')
    max_phase=0.;contacts=int(np.sum(table.contact_depths_m<=nodes[start_interval]));max_norm_drift=0.;before=time.monotonic()
    for i in range(start_interval,len(nodes)-1):
        if limits is not None and i%8==0:limits.check()
        dz=nodes[i+1]-nodes[i];parts=table.regular[indices[i]]
        bg=parts[0];source=np.concatenate((parts[1:3],parts[1:3].conj()));quadratic=parts[3:]
        scale=-1j*dz/(hv*local_substeps);max_phase=max(max_phase,float(np.max(abs(bg))*abs(scale)))
        kinetic=np.exp(-.5j*dz*kinetic_rate)
        def action(a):
            real=to_product(a,m);value=(bg[None]+all_e[:,None,None])*real
            value[1:w]+=source[letters]*real[parents];value[w:]+=quadratic*real[0]
            return scale*from_product(value,n)
        c*=kinetic
        for j in range(local_substeps):
            k1=action(c);k2=action(c+.5*k1);k3=action(c+.5*k2);k4=action(c+k3)
            c+=(k1+2*k2+2*k3+k4)/6
        c*=kinetic
        for plane in np.flatnonzero(table.contact_depths_m==nodes[i+1]):
            contact_real=to_product(table.quadratic_contact_fourier[plane],m)
            c[w:]+=(-1j/hv)*from_product(contact_real*to_product(c[0],m),n);contacts+=1
        max_norm_drift=max(max_norm_drift,abs(float(np.sum(abs(c[0])**2))-1))
        if checkpoint is not None and ((i+1)%64==0 or i+1==len(nodes)-1):checkpoint(i+1,c)
    sequential=np.array([c[lookup[(0 if a==1 else 2,1 if b==1 else 3)]]+c[lookup[(1 if b==1 else 3,0 if a==1 else 2)]] for a,b in SIGNS])
    return dict(direct_amplitudes=c[w:],sequential_amplitudes=sequential,total_amplitudes=c[w:]+sequential,
        word_coefficients=c[:w],final_shifts=word_shifts[final],word_shifts=word_shifts,word_energies_j=energies,
        propagation_seconds=time.monotonic()-before,maximum_local_diagonal_phase=max_phase,
        maximum_elastic_norm_drift_this_invocation=max_norm_drift,contact_planes_applied=contacts,
        propagation_nodes_m=nodes,field_table_nodes_m=table.nodes,field_cells=len(table.nodes)-1,
        actual_intervals=len(nodes)-1,retained_grid=n,product_grid=m,local_substeps=local_substeps,
        antialiasing='project each H action; M>=2N-1; exact transfer convolution',
        longitudinal_representation='fixed piecewise-constant regular table; integrated atomic-plane contacts exact and separate')
