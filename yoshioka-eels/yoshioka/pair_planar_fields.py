"""Analytic static atom fields in transverse Fourier / longitudinal real space.

The Lobato 3D Fourier transform is inverted analytically along angular kz.
The regular second z derivative and its delta contacts are returned separately;
no midpoint sample substitutes for a contact. This is the bare static Taylor
family (zero smoothing covariance). Periodic longitudinal images are summed
in closed form, and all Cartesian components of the supplied complex mode
displacements are retained. This module constructs fields, not a propagator.

Fourier coefficients use V(r_xy,z)=sum_G V_G(z) exp(i G.r_xy), with each
channel's physical transfer G+shift. Regular coefficients are in joules;
quadratic delta coefficients are in joule metres. An emission uses d*, as in
the established complex_depth_field convention. The distinct-mode quadratic
cross vertex has coefficient one, including both mixed Cartesian terms.
"""
from dataclasses import dataclass
import numpy as np
from scipy import constants, fft

from .dynamical_source import full_transverse_vectors
from .electron_phonon import LobatoSilicon
from .two_phonon import SIGNS


def _parameters(potential):
    a=np.asarray(potential.a_angstrom,dtype=float)
    b=np.asarray(potential.b_angstrom2,dtype=float)
    if a.ndim!=1 or b.shape!=a.shape or not len(a) or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(b<=0):
        raise ValueError('finite Lobato a,b arrays with positive b required')
    ca=-(2*np.pi*constants.hbar**2/constants.m_e)*a*1e-10
    beta2=4*np.pi**2/(b*1e-20)
    return ca,beta2


def _kernels(q2,offsets,ca,beta2,period):
    """Return h, h', regular h''; array shape (offset, transverse Q)."""
    q2=np.asarray(q2,dtype=float).reshape(-1)
    offsets=np.asarray(offsets,dtype=float).reshape(-1)
    K=np.sqrt(q2[:,None]+beta2)[None,:,:]
    s=offsets[:,None,None]
    if period is None:
        distance=abs(s)
        exponential=np.exp(-K*distance)
        S0=exponential
        S1=distance*exponential
        Ssign=np.sign(s)*exponential
        Ssigned1=s*exponential
    else:
        # For 0<t<L, positive distances are t+nL and negative distances
        # -(L-t+nL). The geometric and first-moment sums are exact.
        t=np.remainder(s,period)
        r=np.exp(-K*period)
        denominator=-np.expm1(-K*period)
        positive=np.exp(-K*t)
        negative=np.exp(-K*(period-t))
        S0=(positive+negative)/denominator
        S1=(t*positive+(period-t)*negative)/denominator+period*r*(positive+negative)/denominator**2
        Ssign=(positive-negative)/denominator
        Ssigned1=(t*positive-(period-t)*negative)/denominator+period*r*(positive-negative)/denominator**2
        # h' jumps at a contact plane. Its value at the plane itself is a
        # measure-zero convention; choose the symmetric average, not a side.
        at_contact=np.minimum(t,period-t)<=8*np.finfo(float).eps*period
        Ssign=np.where(at_contact,0.,Ssign)
        Ssigned1=np.where(at_contact,0.,Ssigned1)
    h=np.sum(ca*(beta2*S0/(2*K)+beta2**2*(S0+K*S1)/(4*K**3)),axis=-1)
    h1=np.sum(ca*(-beta2*Ssign/2-beta2**2*Ssigned1/(4*K)),axis=-1)
    h2=np.sum(ca*(beta2*K*S0/2+beta2**2*(K*S1-S0)/(4*K)),axis=-1)
    return h,h1,h2


def planar_lobato_kernels(qxy_squared_m_inv2,offsets_m,potential=None,period_m=None):
    """Analytic mixed-space kernels h, h', and regular h''.

    Output shape is ``offsets.shape + qxy_squared.shape``. For one atom,
    the omitted h'' contact has integrated coefficient ``-sum(Ca*beta²)``
    in J m. ``period_m=None`` is the infinite line; a positive period sums
    every image exactly. The symmetric value h'=0 is used at the atom plane.
    """
    q2=np.asarray(qxy_squared_m_inv2,dtype=float)
    s=np.asarray(offsets_m,dtype=float)
    if np.any(~np.isfinite(q2)) or np.any(q2<0) or np.any(~np.isfinite(s)):
        raise ValueError('finite offsets and nonnegative transverse Q² required')
    if period_m is not None and (not np.isfinite(period_m) or period_m<=0):
        raise ValueError('positive finite longitudinal period required')
    ca,beta2=_parameters(potential or LobatoSilicon.from_abtem_table())
    shape=s.shape+q2.shape
    return tuple(a.reshape(shape) for a in _kernels(q2,s,ca,beta2,period_m))


def planar_second_derivative_contact(potential=None):
    """Coefficient of delta(z-z_atom) in h_z'', in J m."""
    ca,beta2=_parameters(potential or LobatoSilicon.from_abtem_table())
    return float(-np.sum(ca*beta2))


class PlanarDerivativeField:
    """One shifted potential or directional Taylor derivative field.

    No displacement gives V0. One ``first`` gives ``-first·grad V``.
    Supplying both gives ``(first·grad)(second·grad)V`` with distinct-mode
    coefficient one. Vectors are already conjugated when emission requires
    it. All atoms and all three Cartesian components are used.
    """
    def __init__(self,positions,grid,length,shift=(0.,0.),first=None,second=None,
                 potential=None,chunk=256):
        positions=np.asarray(positions,dtype=float)
        shift=np.asarray(shift,dtype=float)
        if positions.ndim!=2 or positions.shape[1]!=3 or not len(positions) or np.any(~np.isfinite(positions)):
            raise ValueError('finite nonempty (atoms,3) positions required')
        if not np.isfinite(length) or length<=0 or shift.shape!=(2,) or np.any(~np.isfinite(shift)):
            raise ValueError('positive period and finite two-dimensional shift required')
        if int(chunk)!=chunk or chunk<1:
            raise ValueError('positive integer chunk size required')
        if first is None and second is not None:
            raise ValueError('second derivative requires the first direction')
        directions=[]
        for direction in (first,second):
            if direction is not None:
                direction=np.asarray(direction,dtype=complex)
                if direction.shape!=positions.shape or np.any(~np.isfinite(direction)):
                    raise ValueError('finite directional displacements must match atom positions')
                directions.append(direction)
        self.order=len(directions)
        self.grid=grid;self.length=float(length);self.chunk=int(chunk)
        self.potential=potential or LobatoSilicon.from_abtem_table()
        self.ca,self.beta2=_parameters(self.potential)
        self.qxy=full_transverse_vectors(grid)+shift
        self.q2=np.sum(self.qxy**2,axis=1)
        self.area=grid.lx_m*grid.ly_m
        self.planes_m,inverse=np.unique(np.remainder(positions[:,2],length),return_inverse=True)
        self.coefficients=[np.zeros((len(self.planes_m),len(self.qxy)),complex) for _ in range(self.order+1)]
        for lo in range(0,len(self.qxy),self.chunk):
            hi=min(lo+self.chunk,len(self.qxy));Q=self.qxy[lo:hi]
            for p in range(len(self.planes_m)):
                selected=inverse==p
                phase=np.exp(-1j*positions[selected,:2]@Q.T)
                if self.order==0:
                    self.coefficients[0][p,lo:hi]=phase.sum(axis=0)
                    continue
                a=directions[0][selected];qa=a[:,:2]@Q.T
                if self.order==1:
                    self.coefficients[0][p,lo:hi]=np.sum(phase*qa,axis=0)
                    self.coefficients[1][p,lo:hi]=np.sum(phase*a[:,2,None],axis=0)
                else:
                    b=directions[1][selected];qb=b[:,:2]@Q.T
                    az=a[:,2,None];bz=b[:,2,None]
                    self.coefficients[0][p,lo:hi]=np.sum(phase*qa*qb,axis=0)
                    self.coefficients[1][p,lo:hi]=np.sum(phase*(qa*bz+az*qb),axis=0)
                    self.coefficients[2][p,lo:hi]=np.sum(phase*az*bz,axis=0)

    def regular_fourier(self,z):
        """Regular coefficients in J at arbitrary real z; no delta sample."""
        if np.ndim(z)!=0 or not np.isfinite(z):
            raise ValueError('one finite longitudinal position required')
        out=np.empty(len(self.qxy),complex)
        for lo in range(0,len(self.qxy),self.chunk):
            hi=min(lo+self.chunk,len(self.qxy))
            h,h1,h2=_kernels(self.q2[lo:hi],z-self.planes_m,self.ca,self.beta2,self.length)
            c=[a[:,lo:hi] for a in self.coefficients]
            if self.order==0:value=c[0]*h
            elif self.order==1:value=-1j*c[0]*h-c[1]*h1
            else:value=-c[0]*h+1j*c[1]*h1+c[2]*h2
            out[lo:hi]=np.sum(value,axis=0)/self.area
        return out.reshape(self.grid.shape)

    @property
    def contact_fourier(self):
        """Integrated contact coefficients (plane,ny,nx), in J m.

        Only an order-two field has contacts. The generic class returns zeros
        for lower orders so callers cannot mistake a derivative jump for an
        additional delta source in V1.
        """
        shape=(len(self.planes_m),*self.grid.shape)
        if self.order!=2:return np.zeros(shape,complex)
        contact=-np.sum(self.ca*self.beta2)
        return (self.coefficients[2]*contact/self.area).reshape(shape)


class PairPlanarFields:
    """Static background, two linear fields, four pair vertices and contacts.

    ``displacements`` has shape (atoms,3,2) and contains the original complex
    d, not already conjugated emission vectors. ``shifts`` are the two raw
    emission Bloch shifts in angular m^-1. Nonzero phonon q_z is represented
    by the supplied atom-dependent complex d; u_z is retained explicitly.
    No finite-target nonlinear mode basis is inferred or added here.
    """
    def __init__(self,positions,grid,length,displacements,shifts,covariance=None,
                 potential=None,chunk=256):
        positions=np.asarray(positions,dtype=float)
        d=np.asarray(displacements,dtype=complex)
        shifts=np.asarray(shifts,dtype=float)
        if d.shape!=(*positions.shape,2) or shifts.shape!=(2,2):
            raise ValueError('two (atoms,3) displacements and two transverse shifts required')
        if covariance is not None and (np.asarray(covariance).shape!=(3,3) or np.any(np.asarray(covariance)!=0)):
            raise ValueError('analytic planar fields require the bare static Taylor family with zero covariance')
        self.grid=grid;self.length=length;self.displacements=d;self.shifts=shifts
        self.potential=potential or LobatoSilicon.from_abtem_table()
        common=dict(positions=positions,grid=grid,length=length,potential=self.potential,chunk=chunk)
        self.background=PlanarDerivativeField(**common)
        self.linear=[PlanarDerivativeField(**common,shift=shifts[j],first=d[:,:,j].conj()) for j in range(2)]
        self.final_shifts=np.array([a*shifts[0]+b*shifts[1] for a,b in SIGNS])
        self.quadratic=[]
        for index,(sa,sb) in enumerate(SIGNS):
            a=d[:,:,0].conj() if sa==1 else d[:,:,0]
            b=d[:,:,1].conj() if sb==1 else d[:,:,1]
            self.quadratic.append(PlanarDerivativeField(**common,shift=self.final_shifts[index],first=a,second=b))
        self.contact_depths_m=self.background.planes_m.copy()
        self.quadratic_contact_fourier=np.stack([field.contact_fourier for field in self.quadratic],axis=1)

    def regular_fourier(self,z):
        """Dictionary of background, linear and quadratic coefficients, in J."""
        return dict(background=self.background.regular_fourier(z),
                    linear=np.stack([field.regular_fourier(z) for field in self.linear]),
                    quadratic=np.stack([field.regular_fourier(z) for field in self.quadratic]))

    def fields_at(self,z):
        """Dictionary of periodic real-space factors on the declared grid."""
        out={key:fft.ifft2(value,axes=(-2,-1),workers=1)*(self.grid.nx*self.grid.ny)
             for key,value in self.regular_fourier(z).items()}
        b=out['background']
        if np.linalg.norm(b.imag)>1e-10*max(np.linalg.norm(b),np.finfo(float).tiny):
            raise ValueError('analytic static background failed its reality check')
        out['background']=b.real
        return out

    def contact_field(self,plane):
        """The four integrated delta vertices on one plane, in J m."""
        return fft.ifft2(self.quadratic_contact_fourier[plane],axes=(-2,-1),workers=1)*(self.grid.nx*self.grid.ny)

    def metadata(self):
        return dict(reference='bare static Taylor; zero covariance',
            units='regular Fourier coefficients J; quadratic contact Fourier coefficients J m; angular reciprocal vectors m^-1',
            longitudinal='analytic inverse kz transform with exact periodic image sum; delta contacts separate',
            regular_plane_convention='symmetric value of discontinuous h_prime at each atom plane',
            Cartesian_support=['x','y','z'],complex_displacements=True,phonon_qz='encoded by supplied atom-dependent complex displacement',
            quadratic='distinct-mode coherent cross vertex, coefficient1; four thermal sign sectors; returns excluded',
            propagation='not provided; regular evolution and contact jumps must both be integrated',
            retained_grid=[self.grid.ny,self.grid.nx],contact_planes_m=self.contact_depths_m.tolist())
