"""Exact cell means of the regular analytic planar Taylor fields.

Each interval must have no atomic/contact plane in its open interior. Delta
contacts stay separate and are never divided by a depth step. Exact zeroth
moments remove midpoint field-integration error; kinetic splitting and depth
ordering remain approximations that need their own convergence controls.
"""
import numpy as np
from scipy import fft
from scipy.special import gammainc

from .electron_phonon import LobatoSilicon
from .pair_planar_fields import PairPlanarFields,_parameters


def _averaged_kernels(q2,left_offsets,width,ca,beta2,period):
    """Mean h, h', regular h'' on unbroken periodic-image intervals."""
    q2=np.asarray(q2,dtype=float).reshape(-1)
    offsets=np.asarray(left_offsets,dtype=float).reshape(-1)
    K=np.sqrt(q2[:,None]+beta2)[None,:,:]
    u=np.remainder(offsets,period)[:,None,None]
    v=u+width
    tolerance=16*np.finfo(float).eps*period
    if np.any(v>period+tolerance):
        raise ValueError('an atomic plane lies in the open interval; split the interval at that plane')
    # Modulo and the independent endpoint subtraction can differ by a few
    # ulps when the right endpoint is exactly the plane. Only that roundoff
    # is clipped; an actual interior crossing was rejected above.
    v=np.minimum(v,period)
    negative_lower=np.maximum(period-v,0.)
    x=K*width
    one_minus_exponential=-np.expm1(-x)
    first_moment=gammainc(2.,x)
    positive=np.exp(-K*u)
    negative=np.exp(-K*negative_lower)
    p0=positive*one_minus_exponential/K
    n0=negative*one_minus_exponential/K
    p1=positive*(u*one_minus_exponential/K+first_moment/K**2)
    n1=negative*(negative_lower*one_minus_exponential/K+first_moment/K**2)
    r=np.exp(-K*period)
    denominator=-np.expm1(-K*period)
    S0=(p0+n0)/denominator/width
    S1=((p1+n1)/denominator+period*r*(p0+n0)/denominator**2)/width
    Ssign=(p0-n0)/denominator/width
    Ssigned1=((p1-n1)/denominator+period*r*(p0-n0)/denominator**2)/width
    h=np.sum(ca*(beta2*S0/(2*K)+beta2**2*(S0+K*S1)/(4*K**3)),axis=-1)
    h1=np.sum(ca*(-beta2*Ssign/2-beta2**2*Ssigned1/(4*K)),axis=-1)
    h2=np.sum(ca*(beta2*K*S0/2+beta2**2*(K*S1-S0)/(4*K)),axis=-1)
    return h,h1,h2


def averaged_planar_lobato_kernels(qxy_squared_m_inv2,left_offsets_m,width_m,
                                   period_m,potential=None):
    """Exact interval means h, h', regular h'' in mixed Fourier/real space.

    The offset is left_endpoint minus atom_plane. Each offset interval must
    avoid an image atom plane in its open interior; either endpoint may be a
    plane. Return shape is ``left_offsets.shape + qxy_squared.shape``. The
    regular kernels keep their pointwise units; no contact is included.
    """
    q2=np.asarray(qxy_squared_m_inv2,dtype=float)
    s=np.asarray(left_offsets_m,dtype=float)
    if np.any(~np.isfinite(q2)) or np.any(q2<0) or np.any(~np.isfinite(s)):
        raise ValueError('finite offsets and nonnegative transverse Q² required')
    if not np.isfinite(period_m) or period_m<=0 or not np.isfinite(width_m) or width_m<=0 or width_m>period_m:
        raise ValueError('positive width no larger than the positive period required')
    ca,beta2=_parameters(potential or LobatoSilicon.from_abtem_table())
    shape=s.shape+q2.shape
    return tuple(a.reshape(shape) for a in _averaged_kernels(q2,s,width_m,ca,beta2,period_m))


def _field_average(field,left,right):
    width=right-left
    out=np.empty(len(field.qxy),complex)
    for lo in range(0,len(field.qxy),field.chunk):
        hi=min(lo+field.chunk,len(field.qxy))
        h,h1,h2=_averaged_kernels(field.q2[lo:hi],left-field.planes_m,width,
                                 field.ca,field.beta2,field.length)
        c=[a[:,lo:hi] for a in field.coefficients]
        if field.order==0:value=c[0]*h
        elif field.order==1:value=-1j*c[0]*h-c[1]*h1
        else:value=-c[0]*h+1j*c[1]*h1+c[2]*h2
        out[lo:hi]=np.sum(value,axis=0)/field.area
    return out.reshape(field.grid.shape)


class AveragedPairPlanarFields(PairPlanarFields):
    """The identical physical fields, with exact regular interval means."""
    def _check_interval(self,left,right):
        if np.ndim(left)!=0 or np.ndim(right)!=0 or not np.isfinite(left) or not np.isfinite(right):
            raise ValueError('finite scalar interval endpoints required')
        if not 0<=left<right<=self.length:
            raise ValueError('interval must lie in the declared longitudinal period')
        if np.any((self.contact_depths_m>left)&(self.contact_depths_m<right)):
            raise ValueError('atomic plane in open interval; split it before averaging')

    def averaged_fourier(self,left,right):
        """Exact regular means in J; contacts remain the inherited J m arrays."""
        self._check_interval(left,right)
        return dict(background=_field_average(self.background,left,right),
                    linear=np.stack([_field_average(field,left,right) for field in self.linear]),
                    quadratic=np.stack([_field_average(field,left,right) for field in self.quadratic]))

    def averaged_fields(self,left,right):
        """Exact regular means as periodic real-space factors on this grid."""
        out={key:fft.ifft2(value,axes=(-2,-1),workers=1)*(self.grid.nx*self.grid.ny)
             for key,value in self.averaged_fourier(left,right).items()}
        b=out['background']
        if np.linalg.norm(b.imag)>1e-10*max(np.linalg.norm(b),np.finfo(float).tiny):
            raise ValueError('averaged static background failed its reality check')
        out['background']=b.real
        return out

    def metadata(self):
        return dict(super().metadata(),regular_sampling='exact interval mean; no atomic plane in open interval',
                    remaining_depth_errors='kinetic splitting and source/elastic depth ordering; exact means alone do not establish convergence')
