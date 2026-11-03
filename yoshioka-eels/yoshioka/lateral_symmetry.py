"""Exact zero-Bloch-sector isometry for a laterally repeated finite bath.

A unit plane wave and Gamma-mode operator stay in beams divisible by (rx,ry).
Coefficient amplitudes are unchanged under the isometry: the reduced periodic
wave is repeated in real space with factor 1/sqrt(rx*ry). Physical momenta and
absolute detector probabilities are consequently unchanged.
"""
from dataclasses import dataclass
import numpy as np
from .finite_snap_bath import FiniteOpticalBath


@dataclass
class LateralTranslationSector:
    repetitions: tuple
    full_shape: tuple  # (ny,nx), including unreachable zero coefficients

    def __post_init__(self):
        if len(self.repetitions)!=2 or any(int(v)!=v or v<1 for v in self.repetitions):
            raise ValueError('two positive integer lateral repetitions required')
        if len(self.full_shape)!=2 or any(int(v)!=v or v<1 or v%2!=1 for v in self.full_shape):
            raise ValueError('two positive odd full electron grid dimensions required')
        ny,nx=self.full_shape;rx,ry=self.repetitions
        bx=np.arange(-(nx//2),nx//2+1);by=np.arange(-(ny//2),ny//2+1)
        self.ix=np.flatnonzero(bx%rx==0);self.iy=np.flatnonzero(by%ry==0)
        self.reduced_shape=(len(self.iy),len(self.ix))
        self.reduced_beams=(bx[self.ix]//rx,by[self.iy]//ry)

    def expand(self, wave):
        wave=np.asarray(wave)
        if wave.shape[-2:]!=self.reduced_shape: raise ValueError('incorrect reduced shape')
        output=np.zeros((*wave.shape[:-2],*self.full_shape),dtype=wave.dtype)
        output[...,self.iy[:,None],self.ix[None,:]]=wave
        return output

    def reduce_bath(self, bath):
        """Validate the entire supplied specimen before returning one lateral tile.

        Active displacements keep their FULL specimen normalization. No rescaling
        by sqrt(rx*ry) is permitted here. Roundoff-level equivalent covariances
        are averaged across lateral translates after checking their equality.
        """
        rx,ry=self.repetitions;cell=np.array(bath.cell_m,copy=True)
        if np.linalg.norm(cell-np.diag(np.diag(cell)))>1e-12*np.linalg.norm(cell):
            raise ValueError('orthorhombic bath required')
        cell[0]/=rx;cell[1]/=ry
        pos=np.asarray(bath.positions_m);fraction=pos@np.linalg.inv(cell)
        origin=(fraction[:,0]<1-1e-10)&(fraction[:,1]<1-1e-10)
        indices=np.flatnonzero(origin)
        if len(indices)*rx*ry!=len(pos): raise ValueError('incomplete lateral translations')
        groups=[];used=set();maximum=0.
        for i in indices:
            group=[]
            for x,y in np.ndindex(rx,ry):
                expected=pos[i]+x*cell[0]+y*cell[1]
                distance=np.linalg.norm(pos-expected,axis=1)
                j=int(np.argmin(distance))
                if distance[j]>1e-10*np.linalg.norm(cell) or j in used:
                    raise ValueError('atom mapping is not a complete translation bijection')
                used.add(j);group.append(j)
                for array in (bath.displacement_m,bath.inactive_covariance_m2):
                    error=np.linalg.norm(array[j]-array[i])/max(np.linalg.norm(array[i]),np.finfo(float).tiny)
                    maximum=max(maximum,error)
                    if error>1e-10: raise ValueError('bath breaks lateral translation symmetry')
            groups.append(group)
        if len(used)!=len(pos): raise ValueError('unmapped specimen atoms')
        # Exact equal inputs retain their bytes; averaging is only for roundoff.
        covariance=np.array([bath.inactive_covariance_m2[g[0]] if np.all(bath.inactive_covariance_m2[g]==bath.inactive_covariance_m2[g[0]]) else bath.inactive_covariance_m2[g].mean(axis=0) for g in groups])
        result=FiniteOpticalBath(np.array(bath.displacement_m[indices],copy=True),
            np.array(bath.omega_rad_s,copy=True),covariance,cell,pos[indices].copy(),list(bath.labels))
        self.evidence={'specimen_atoms':len(pos),'lateral_tiles':rx*ry,'representative_atoms':len(indices),
            'maximum_translation_relative_error':maximum,'coefficient_scale':1.,
            'active_displacements_rescaled':False,'full_shape':self.full_shape,'reduced_shape':self.reduced_shape}
        return result
