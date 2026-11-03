"""Continuous joint q/branch importance sampling from a fixed Born table.

The table is a proposal, never a replacement for the propagated integrand.
Mixture components retain full BZ support and an acoustic radial singularity.
Eigenvectors/dynamical matrices are recomputed at every continuous sampled q.
"""
import numpy as np
from scipy.stats import qmc


class BornPairProposal:
    def __init__(self,mass,mesh,uniform=.2,radial=.4,radius=.35):
        mass=np.asarray(mass,dtype=float)
        if mass.shape!=(mesh**3,6) or np.any(mass<0) or not np.isfinite(mass).all():
            raise ValueError('invalid joint q/branch probability table')
        if abs(mass.sum()-1)>1e-12:raise ValueError('proposal table must sum to one')
        if not 0<uniform<1 or not 0<=radial<1-uniform or not 0<radius<.5:
            raise ValueError('invalid proposal mixture')
        self.mass=mass;self.mesh=mesh;self.uniform=uniform;self.radial=radial
        self.born=1-uniform-radial;self.radius=radius
        self.cdf=np.cumsum(mass.ravel());self.cdf[-1]=1.

    def density(self,q,branch):
        q=np.asarray(q);branch=np.asarray(branch,dtype=int)
        idx=np.floor(q*self.mesh+.5).astype(int)%self.mesh
        cell=np.ravel_multi_index(tuple(idx.T),(self.mesh,)*3)
        r=np.linalg.norm(q,axis=1)
        density=np.full(len(q),self.uniform/6)
        inside=r<self.radius
        density[inside]+=self.radial/(6*4*np.pi*self.radius*r[inside]**2)
        density+=self.born*self.mesh**3*self.mass[cell,branch]
        return density

    def transform(self,u):
        q=u[:,2:5].copy()-.5
        branch=np.floor(6*u[:,1]).astype(int)
        choose_radial=(u[:,0]>=self.uniform)&(u[:,0]<self.uniform+self.radial)
        r=self.radius*u[choose_radial,2];cosine=2*u[choose_radial,3]-1
        phi=2*np.pi*u[choose_radial,4];sine=np.sqrt(1-cosine*cosine)
        q[choose_radial]=r[:,None]*np.column_stack((sine*np.cos(phi),sine*np.sin(phi),cosine))
        choose_born=u[:,0]>=self.uniform+self.radial
        selected=np.searchsorted(self.cdf,u[choose_born,1],side='right')
        branch[choose_born]=selected%6
        indices=np.array(np.unravel_index(selected//6,(self.mesh,)*3)).T
        q[choose_born]=(indices+u[choose_born,2:5]-.5)/self.mesh
        q=(q+.5)%1-.5
        return q,branch,1/self.density(q,branch)

    def nodes(self,power,seed):
        u=qmc.Sobol(10,scramble=True,seed=int(seed)).random_base2(power)
        a,ba,wa=self.transform(u[:,:5]);b,bb,wb=self.transform(u[:,5:])
        return np.stack((a,b),axis=1),np.column_stack((ba,bb)),np.column_stack((wa,wb))
