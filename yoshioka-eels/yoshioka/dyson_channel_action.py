"""Matrix-free Dyson amplitudes for the same diagonal/off-diagonal Hamiltonian.

Order is number of inter-channel interactions, not phonon occupation. A
nonlinear phase can create several phonons in a single interaction.
"""
import numpy as np
from scipy import constants
from scipy.sparse.linalg import LinearOperator,expm_multiply


class DysonChannelAction:
    def __init__(self,hamiltonian,maximum_order=2):
        self.h=hamiltonian;self.order=maximum_order;self.index=None
        self.shape=(maximum_order+1,*hamiltonian.shape)

    def diagonal_action(self,wave,index):
        h=self.h;h.set_slice(index)
        if self.index!=index:
            self.diagonal_position=(h.overlap*h.overlap).T@h.potential
            self.index=index
        wave=np.asarray(wave,dtype=complex)
        result=(h.channel_j[:,None,None]+h.kinetic_j[None])*wave
        full=np.zeros((len(h.basis),h.pad,h.pad),complex);ix=h.beam_indices
        full[:,ix[:,None],ix[None,:]]=wave
        position=np.fft.ifft2(full,axes=(-2,-1),norm='ortho')
        changed=position*self.diagonal_position.reshape(position.shape)
        fourier=np.fft.fft2(changed,axes=(-2,-1),norm='ortho')
        return result+fourier[:,ix[:,None],ix[None,:]]

    def apply(self,orders,index,adjoint=False):
        diagonal=[self.diagonal_action(w,index) for w in orders]
        output=np.array(diagonal)
        for j in range(self.order):
            source=j+1 if adjoint else j;target=j if adjoint else j+1
            output[target]+=self.h.apply(orders[source],index)-diagonal[source]
        return output

    def step(self,orders,index):
        h=self.h;count=orders.size;scale=-1j*h.dz/(constants.hbar*h.velocity)
        def action(x,factor,adjoint=False):return (factor*self.apply(np.asarray(x).reshape(self.shape),index,adjoint)).ravel()
        def batch(x,factor,adjoint=False):return np.column_stack([action(x[:,i],factor,adjoint) for i in range(x.shape[1])])
        operator=LinearOperator((count,count),dtype=complex,matvec=lambda x:action(x,scale),rmatvec=lambda x:action(x,scale.conjugate(),True),
            matmat=lambda x:batch(x,scale),rmatmat=lambda x:batch(x,scale.conjugate(),True))
        trace=(self.order+1)*(np.sum(h.channel_j)*h.n**2+len(h.basis)*np.sum(h.kinetic_j)+int(np.prod(h.shape))*h.scalar[index])
        return expm_multiply(operator,orders.ravel(),traceA=scale*trace).reshape(self.shape)

    def propagate(self,limits=None):
        orders=np.zeros(self.shape,complex);orders[0,0,self.h.n//2,self.h.n//2]=1
        for i in range(self.h.steps):
            if limits is not None and i%8==0:limits.check()
            orders=self.step(orders,i)
        return orders
