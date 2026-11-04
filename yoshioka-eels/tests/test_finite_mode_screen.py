import unittest,tempfile
from types import SimpleNamespace
import numpy as np
from scipy import constants as c
from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
from yoshioka.finite_mode_screen import commensurate_bath,born_mode,inactive_covariance
from yoshioka.longitudinal_material_action import exact_phase_blocks,LongitudinalMaterialAction
from yoshioka.projected_material_action import phase_blocks,ProjectedMaterialAction
from yoshioka.electron_phonon import LobatoSilicon

class FiniteScreenTests(unittest.TestCase):
 def test_finite_q_gauge_against_independent_real_matrix(self):
  shifts=[];phi=[]
  for axis in range(3):
   for sign in (-1,1):
    r=np.zeros(3,int);r[axis]=sign;shifts.append(r);phi.append(-np.diag([2.,3.,4.]))
  shifts.append([0,0,0]);phi.append(np.diag([12.,18.,24.]))
  model=RealSpaceForceConstants(np.eye(3)*4e-10,np.zeros((1,3)),np.array([28*c.atomic_mass]),np.zeros(7,int),np.zeros(7,int),np.array(shifts),np.array(phi))
  b=commensurate_bath(model,(3,3,2),10);d=finite_periodic_modes(model,(3,3,2))
  np.testing.assert_allclose(np.sort(b['omega_rad_s']),d['omega_rad_s'],rtol=1e-13)
  e=b['eigenvectors'].reshape(54,51);np.testing.assert_allclose(e.T@e,np.eye(51),atol=3e-14)
  np.testing.assert_allclose(e@e.T,np.eye(54)-d['translation_vectors']@d['translation_vectors'].T,atol=3e-14)
  inactive,active=inactive_covariance(b['local_covariance_m2'],b['displacements_m'][:,:,:8],b['omega_rad_s'][:8],10)
  np.testing.assert_allclose(inactive+active,b['local_covariance_m2'],rtol=2e-16)
 def test_exact_phase_independent_hermite_and_finite_oscillator(self):
  from yoshioka.coupled_channels import channel_basis
  b=channel_basis(3,3);xi=np.array([[.13,-.21,.31],[0,0,0]])
  np.testing.assert_allclose(exact_phase_blocks(b,xi),phase_blocks(b,xi,48),atol=4e-14)
  np.testing.assert_allclose(exact_phase_blocks(b,xi),phase_blocks(b,xi,quadrature_order=32),atol=4e-14)
 def test_born_independent_one_atom_and_mode_rotation(self):
  from yoshioka.elastic_propagation import electron_speed
  from yoshioka.free_propagation import electron_wavenumber
  pot=LobatoSilicon(np.array([1.]),np.array([2.]));pos=np.zeros((1,3));cov=np.array([np.eye(3)*1e-23]);d=np.array([[1.,2.,3.]])*1e-12;q=np.array([[2e10,1e10],[0,0.]])
  w=7e13;v=electron_speed(60000*c.e);kap=np.sum(q*q,axis=1)/(2*electron_wavenumber(60000*c.e))+w/v;K=np.column_stack((q,-kap))
  ref=-pot.fourier_energy_j_m3(np.sum(K*K,axis=1))*np.exp(-.5*np.sum(K*K,axis=1)*1e-23)*(K@d[0])/(c.hbar*v*1e-18)*np.exp(-1j*kap*3e-9)
  np.testing.assert_allclose(born_mode(pot,pos,d,cov,q,w,60000*c.e,1e-18,3e-9),ref,rtol=2e-15)
  self.assertGreater(abs(ref[1]),0) # pure longitudinal forward source retained
  ds=[d,np.roll(d,1,axis=1)];amps=[born_mode(pot,pos,x,cov,q,w,60000*c.e,1e-18,3e-9) for x in ds]
  rotated=[(ds[0]+ds[1])/np.sqrt(2),(ds[0]-ds[1])/np.sqrt(2)]
  other=[born_mode(pot,pos,x,cov,q,w,60000*c.e,1e-18,3e-9) for x in rotated]
  np.testing.assert_allclose(np.sum(abs(np.array(amps))**2,axis=0),np.sum(abs(np.array(other))**2,axis=0),rtol=2e-15)
 def test_all_direction_action_reduces_to_independent_transverse_solver(self):
  bath=SimpleNamespace(displacement_m=np.array([[[.02e-10,0],[0,.03e-10],[0,0]]]),omega_rad_s=np.array([8e13,9e13]),inactive_covariance_m2=np.array([np.eye(3)*.01e-20]),labels=['x','y'],cell_m=np.diag([5.,4.,6.])*1e-10,positions_m=np.array([[.8,1.2,2.]])*1e-10)
  with tempfile.TemporaryDirectory() as folder:
   a=ProjectedMaterialAction(None,2,3,17,17,folder,bath=bath)
   b=LongitudinalMaterialAction(bath,2,3,17,17,folder,minimum_free_bytes=0)
   rng=np.random.default_rng(12);wave=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
   np.testing.assert_allclose(a.apply(wave,8),b.apply(wave,8),rtol=2e-13,atol=1e-33)
   bath.displacement_m[0,2,0]=.01e-10
   full=LongitudinalMaterialAction(bath,2,3,17,17,folder,minimum_free_bytes=0)
   matrix=np.column_stack([full.apply(x.reshape(full.shape),8).ravel() for x in np.eye(wave.size)])
   np.testing.assert_allclose(matrix,matrix.conj().T,rtol=1e-13,atol=1e-33)

class SectorScreenTests(unittest.TestCase):
 def test_sector_vertex_against_full_atomic_sum(self):
  from yoshioka.sector_screen import source_profiles
  from yoshioka.electron_phonon import LobatoSilicon
  prim=np.diag([5.,4.,6.])*1e-10;cells=np.array(list(np.ndindex(5,5,1)));pos=cells@prim+np.array([.7,1.1,2.])*1e-10
  q=np.array([1/5,2/5,0]);pol=np.array([1.,2.,3.])*1e-12
  d=(np.exp(2j*np.pi*cells@q)[:,None]*pol[None]/5)[:,:,None]
  cov=np.tile(np.eye(3)*1e-23,(25,1,1));cell=prim*np.array([5,5,1])[:,None]
  b=dict(cell_m=cell,positions_m=pos,traveling_displacements_m=d,local_covariance_m2=cov)
  fields=17;steps=17;prof,r=source_profiles(b,[0],[dict(q_index=[1,2,0])],3,steps,fields)
  diff=np.rint(np.fft.fftfreq(5)*5).astype(int);kz=2*np.pi*np.fft.fftfreq(fields,(cell[2,2]+2.4e-9)/fields);pot=LobatoSilicon.from_abtem_table();length=cell[2,2]+2.4e-9;dz=length/steps
  for iy,ix in ((0,0),(1,2),(4,3)):
   Q=2*np.pi*np.array([diff[ix]+r[0]/5,diff[iy]+r[1]/5])/np.diag(prim)[:2];K=np.column_stack((np.tile(Q,(fields,1)),kz))
   summ=np.zeros(fields,complex)
   for a in range(25):
    shifted=pos[a]+[0,0,1.2e-9-dz/2]
    summ+=-1j*(K@d[a,:,0].conj())*np.exp(-1j*K@shifted-.5*np.einsum('fa,ab,fb->f',K,cov[a],K))
   direct=np.fft.ifft(summ*pot.fourier_energy_j_m3(np.sum(K*K,axis=1))/(cell[0,0]*cell[1,1]*length))*fields
   np.testing.assert_allclose(prof[0,:,iy*5+ix],direct,rtol=2e-12,atol=1e-33)
 def test_distorted_source_weak_nonlinear_limit(self):
  from yoshioka.sector_screen import source_profiles,distorted_sector
  prim=np.diag([5.,4.,6.])*1e-10;cells=np.array(list(np.ndindex(5,5,1)));pos=cells@prim+np.array([.7,1.1,2.])*1e-10
  cell=prim*np.array([5,5,1])[:,None];cov=np.tile(np.eye(3)*.01e-20,(25,1,1));w=np.array([8e13]);d=np.tile(np.array([1.,2.,3.])*1e-14,(25,1))[:,:,None]
  b=dict(cell_m=cell,positions_m=pos,traveling_displacements_m=d,local_covariance_m2=cov)
  bath=SimpleNamespace(displacement_m=np.zeros((1,3,1)),omega_rad_s=w,inactive_covariance_m2=cov[:1],labels=['z'],cell_m=prim,positions_m=pos[:1])
  with tempfile.TemporaryDirectory() as folder:
   steps=257
   h=ProjectedMaterialAction(None,0,3,steps,257,folder,bath=bath)
   prof,r=source_profiles(b,[0],[dict(q_index=[0,0,0])],3,steps,257)
   linear,q,audit=distorted_sector(h,prof,r,w,order=8)
   bath.displacement_m=d[:1].real;bath.inactive_covariance_m2=cov[:1]-np.einsum('iam,ibm->iab',d[:1].real,d[:1].real)
   nonlinear=LongitudinalMaterialAction(bath,1,3,steps,257,folder,minimum_free_bytes=0)
   wave=np.zeros(nonlinear.shape,complex);wave[0,1,1]=1
   for i in range(steps):wave=nonlinear.exponential_action(wave,i)
   error=np.linalg.norm(wave[1]-linear[0])/np.linalg.norm(linear[0])
   self.assertLess(error,2e-5)
   self.assertLess(audit['incident_norm_drift'],1e-12)

class SectorEquivalenceTests(unittest.TestCase):
 def test_distorted_full_beams_equal_selected_sectors(self):
  from yoshioka.sector_screen import distorted_sector
  from yoshioka.free_propagation import electron_wavenumber
  from yoshioka.elastic_propagation import electron_speed
  from scipy.linalg import expm
  rng=np.random.default_rng(78);n=3;pad=5;steps=3;energy=60000*c.e;v=electron_speed(energy);hv=c.hbar*v;dz=2e-12
  cell=np.diag([5.,4.,6.])*1e-10;beam=np.arange(-1,2);qx,qy=np.meshgrid(2*np.pi*beam/cell[0,0],2*np.pi*beam/cell[1,1]);k=(qx*qx+qy*qy)/(2*electron_wavenumber(energy))
  position=rng.normal(size=(pad,pad))*.02*hv/dz;U=np.fft.fft2(position)/pad**2
  B=(rng.normal(size=(pad,pad))+1j*rng.normal(size=(pad,pad)))*.002*hv/dz
  h=SimpleNamespace(n=n,pad=pad,steps=steps,field_points=steps,beam_indices=beam%pad,velocity=v,energy_j=energy,cell_m=cell,dz=dz,kinetic_j=hv*k,potential=position.reshape(-1,1,1),set_slice=lambda _:None)
  omega=np.array([8e13]);r=np.array([1,-1]);profiles=np.tile(B.ravel(),(1,steps,1))
  result,q,audit=distorted_sector(h,profiles,r,omega,order=10)
  full=np.arange(-5,6);xx,yy=np.meshgrid(full,full);m=np.column_stack((xx.ravel(),yy.ravel()));dim=len(m)
  fullU=np.zeros((dim,dim),complex);fullB=fullU.copy()
  for i in range(dim):
   for j in range(dim):
    delta=m[i]-m[j]
    if np.all(delta%5==0):
     dx,dy=delta//5
     if abs(dx)<=2 and abs(dy)<=2:fullU[i,j]=U[dy%pad,dx%pad]
    if np.all((delta-r)%5==0):
     dx,dy=(delta-r)//5
     if abs(dx)<=2 and abs(dy)<=2:fullB[i,j]=B[dy%pad,dx%pad]
  local=np.block([[fullU,np.zeros_like(fullU)],[fullB,fullU]])
  exact=expm(-1j*dz/hv*local)
  fullk=np.sum((2*np.pi*m/(5*np.diag(cell)[:2]))**2,axis=1)/(2*electron_wavenumber(energy))
  phase=np.exp(-.5j*dz*np.r_[fullk,fullk+omega[0]/v]);wave=np.zeros(2*dim,complex);wave[60]=1
  for _ in range(steps):wave=phase*(exact@(phase*wave))
  expanded=np.zeros((11,11),complex);mx,my=np.meshgrid(5*beam+r[0],5*beam+r[1]);valid=(abs(mx)<=5)&(abs(my)<=5)
  expanded[my[valid]+5,mx[valid]+5]=result[0][valid]
  np.testing.assert_allclose(wave[dim:].reshape(11,11),expanded,rtol=2e-12,atol=2e-15)

class JointMomentumTests(unittest.TestCase):
 def test_complex_phase_real_limit(self):
  from yoshioka.momentum_material_action import complex_phase_entries
  from yoshioka.coupled_channels import channel_basis
  b=channel_basis(2,2);xi=np.array([[.2,.3],[-.1,.4]])
  expanded=np.broadcast_to(xi[:,None,None,:],(2,len(b),len(b),2))
  np.testing.assert_allclose(complex_phase_entries(b,expanded),exact_phase_blocks(b,xi),atol=2e-15)
 def test_nonlinear_joint_sector_against_full_real_standing_modes(self):
  from yoshioka.momentum_material_action import MomentumMaterialAction
  from math import comb,factorial
  prim=np.diag([5.,4.,6.])*1e-10;cells=np.array(list(np.ndindex(5,5,1)));pos=cells@prim+np.array([.7,1.1,2.])*1e-10
  cell=prim*np.array([5,5,1])[:,None];q=np.array([1/5,0,0]);z=np.exp(2j*np.pi*cells@q)[:,None]*np.array([1.,2.,1.])*1e-12/5
  traveling=np.stack((z,z.conj()),axis=-1);real=np.stack((np.sqrt(2)*z.real,np.sqrt(2)*z.imag),axis=-1)
  cov=np.tile(np.eye(3)*.01e-20,(25,1,1));w=np.array([8e13,8e13]);base=dict(omega_rad_s=w,inactive_covariance_m2=cov,labels=['q','-q'],cell_m=cell,positions_m=pos)
  with tempfile.TemporaryDirectory() as folder:
   joint=MomentumMaterialAction(SimpleNamespace(displacement_m=traveling,**base),[[1,0,0],[4,0,0]],2,3,9,17,folder,minimum_free_bytes=0)
   full=LongitudinalMaterialAction(SimpleNamespace(displacement_m=real,**base),2,11,9,17,folder,minimum_free_bytes=0)
   basis=joint.basis;T=np.zeros((6,6),complex)
   for j,(np_,nm_) in enumerate(basis):
    for kp in range(np_+1):
     for km in range(nm_+1):
      ns=kp+km;nc=np_+nm_-ns;i=np.flatnonzero(np.all(basis==[nc,ns],axis=1))[0]
      T[i,j]+=comb(np_,kp)*comb(nm_,km)*1j**kp*(-1j)**km*np.sqrt(factorial(nc)*factorial(ns)/(factorial(np_)*factorial(nm_)))/np.sqrt(2)**(np_+nm_)
   np.testing.assert_allclose(T.conj().T@T,np.eye(6),atol=8e-16)
   rng=np.random.default_rng(90);wave=(rng.normal(size=joint.shape)+1j*rng.normal(size=joint.shape))*joint.valid
   embedded=joint.expand(wave);realwave=np.einsum('ij,jyx->iyx',T,embedded)
   reference=np.einsum('ji,jyx->iyx',T.conj(),full.apply(realwave,4))
   result=joint.expand(joint.apply(wave,4))
   error=np.linalg.norm(result-reference)/np.linalg.norm(reference)
   self.assertLess(error,2e-12)


class ConstructionCheckpointTests(unittest.TestCase):
 def test_longitudinal_interruption_resume_and_corruption(self):
  bath=SimpleNamespace(displacement_m=np.array([[[.02e-10],[0],[.01e-10]]]),omega_rad_s=np.array([8e13]),inactive_covariance_m2=np.array([np.eye(3)*.01e-20]),labels=['xz'],cell_m=np.diag([5.,4.,6.])*1e-10,positions_m=np.array([[.8,1.2,2.]])*1e-10)
  class Limits:
   def __init__(self):self.calls=0
   def check(self,*_):
    self.calls+=1
    if self.calls>3:raise TimeoutError('intentional construction interruption')
  from pathlib import Path
  with tempfile.TemporaryDirectory() as folder:
   with self.assertRaises(TimeoutError):LongitudinalMaterialAction(bath,2,5,9,17,folder,minimum_free_bytes=0,limits=Limits())
   resumed=LongitudinalMaterialAction(bath,2,5,9,17,folder,minimum_free_bytes=0)
   with tempfile.TemporaryDirectory() as other:
    direct=LongitudinalMaterialAction(bath,2,5,9,17,other,minimum_free_bytes=0)
    np.testing.assert_array_equal(resumed.profile,direct.profile)
   # A corrupted committed construction tile is rejected before reuse.
   target=Path(folder)/resumed.field_digest
   (target/'manifest.json').unlink();(target/'total.npy').rename(target/'total.tmp.npy')
   array=np.load(target/'total.tmp.npy',mmap_mode='r+');array[0,0,0,0]+=1e-20;array.flush();del array
   with self.assertRaises(ValueError):LongitudinalMaterialAction(bath,2,5,9,17,folder,minimum_free_bytes=0)

if __name__=='__main__':unittest.main()
