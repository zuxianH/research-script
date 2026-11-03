"""SI Lobato transition potentials and actual first-order longitudinal channels.

The lattice MOD potential is never used here as an electron interaction.
Transverse channel Fourier modes decouple in the undepleted first-Born source
problem, so a predeclared detector/momentum region can be propagated exactly
without allocating unobserved electron Fourier modes. No phonon modes are cut.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge, hbar, m_e

from .elastic_propagation import electron_speed
from .free_propagation import FreePropagator, TransverseGrid, electron_wavenumber


@dataclass
class LobatoSilicon:
    a_angstrom: np.ndarray
    b_angstrom2: np.ndarray
    source_path: str = ""

    @classmethod
    def from_abtem_table(cls):
        import importlib.util
        package=importlib.util.find_spec('abtem')
        if package is None:raise RuntimeError('the validated Lobato parameter table requires the abTEM reference dependency')
        path = Path(package.origin).parent/"parametrizations/data/lobato.json"
        values = np.array(json.loads(path.read_text())["Si"])
        return cls(values[0], values[1], str(path))

    def fourier_energy_j_m3(self, angular_q_squared_m_inv2):
        """h_tilde(Q)=-(2*pi*hbar²/m_e) f_Lobato(Q/(2*pi)).

        Fourier transform is integral d³r h(r) exp(-iQ.r). Tabulated scattering
        factors use cycles/angstrom and return angstrom, converted exactly once.
        """
        squared = np.asarray(angular_q_squared_m_inv2)*1e-20/(4*np.pi**2)
        denominator = 1 + squared[..., None]*self.b_angstrom2
        f_m = np.sum(self.a_angstrom*(1/denominator + 1/denominator**2), axis=-1)*1e-10
        return -(2*np.pi*hbar**2/m_e)*f_m

    def radial_energy_j(self, radius_m):
        radius = np.asarray(radius_m)*1e10
        b = 2*np.pi/np.sqrt(self.b_angstrom2)
        a = -(2*np.pi*hbar**2/m_e)*1e20*np.pi**2*self.a_angstrom/self.b_angstrom2**1.5
        return np.sum(a*(2/(radius[..., None]*b)+1)*np.exp(-radius[..., None]*b), axis=-1)


def fourth_order_derivative(array, spacing_m, axis=-1):
    """Actual periodic four-point Cartesian derivative, independent of iQ."""
    return (-np.roll(array,-2,axis=axis)+8*np.roll(array,-1,axis=axis)-8*np.roll(array,1,axis=axis)+np.roll(array,2,axis=axis))/(12*spacing_m)


def derivative_symbol(angular_q_m_inv, spacing_m):
    """Fourier symbol of fourth_order_derivative, also for Bloch boundaries."""
    x = np.asarray(angular_q_m_inv)*spacing_m
    return 1j*(8*np.sin(x)-np.sin(2*x))/(6*spacing_m)


def transverse_modes(grid, kinetic_energy_j, max_angle_mrad, bloch_shift_m_inv=(0,0)):
    qx, qy = np.meshgrid(2*np.pi*np.fft.fftfreq(grid.nx,grid.dx_m),2*np.pi*np.fft.fftfreq(grid.ny,grid.dy_m))
    regular = np.stack((qx,qy),axis=-1).reshape(-1,2)
    physical = regular+np.asarray(bloch_shift_m_inv)
    angle = np.linalg.norm(physical,axis=-1)/electron_wavenumber(kinetic_energy_j)*1000
    retained = angle <= max_angle_mrad+1e-12
    return physical[retained], np.flatnonzero(retained), angle[retained]


def common_diagonal_covariance(displacements_m):
    covariance = np.einsum("iam,ibm->iab", displacements_m, displacements_m.conj()).real
    common = np.mean(covariance,axis=0)
    discrepancy = float(np.max(np.linalg.norm(covariance-common,axis=(-2,-1)))/np.linalg.norm(common))
    off_diagonal = common-np.diag(np.diag(common))
    relative_off = float(np.linalg.norm(off_diagonal)/np.linalg.norm(common))
    if discrepancy > 1e-9 or relative_off > 1e-9:
        raise ValueError("this focused Si source implementation requires verified equivalent sites and diagonal covariance; no tensor components are silently discarded")
    return np.diag(np.diag(common)), {"site_equivalence_relative_residual": discrepancy,"off_diagonal_relative_residual":relative_off,
                                    "covariance_m2":common.tolist(),"all_physical_modes_used":displacements_m.shape[-1]}


def mode_plane_coefficients(positions_m, displacements_m, qxy_m_inv):
    """Actual finite atom sum, grouped only by equal z for efficient evaluation."""
    planes, inverse = np.unique(positions_m[:,2],return_inverse=True)
    coefficients = np.zeros((displacements_m.shape[-1],len(planes),3,len(qxy_m_inv)),dtype=np.complex128)
    for plane in range(len(planes)):
        selected = inverse==plane
        phase = np.exp(-1j*(positions_m[selected,:2]@qxy_m_inv.T))
        coefficients[:,plane] = np.einsum("iam,iq->maq",displacements_m[selected].conj(),phase)
    return planes,coefficients


def numerical_transition_source(potential, positions_m, displacements_m, covariance_m2, qxy_m_inv, grid, length_m, longitudinal_samples, kernel_cache=None):
    """Construct coupling from an inverse-FFT atomic potential and FD gradients.

    The atomic potential is DW-convolved in reciprocal space, inverse transformed
    along z, and differentiated using a real-space fourth-order stencil. The
    transverse stencil is evaluated by its exactly equivalent Fourier symbol.
    The independently evaluated Born reference uses analytical plane kernels
    and exact Cartesian derivatives, not these stencil coefficients.
    """
    samples = int(longitudinal_samples)
    if samples < 8 or samples%2:
        raise ValueError("longitudinal_samples must be even and >=8")
    area = grid.lx_m*grid.ly_m
    dz = length_m/samples
    z = np.arange(samples)*dz
    kz = 2*np.pi*np.fft.fftfreq(samples,dz)
    planes,coefficients = mode_plane_coefficients(positions_m,displacements_m,qxy_m_inv)
    cache_key=(qxy_m_inv.tobytes(),planes.tobytes(),covariance_m2.tobytes(),potential.a_angstrom.tobytes(),potential.b_angstrom2.tobytes(),length_m,samples,grid.dx_m,grid.dy_m,area)
    kernels=None if kernel_cache is None else kernel_cache.get(cache_key)
    if kernels is None:
        q_squared = np.sum(qxy_m_inv**2,axis=1)[:,None]+kz[None]**2
        dw = np.exp(-.5*(covariance_m2[0,0]*qxy_m_inv[:,0,None]**2+covariance_m2[1,1]*qxy_m_inv[:,1,None]**2+covariance_m2[2,2]*kz[None]**2))
        fourier = potential.fourier_energy_j_m3(q_squared)*dw
        kernels=[]
        for position in planes:
            field=np.fft.ifft(fourier*np.exp(-1j*kz[None]*position),axis=-1)/(area*dz)
            kernels.append((field,fourth_order_derivative(field,dz,axis=-1)))
        if kernel_cache is not None:
            if len(kernel_cache)>=16:kernel_cache.pop(next(iter(kernel_cache)))
            kernel_cache[cache_key]=kernels
    source = np.zeros((displacements_m.shape[-1],len(qxy_m_inv),samples),dtype=np.complex128)
    dx = derivative_symbol(qxy_m_inv[:,0],grid.dx_m)
    dy = derivative_symbol(qxy_m_inv[:,1],grid.dy_m)
    for plane,(field,derivative_z) in enumerate(kernels):
        transverse = coefficients[:,plane,0]*dx+coefficients[:,plane,1]*dy
        source -= transverse[:,:,None]*field[None]+coefficients[:,plane,2,:,None]*derivative_z[None]
    return z,source


class FirstOrderSourcePropagator:
    """Step longitudinal channel waves using exact free evolution and Simpson source.

    Uses the existing Benchmark 1 Fourier propagation multiplier and a common
    carrier/velocity. Samples at each endpoint and midpoint are independently
    supplied source values. This advances initially empty channels; it is not
    an evaluation of the final Born formula.
    """

    def __init__(self,grid,kinetic_energy_j,omega_rad_s,indices,bloch_shift_m_inv=(0,0)):
        self.grid=grid
        self.free=FreePropagator(grid,electron_wavenumber(kinetic_energy_j))
        self.velocity=electron_speed(kinetic_energy_j)
        self.omega=np.asarray(omega_rad_s)
        self.indices=np.asarray(indices,dtype=int)
        self.shift=np.asarray(bloch_shift_m_inv)
        qx,qy=np.meshgrid(2*np.pi*np.fft.fftfreq(grid.nx,grid.dx_m),2*np.pi*np.fft.fftfreq(grid.ny,grid.dy_m))
        regular=np.stack((qx,qy),axis=-1).reshape(-1,2)[self.indices]
        self.bloch_q2_difference=np.sum((regular+self.shift)**2-regular**2,axis=-1)

    def multiplier(self,distance_m):
        transverse=self.free.multiplier(distance_m).ravel()[self.indices]
        return transverse[None]*np.exp(-1j*distance_m*(self.omega[:,None]/self.velocity+self.bloch_q2_difference[None]/(2*self.free.wavenumber_m_inv)))

    def propagate(self,source_j,length_m,g=1.0,record_history=False):
        source=np.asarray(source_j,dtype=np.complex128)
        if source.shape[:2]!=(len(self.omega),len(self.indices)) or source.shape[-1]%2:
            raise ValueError("source must have shape (channels,retained q,even z samples)")
        dz=2*length_m/source.shape[-1]
        full,half=self.multiplier(dz),self.multiplier(dz/2)
        channels=np.zeros(source.shape[:2],dtype=np.complex128)
        history=[0.0]
        prefactor=-1j*g/(hbar*self.velocity)*dz/6
        for index in range(0,source.shape[-1],2):
            next_source=source[:,:,(index+2)%source.shape[-1]]
            channels=full*channels+prefactor*(full*source[:,:,index]+4*half*source[:,:,index+1]+next_source)
            if record_history:history.append(float(np.sum(abs(channels)**2)))
        return channels,np.array(history)

    def propagate_scan(self,source_j,length_m,g=1.0):
        """Vectorized prefix-scan of exactly the same discrete source steps.

        This changes neither the sampled potential nor Simpson integration.
        It is useful when thousands of independent Bloch sectors must be
        propagated. The ordinary stepper remains the independent control.
        """
        source=np.asarray(source_j,dtype=np.complex128)
        if source.shape[:2]!=(len(self.omega),len(self.indices)) or source.shape[-1]%2:
            raise ValueError('source must contain all endpoint/midpoint pairs')
        count=source.shape[-1]//2
        distance=length_m/count
        full,half=self.multiplier(distance),self.multiplier(distance/2)
        even=source[:,:,::2]
        increments=-1j*g/(hbar*self.velocity)*distance/6*(full[:,:,None]*even+4*half[:,:,None]*source[:,:,1::2]+np.roll(even,-1,axis=-1))
        powers=np.arange(1,count+1)
        logarithm=np.log(full)
        prefix=np.cumsum(increments*np.exp(-logarithm[:,:,None]*powers),axis=-1)
        return np.exp(logarithm*count)*prefix[:,:,-1]


def atomic_transition_fourier(potential,positions_m,displacements_m,covariance_m2,vectors_m_inv):
    """Independent exact atomic sum for <1_lambda|h|0> in J m^3."""
    vectors=np.asarray(vectors_m_inv)
    atomic_factor=potential.fourier_energy_j_m3(np.sum(vectors**2,axis=1))*np.exp(-.5*np.einsum('qa,ab,qb->q',vectors,covariance_m2,vectors))
    phase=np.exp(-1j*positions_m@vectors.T)
    polarization=np.einsum('qa,iam->miq',vectors,displacements_m.conj())
    return -1j*atomic_factor[None]*np.einsum('iq,miq->mq',phase,polarization)


def full_space_born(potential,positions_m,displacements_m,covariance_m2,qxy_m_inv,omega_rad_s,kinetic_energy_j,area_m2,length_m,g=1):
    """Independent atomic-sum Born amplitude at Qz=-kappa, retaining d_z."""
    k=electron_wavenumber(kinetic_energy_j);v=electron_speed(kinetic_energy_j)
    kappa=np.sum(qxy_m_inv**2,axis=-1)[None]/(2*k)+omega_rad_s[:,None]/v
    output=np.zeros(kappa.shape,dtype=np.complex128)
    for mode in range(len(omega_rad_s)):
        vectors=np.column_stack((qxy_m_inv,-kappa[mode]))
        coupling=atomic_transition_fourier(potential,positions_m,displacements_m[:,:,mode:mode+1],covariance_m2,vectors)[0]
        output[mode]=-1j*g/(hbar*v*area_m2)*np.exp(-1j*kappa[mode]*length_m)*coupling
    return output
