"""Independent finite-window Born quadrature from analytical Lobato plane kernels.

Inverse 1D transforms of Yukawa and squared-Yukawa kernels are convolved
analytically with the longitudinal Gaussian displacement. No numerical source
grid, finite-difference stencil, or channel propagator is used in this module.
"""

import numpy as np
from scipy.constants import elementary_charge,hbar,m_e
from scipy.integrate import quad,quad_vec
from scipy.special import log_ndtr

from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import electron_wavenumber


def smeared_plane_kernel(potential,qxy_m_inv,z_m,covariance_m2):
    q2=np.sum(qxy_m_inv**2,axis=-1)
    sigma=np.sqrt(covariance_m2[2,2])
    gamma=potential.b_angstrom2*1e-20/(4*np.pi**2)
    mu=np.sqrt(q2[:,None]+1/gamma)
    t_plus=np.exp(.5*(mu*sigma)**2-mu*z_m+log_ndtr(z_m/sigma-mu*sigma))
    t_minus=np.exp(.5*(mu*sigma)**2+mu*z_m+log_ndtr(-z_m/sigma-mu*sigma))
    gaussian=np.exp(-.5*(z_m/sigma)**2)/np.sqrt(2*np.pi)
    first=t_plus+t_minus
    derivative_mu=(mu*sigma**2-z_m)*t_plus+(mu*sigma**2+z_m)*t_minus-2*sigma*gaussian
    second=first-mu*derivative_mu
    first_z=mu*(t_minus-t_plus)
    second_z=mu**2*((mu*sigma**2-z_m)*t_plus-(mu*sigma**2+z_m)*t_minus)
    coefficient=-(2*np.pi*hbar**2/m_e)*potential.a_angstrom*1e-10
    transverse_dw=np.exp(-.5*(covariance_m2[0,0]*qxy_m_inv[:,0]**2+covariance_m2[1,1]*qxy_m_inv[:,1]**2))
    kernel=np.sum(coefficient*(first/(2*gamma*mu)+second/(4*gamma**2*mu**3)),axis=-1)*transverse_dw
    derivative=np.sum(coefficient*(first_z/(2*gamma*mu)+second_z/(4*gamma**2*mu**3)),axis=-1)*transverse_dw
    return kernel,derivative


def finite_window_born(potential,positions_m,displacements_m,covariance_m2,qxy_m_inv,omega_rad_s,kinetic_energy_j,area_m2,length_m,g=1,relative_tolerance=1e-10):
    k=electron_wavenumber(kinetic_energy_j);v=electron_speed(kinetic_energy_j)
    kappa=np.sum(qxy_m_inv**2,axis=-1)[None]/(2*k)+omega_rad_s[:,None]/v
    planes,inverse=np.unique(positions_m[:,2],return_inverse=True)
    coefficients=np.zeros((len(omega_rad_s),len(planes),3,len(qxy_m_inv)),dtype=complex)
    for plane in range(len(planes)):
        for atom in np.flatnonzero(inverse==plane):
            phase=np.exp(-1j*(qxy_m_inv@positions_m[atom,:2]))
            coefficients[:,plane]+=displacements_m[atom].conj().T[:,:,None]*phase[None,None]
    evaluations=0
    def integrand(z_angstrom):
        nonlocal evaluations
        evaluations+=1
        coupling=np.zeros(kappa.shape,dtype=complex)
        for plane,position in enumerate(planes):
            h,derivative=smeared_plane_kernel(potential,qxy_m_inv,z_angstrom*1e-10-position,covariance_m2)
            coupling-=((1j*qxy_m_inv[:,0][None]*coefficients[:,plane,0]+1j*qxy_m_inv[:,1][None]*coefficients[:,plane,1])*h[None]+coefficients[:,plane,2]*derivative[None])/area_m2
        # Scale to eV and angstrom for meaningful quadrature absolute tolerances.
        return coupling/elementary_charge*np.exp(1j*kappa*z_angstrom*1e-10)
    integral,error=quad_vec(integrand,0,length_m*1e10,epsabs=1e-12,epsrel=relative_tolerance,points=planes*1e10,quadrature="gk21")
    output=-1j*g/(hbar*v)*np.exp(-1j*kappa*length_m)*(integral*elementary_charge*1e-10)
    return output,{"quadrature":"independent adaptive Gauss-Kronrod in z using analytical Gaussian-convolved plane kernels","relative_tolerance":relative_tolerance,
                   "estimated_integral_error_ev_angstrom":float(error),"source_evaluations":evaluations,"energy_units":"J","longitudinal_wavevector_sign":"Qz=-kappa"}


def single_atom_fourier_audit(potential):
    tests=[]
    for angular_q_inv_angstrom in (0,1,3,10):
        def integrand(r):
            if r==0:return 0.0
            return 4*np.pi*r*r*float(potential.radial_energy_j(r*1e-10)/elementary_charge)*np.sinc(angular_q_inv_angstrom*r/np.pi)
        value,error=quad(integrand,0,40,epsabs=1e-10,epsrel=1e-11,limit=1000)
        numerical=value*elementary_charge*1e-30
        exact=float(potential.fourier_energy_j_m3((angular_q_inv_angstrom*1e10)**2))
        tests.append({"angular_Q_inv_angstrom":angular_q_inv_angstrom,"radial_quadrature_j_m3":numerical,"analytic_j_m3":exact,
                      "relative_error":abs(numerical-exact)/abs(exact),"negative_interaction_energy":bool(exact<0)})
    return {"Fourier_convention":"integral d^3r exp(-iQ.r) h(r), h=-e V; angular Q/(2*pi) converts to tabulated cycles/angstrom",
            "tests":tests,"maximum_relative_error":max(row['relative_error'] for row in tests)}
