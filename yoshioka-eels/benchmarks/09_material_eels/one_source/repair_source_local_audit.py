"""Independent quadrature check of the already-exact frozen-slice source step.

Incoming and outgoing local elastic potentials are the same scalar field.
They commute with multiplicative G, so the local Duhamel integral is a sinc.
This check isolates its integration; z variation/kinetic splitting still needs
the separate slice and joint refinements. No production amplitude is changed.
"""
import json
import numpy as np
from scipy import constants
from scipy.linalg import expm
from scipy.special import roots_legendre
from .workflow_io import ROOT,json_write
from .repair_bloch_analysis import collect
from yoshioka.elastic_propagation import electron_speed


def quadrature_factor(alpha,order):
    nodes,weights=roots_legendre(order)
    return np.exp(-.5j*np.asarray(alpha))*.5*np.sum(weights*np.exp(.5j*np.asarray(alpha)[...,None]*nodes),axis=-1)


def independent_control():
    # Dimensionless local time step with deliberately large channel phase.
    u=2.7;alpha=1.3;g=.4+.8j;incoming=.7-.2j;outgoing=-.1+.3j
    h=np.array([[u,0],[g,u+alpha]],complex)
    exact=expm(-1j*h)@np.array([incoming,outgoing])
    rows=[]
    for order in (2,4,8,16):
        result=np.exp(-1j*(u+alpha))*outgoing-1j*np.exp(-1j*u)*g*incoming*quadrature_factor(alpha,order)
        rows.append({'order':order,'complex_error':float(abs(result-exact[1]))})
    return rows


def main():
    rows=[];velocity=electron_speed(60000*constants.elementary_charge)
    for case in collect():
        s=case['physical_settings']
        if s['phonon_xy']!=2 or s['phonon_z']!=2 or s['thickness_repeats']!=52:continue
        state=json.loads(open(case['checkpoint']).read());omega=np.asarray(state['contract']['mode_energies'])
        alpha=omega*case['physical_spacings_angstrom']['dz']*1e-10/velocity
        analytical=np.exp(-.5j*alpha)*np.sinc(alpha/(2*np.pi))
        rows.append({'settings':s,'maximum_local_channel_phase':float(np.max(abs(alpha))),
            'quadrature_errors':[{ 'order':order,'max_absolute_complex_error':float(np.max(abs(quadrature_factor(alpha,order)-analytical)))} for order in (2,4,8,16)]})
    result={'status':'PASS' if rows and max(r['complex_error'] for r in independent_control() if r['order']>=8)<1e-12 and all(r['quadrature_errors'][-1]['max_absolute_complex_error']<1e-12 for r in rows) else 'INCOMPLETE',
        'independent_dense_control':independent_control(),'material_frozen_slice_checks':rows,
        'scope':'independent local source integration at fixed electron slices; existing production uses exact sinc; no midpoint-G z-variation or Strang-error cancellation inferred from this local check'}
    output=ROOT/'benchmark_results/repair_20261102_v2/B9A_source_integration_audit.json';json_write(output,result);print(result['status'])


if __name__=='__main__':main()
