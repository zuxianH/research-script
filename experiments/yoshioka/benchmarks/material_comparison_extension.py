"""Matched reduced-Si frozen quantum, harmonic-MD and linear FRFPMS controls.

This is a fully declared *two-active-mode harmonic* material model. It neither
calibrates old TACAW arrays nor supplies anharmonic/TRPMD dynamics. Original
arrays are read-only and their missing metadata remain blocking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.special import roots_hermite, ndtri
from scipy.stats import qmc

from benchmarks.workflow_io import ResourceLimits, environment, json_write, sha256
from yoshioka.harmonic_oscillator import normalized_hermite_polynomials
from yoshioka.leading_reference import DenseDisplacedMaterial, gamma_material_bath, thermal_initial_weights
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.spectral_data import kubo_factor

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1/thermal_comparison'


class FrozenMaterialCoordinates:
    """Independent coordinate-space frozen electron propagation for Gamma modes.

    The quantum matrix reference projects each displaced atom on Fock states
    before z propagation. Here the electron is propagated at fixed continuous
    coordinates first and then projected by independent Gauss-Hermite integrals.
    """
    def __init__(self,bath,n=3,steps=65):
        self.reference=DenseDisplacedMaterial(bath,p=0,n=n,steps=steps,phonon_evolution=False)
        self.n=n; self.steps=steps; self.bath=bath
        self.ne=n*n; self.xi=[]; fields=[]
        for coefficient,xi,source,target in self.reference.terms:
            if np.max(abs(xi-xi[0]))>1e-12:
                raise ValueError('coordinate fast path requires transverse Gamma coordinates')
            self.xi.append(xi[0]); field=np.zeros((steps,self.ne,self.ne),complex)
            profile=np.fft.ifft(coefficient)
            for src,dst in zip(source,target): field[:,dst,src]+=profile
            fields.append(field)
        self.xi=np.asarray(self.xi);self.fields=np.asarray(fields)

    def propagate(self,coordinates,batch_size=128):
        coordinates=np.atleast_2d(coordinates)
        if coordinates.shape[1]!=len(self.bath.omega_rad_s):raise ValueError('coordinate count mismatch')
        result=np.empty((len(coordinates),self.ne),complex)
        for lo in range(0,len(coordinates),batch_size):
            x=coordinates[lo:lo+batch_size];phase=np.exp(-1j*x@self.xi.T)
            state=np.zeros((len(x),self.ne),complex);state[:,self.ne//2]=1
            for z in range(self.steps):
                matrices=(phase@self.fields[:,z].reshape(len(self.xi),-1)).reshape(len(x),self.ne,self.ne)
                ii=np.arange(self.ne);matrices[:,ii,ii]+=self.reference.free_diagonal
                norm=max(np.linalg.norm(matrices),1e-300)
                if np.linalg.norm(matrices-matrices.swapaxes(-1,-2).conj())/norm>1e-11:
                    raise ValueError('non-Hermitian coordinate potential')
                energy,vectors=np.linalg.eigh(matrices)
                coeff=np.einsum('qji,qj->qi',vectors.conj(),state)
                state=np.einsum('qij,qj->qi',vectors,coeff*np.exp(-1j*self.reference.dz*energy/self.reference.hv))
            result[lo:lo+len(x)]=state
        return result

    def fock_operator(self,p,order):
        from yoshioka.coupled_channels import channel_basis
        basis=channel_basis(len(self.bath.omega_rad_s),p)
        nodes,weights=roots_hermite(order)
        # The bounded matched model has two active coordinates.
        x,y=np.meshgrid(nodes,nodes,indexing='ij')
        coordinates=np.sqrt(2)*np.column_stack((x.ravel(),y.ravel()))
        quadrature=(weights[:,None]*weights[None]/np.pi).ravel()
        hermite=normalized_hermite_polynomials(nodes,p)
        phi=hermite[basis[:,0]][:,np.repeat(np.arange(order),order)]*hermite[basis[:,1]][:,np.tile(np.arange(order),order)]
        scattered=self.propagate(coordinates)
        matrices=np.einsum('aq,bq,qe,q->eab',phi,phi,scattered,quadrature,optimize=True)
        return basis,matrices


def transition_lines(operator,basis,omega,temperature,detector):
    probabilities,tail=thermal_initial_weights(omega,temperature,basis)
    channel_energy=basis@(constants.hbar*omega)/(1e-3*constants.elementary_charge)
    energies=channel_energy[:,None]-channel_energy[None,:]
    weights=np.sum(abs(operator[detector])**2,axis=0)*probabilities[None,:]
    quantum=float(np.mean(constants.hbar*omega)/(1e-3*constants.elementary_charge))
    if temperature>0 and len(omega)==2 and np.ptp(omega)<1e-10*np.mean(omega):
        ratio=np.exp(-constants.hbar*np.mean(omega)/(constants.Boltzmann*temperature));p=int(np.max(np.sum(basis,axis=1)))
        tail=max(tail,float(ratio**(p+1)*((p+2)-(p+1)*ratio)))
    ell=np.rint(energies/quantum).astype(int)
    if np.max(abs(energies-ell*quantum))>1e-8:raise ValueError('only degenerate two-mode line assignment supported')
    indices=np.arange(-np.max(np.sum(basis,axis=1)),np.max(np.sum(basis,axis=1))+1,dtype=int)
    lines=np.array([np.sum(weights[ell==i]) for i in indices])
    return indices,lines,tail,probabilities


def dense_trace_correlation(operator,channel_energy_mev,probabilities,times_ps,detector):
    output=[]
    for t in times_ps:
        evolution=np.diag(np.exp(-1j*channel_energy_mev*1e-3*constants.elementary_charge*t*1e-12/constants.hbar))
        value=0j
        for scatter in operator[detector]:
            # Independent trace evaluation of rho S† exp(-iHt) S exp(+iHt).
            product=scatter.conj().T@evolution@scatter@evolution.conj().T
            value+=np.sum(probabilities*np.diag(product))
        output.append(value)
    return np.array(output)


def sampled_classical_lines(frozen,temperature,scramble_seed,samples=32,time_count=64):
    """Analytic harmonic MD; independently scrambled initial phase-space samples.

    Exact nuclear solutions are used instead of a time integrator. A complete
    oscillator period makes the Fourier bins exact multiples of its frequency;
    no physical linewidth or finite-resolution extrapolation is inferred.
    """
    if samples&(samples-1):raise ValueError('Sobol sample count must be a power of two')
    omega=frozen.bath.omega_rad_s
    if np.ptp(omega)>1e-10*np.mean(omega):raise ValueError('equal-frequency period required')
    occupation=constants.Boltzmann*temperature/(constants.hbar*omega)
    normal=ndtri(qmc.Sobol(4,scramble=True,seed=scramble_seed).random_base2(int(np.log2(samples))))
    a=normal[:,:2]*np.sqrt(2*occupation);b=normal[:,2:]*np.sqrt(2*occupation)
    phase=2*np.pi*np.arange(time_count)/time_count
    coordinates=a[:,None,:]*np.cos(phase)[None,:,None]+b[:,None,:]*np.sin(phase)[None,:,None]
    scattered=frozen.propagate(coordinates.reshape(-1,2)).reshape(samples,time_count,frozen.ne)
    fourier=np.fft.fft(scattered,axis=1)/time_count
    ell=np.fft.fftfreq(time_count)*time_count
    order=np.argsort(ell)
    return ell[order].astype(int),np.mean(abs(fourier[:,order])**2,axis=0),{'samples':samples,'independent_scramble_seed':scramble_seed,
        'time_count':time_count,'nuclear_period_fs':2*np.pi/np.mean(omega)*1e15,
        'nuclear_solver':'exact analytic classical harmonic trajectory', 'duration':'one full period; discrete harmonic lines; no linewidth interpretation'}


def run_matched(output):
    folder=output/'matched_material';folder.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(600,4);started=time.monotonic()
    model_path=ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'
    model=RealSpaceForceConstants.load(model_path);bath=gamma_material_bath(model,2)
    quantum=float(np.mean(bath.omega_rad_s)*constants.hbar/(1e-3*constants.elementary_charge))
    config={'model':'actual-Si MOD-derived fixed harmonic modes + finite-z Lobato electron potential',
        'active_modes':bath.labels,'active_energies_mev':(bath.omega_rad_s*constants.hbar/(1e-3*constants.elementary_charge)).tolist(),
        'inactive_modes':'same vacuum-projected inactive Gaussian covariance in every method',
        'beam_energy_kev':60,'incident_probe':'unit-flux central reciprocal plane wave',
        'electron_beams_per_axis':3,'z_steps':65,'cell_m':bath.cell_m.tolist(),
        'nuclear_slab_thickness_m':float(bath.cell_m[2,2]),'electron_padding_m_per_side':12e-10,
        'detectors':{'all_beams':'all9retainedreciprocalbeams','noncentral':'8retainedbeams excluding incidentcentral'},
        'coordinate_convention':'X=a+a†; atomic u=sum_m d_m X_m; angular electron momentum2pi*h/cell',
        'quantum_projection_cutoffs':[8,12],'coordinate_quadrature_orders':[20,28],
        'classical_scramble_seeds':[1851,2843,3851,4877],'samples_per_scramble':32,'samples_per_period':64,
        'temperature_k':[300,1000],'spectral_assignment':'exact degenerate oscillator harmonics; signed loss positive',
        'comparison_gates':{'quadrature_absolute_line_L1':1e-8,'correlation_absolute':1e-12,'linear_FRFPMS_relative':1e-8},
        'FRFPMS_scope':'linearized frozen scattering only, derived harmonic one-phonon regime; no nonlinear exactness claim',
        'normalization':'absolute probability per unit incident electron; no fitted scale, energy offset, global renormalization or broadening'}
    json_write(folder/'contract.json',{'config':config,'harmonic_model_sha256':sha256(model_path),
        'source_sha256':{name:sha256(ROOT/name) for name in ['benchmarks/material_comparison_extension.py','yoshioka/leading_reference.py']},'environment':environment()})
    frozen=FrozenMaterialCoordinates(bath);records=[]
    operators={}
    for p,order in [(8,20),(12,20),(12,28)]:
        limits.check();clock=time.monotonic();basis,operator=frozen.fock_operator(p,order);operators[(p,order)]=(basis,operator)
        np.savez_compressed(folder/f'frozen_p{p}_quadrature{order}.npz',basis=basis,operator=operator)
        records.append({'stage':'coordinate_quantum_operator','p':p,'quadrature_order':order,'elapsed_seconds':time.monotonic()-clock})
        print('matched frozen',p,order,records[-1]['elapsed_seconds'],flush=True)
    basis,operator=operators[(12,28)]
    detectors={'all_beams':np.ones(frozen.ne,dtype=bool),'noncentral':np.arange(frozen.ne)!=frozen.ne//2}
    result={'status':'INCOMPLETE','config':config,'comparisons':[],'timings':records,'full_channel':[]}
    for temperature in config['temperature_k']:
        classical=[]
        for seed in config['classical_scramble_seeds']:
            limits.check();clock=time.monotonic();ell,weights,meta=sampled_classical_lines(frozen,temperature,seed,32,64)
            classical.append(weights);np.savez_compressed(folder/f'classical_T{temperature}_seed{seed}.npz',ell=ell,weights=weights)
            records.append({'stage':'classical_harmonic_MD_frozen','temperature_k':temperature,**meta,'elapsed_seconds':time.monotonic()-clock})
            print('matched classical',temperature,seed,records[-1]['elapsed_seconds'],flush=True)
        classical=np.asarray(classical)
        correction=kubo_factor(ell*quantum,temperature)
        corrected=classical*correction[None,:,None]
        for name,detector in detectors.items():
            qell,qline,tail,prob=transition_lines(operator,basis,bath.omega_rad_s,temperature,detector)
            _,prior,_,_=transition_lines(operators[(12,20)][1],basis,bath.omega_rad_s,temperature,detector)
            pell,pline,ptail,_=transition_lines(operators[(8,20)][1],operators[(8,20)][0],bath.omega_rad_s,temperature,detector)
            cutoff_change=np.sum(abs(qline[np.isin(qell,pell)]-pline))+np.sum(qline[~np.isin(qell,pell)])
            times=np.linspace(0,.12,9)
            trace=dense_trace_correlation(operator,basis@(constants.hbar*bath.omega_rad_s)/(1e-3*constants.elementary_charge),prob,times,detector)
            line_correlation=np.exp(-1j*times[:,None]*1e-12*(qell[None,:]*quantum*1e-3*constants.elementary_charge)/constants.hbar)@qline
            common=abs(ell)<=12
            qcommon=np.array([qline[qell==value][0] for value in ell[common]])
            raw=np.sum(classical[:,:,detector],axis=-1)
            corrected_detector=np.sum(corrected[:,:,detector],axis=-1)
            mean=corrected_detector.mean(axis=0);se=corrected_detector.std(axis=0,ddof=1)/np.sqrt(len(classical))
            row={'temperature_k':temperature,'detector':name,'quantum_thermal_initial_tail':tail,
                'frozen_final_and_initial_occupation_cutoff_line_L1_change_8_to12':float(cutoff_change),
                'quantum_quadrature20_to28_absolute_line_L1':float(np.sum(abs(prior-qline))),
                'quantum_correlation_vs_channel_absolute_error':float(np.max(abs(trace-line_correlation))),
                'quantum_total_represented_probability':float(np.sum(qline)),
                'classical_Kubo_corrected_total_probability':float(np.sum(mean)),
                'classical_Kubo_vs_quantum_absolute_line_L1':float(np.sum(abs(mean[common]-qcommon))),
                'classical_Kubo_multiphonon_absolute_line_L1':float(np.sum(abs(mean[common & (abs(ell)>=2)]-qcommon[abs(ell[common])>=2]))),
                'classical_scramble_standard_error_integrated':float(np.std(np.sum(corrected_detector,axis=1),ddof=1)/np.sqrt(len(classical))),
                'uncertainty_scope':'four independent phase-space scrambles; time Fourier points and within-scramble phase-space nodes are correlated; no95%normality claim',
                'interpretation':'classical nonlinear frozen correlation times Kubo factor is an approximation; normalization difference retained'}
            result['comparisons'].append(row)
            np.savez_compressed(folder/f'T{temperature}_{name}_comparison.npz',quantum_ell=qell,quantum_lines=qline,classical_ell=ell,
                classical_raw_batches=raw,classical_corrected_batches=corrected_detector,corrected_line_standard_error=se,times_ps=times,
                correlation_trace=trace,correlation_lines=line_correlation)
        # FRFPMS is tested only for the explicitly linearized material operator.
        delta=.001;zero=frozen.propagate([[0.,0.]])[0];derivatives=[];derivative_refined=[]
        for mode in range(2):
            shift=np.zeros((2,2));shift[:,mode]=[delta,-delta]
            wave=frozen.propagate(shift);derivatives.append((wave[0]-wave[1])/(2*delta))
            wave=frozen.propagate(shift/2);derivative_refined.append((wave[0]-wave[1])/delta)
        derivative=np.asarray(derivative_refined)
        for name,detector in detectors.items():
            norm=np.sum(abs(derivative[:,detector])**2,axis=1)
            ncl=constants.Boltzmann*temperature/(constants.hbar*bath.omega_rad_s)
            nq=1/np.expm1(constants.hbar*bath.omega_rad_s/(constants.Boltzmann*temperature))
            quantum_linear=np.array([np.sum(nq*norm),np.sum((nq+1)*norm)])
            # Each isolated frequency-bin variance is 2*ncl*|D|².
            bin_variance=2*ncl*norm
            classical_lines=np.array([np.sum(bin_variance)/2]*2)
            frfpms_corrected=classical_lines*kubo_factor(np.array([-quantum,quantum]),temperature)
            result.setdefault('linear_FRFPMS',[]).append({'temperature_k':temperature,'detector':name,
                'isolated_bin_incoherent_variance':bin_variance.tolist(),'raw_classical_gain_loss':classical_lines.tolist(),
                'Kubo_corrected_gain_loss':frfpms_corrected.tolist(),'quantum_linear_gain_loss':quantum_linear.tolist(),
                'relative_error':float(np.linalg.norm(frfpms_corrected-quantum_linear)/max(np.linalg.norm(quantum_linear),1e-300)),
                'derivative_step_halving_relative_change':float(np.linalg.norm(np.asarray(derivatives)-derivative)/np.linalg.norm(derivative)),
                'outside_bin_treatment':'all other active coordinates fixed at mean0; same inactive-vacuum DW field in S0',
                'background':'coherent |S0|² separated; bin variance excludes S0; quadratic DW shifts and nonlinear cross-bin terms outside this validated linear comparison',
                'spectral_assignment':'half uncorrected classical bin variance at each signed harmonic energy, then one Kubo factor',
                'scope':'exact only for declared harmonic linearized frozen scattering; not fully corrected nonlinear FRFPMS'})
        json_write(folder/'checkpoint.json',result)
    # Actual coupled-channel propagation retains Hphonon. This measures the
    # frozen-electron approximation at the same declared material/electron model.
    for temperature,p in [(300,6),(300,8),(1000,8),(1000,10)]:
        limits.check();clock=time.monotonic()
        h=DenseDisplacedMaterial(bath,p=p,n=3,steps=65,phonon_evolution=True,maximum_bytes=3*1024**3)
        initial=np.column_stack([h.initial_state(state) for state in h.basis])
        wave=h.propagate(initial=initial).reshape(len(h.basis),frozen.ne,len(h.basis)).transpose(1,0,2)
        full_record={'temperature_k':temperature,'p':p,'elapsed_seconds':time.monotonic()-clock,'detectors':[]}
        for name,detector in detectors.items():
            fell,fline,tail,_=transition_lines(wave,h.basis,bath.omega_rad_s,temperature,detector)
            fbas,fop=operators[(8,20)] if p==8 else frozen.fock_operator(p,20)
            _,frozen_lines,_,_=transition_lines(fop,fbas,bath.omega_rad_s,temperature,detector)
            full_record['detectors'].append({'detector':name,'thermal_initial_tail':tail,
                'absolute_line_L1_full_vs_frozen':float(np.sum(abs(fline-frozen_lines))),
                'full_total_represented_probability':float(np.sum(fline))})
            np.savez_compressed(folder/f'full_T{temperature}_p{p}_{name}.npz',ell=fell,full_lines=fline,frozen_lines=frozen_lines)
        result['full_channel'].append(full_record);json_write(folder/'checkpoint.json',result)
        print('matched full',temperature,p,full_record['elapsed_seconds'],flush=True)
    result['status']='VERIFIED_REDUCED_HARMONIC_MATERIAL_COMPARISON' if (
        all(r['quantum_correlation_vs_channel_absolute_error']<1e-12 and r['quantum_quadrature20_to28_absolute_line_L1']<1e-8 for r in result['comparisons']) and
        all(r['relative_error']<1e-8 for r in result['linear_FRFPMS'])) else 'FAILED_IMPLEMENTATION_GATES'
    result['external_TACAW_status']='BLOCKED_MISSING_GENERATION_METADATA'
    result['TRPMD_status']='NOT_AVAILABLE_FOR_MATCHED_REDUCED_MODEL; harmonic classical control alone does not instantiate TRPMD'
    result['resource_costs']=limits.costs(); result['elapsed_seconds']=time.monotonic()-started
    json_write(folder/'results.json',result)
    return result


def refine_matched_quantum(output):
    """Refine the frozen quantum thermal trace without altering pilot evidence."""
    source=output/'matched_material'
    folder=source/'quantum_refinement_p24';folder.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(600,4);model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    bath=gamma_material_bath(model,2);frozen=FrozenMaterialCoordinates(bath)
    quantum=float(np.mean(bath.omega_rad_s)*constants.hbar/(1e-3*constants.elementary_charge))
    contract={'p':24,'quadrature_orders':[32,40],'detectors':['all_beams','noncentral'],
        'target_quantum_initial_tail':1e-8,'line_windows':{'zero':[0],'one':[-1,1],'multiphonon':'abs(ell)>=2'},
        'classical_input':'pilot raw independently scrambled batches reused exactly',
        'interpretation_gate':'a method difference is identified only when larger than3*scrambleSE plus retained quantum initial-tail bound; no normality confidence claim'}
    json_write(folder/'contract.json',contract)
    operators=[];records=[]
    for order in [32,40]:
        limits.check();started=time.monotonic();basis,operator=frozen.fock_operator(24,order)
        operators.append(operator);records.append({'quadrature_order':order,'elapsed_seconds':time.monotonic()-started})
        np.savez_compressed(folder/f'operator_order{order}.npz',basis=basis,operator=operator)
    detectors={'all_beams':np.ones(frozen.ne,dtype=bool),'noncentral':np.arange(frozen.ne)!=frozen.ne//2}
    result={'status':'INCOMPLETE','contract':contract,'comparisons':[],'timings':records}
    for temperature in [300,1000]:
        for name,detector in detectors.items():
            ell,quantum_line,tail,_=transition_lines(operators[-1],basis,bath.omega_rad_s,temperature,detector)
            _,previous,_,_=transition_lines(operators[-2],basis,bath.omega_rad_s,temperature,detector)
            with np.load(source/f'T{temperature}_{name}_comparison.npz') as raw:
                classical_ell=raw['classical_ell'];batches=raw['classical_corrected_batches']
            rows=[]
            for label in ['zero','one','multiphonon','all']:
                selected={'zero':ell==0,'one':abs(ell)==1,'multiphonon':abs(ell)>=2,'all':np.ones(len(ell),bool)}[label]
                cselected={'zero':classical_ell==0,'one':abs(classical_ell)==1,'multiphonon':abs(classical_ell)>=2,'all':np.ones(len(classical_ell),bool)}[label]
                quantum_value=float(np.sum(quantum_line[selected]));classical_values=np.sum(batches[:,cselected],axis=1)
                classical_value=float(np.mean(classical_values));se=float(np.std(classical_values,ddof=1)/np.sqrt(len(classical_values)))
                difference=classical_value-quantum_value
                rows.append({'window':label,'quantum_probability':quantum_value,'classical_Kubo_probability':classical_value,
                    'classical_minus_quantum':difference,'classical_independent_scramble_SE':se,
                    'quantum_missing_initial_probability_upper_bound':tail,
                    'difference_exceeds3SE_plus_initial_tail':bool(abs(difference)>3*se+tail)})
            result['comparisons'].append({'temperature_k':temperature,'detector':name,'thermal_initial_tail':tail,
                'quadrature32_to40_line_L1':float(np.sum(abs(quantum_line-previous))),'windows':rows})
            np.savez_compressed(folder/f'T{temperature}_{name}.npz',ell=ell,quantum_lines=quantum_line,
                                classical_ell=classical_ell,classical_corrected_batches=batches)
    result['status']='VERIFIED_FROZEN_REDUCED_MODEL_THERMAL_TRACE' if all(r['thermal_initial_tail']<1e-8 and r['quadrature32_to40_line_L1']<1e-10 for r in result['comparisons']) else 'INCOMPLETE'
    result['full_quantum_channel_scope']='pilot p<=10 remains separately truncated; frozen trace refinement does not establish its finite-temperature convergence'
    result['external_TACAW_status']='BLOCKED_MISSING_GENERATION_METADATA'
    result['resource_costs']=limits.costs();json_write(folder/'results.json',result)
    return result


def separate_frozen_projection(output):
    """Separate finite Fock projection error from phonon channel-energy phases."""
    source=output/'matched_material';folder=source/'frozen_projection_control';folder.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(600,4);model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    bath=gamma_material_bath(model,2);frozen=FrozenMaterialCoordinates(bath);rows=[]
    for temperature,p in [(300,6),(300,8),(1000,8),(1000,10)]:
        limits.check();start=time.monotonic();h=DenseDisplacedMaterial(bath,p=p,n=3,steps=65,phonon_evolution=False,maximum_bytes=3*1024**3)
        initial=np.column_stack([h.initial_state(state) for state in h.basis]);wave=h.propagate(initial=initial).reshape(len(h.basis),frozen.ne,len(h.basis)).transpose(1,0,2)
        for name,detector in {'all_beams':np.ones(frozen.ne,bool),'noncentral':np.arange(frozen.ne)!=frozen.ne//2}.items():
            ell,line,tail,_=transition_lines(wave,h.basis,bath.omega_rad_s,temperature,detector)
            with np.load(source/f'full_T{temperature}_p{p}_{name}.npz') as old:full=old['full_lines'];coordinate=old['frozen_lines']
            rows.append({'temperature_k':temperature,'p':p,'detector':name,'thermal_initial_tail':tail,
                'full_vs_same_projection_frozen_line_L1':float(np.sum(abs(full-line))),
                'projected_frozen_vs_coordinate_frozen_line_L1':float(np.sum(abs(line-coordinate))),
                'scope':'finite declared representation; thermal tail still bounds missing full-channel initial states separately'})
            np.savez_compressed(folder/f'T{temperature}_p{p}_{name}.npz',ell=ell,full_lines=full,frozen_projected_lines=line,frozen_coordinate_lines=coordinate)
        print('frozen projection',temperature,p,time.monotonic()-start,flush=True)
    result={'status':'EXECUTED_SEPARATE_APPROXIMATION_CONTROLS','rows':rows,'resource_costs':limits.costs(),
        'interpretation':'quantum correlation and explicit channel summation of same frozen operator are equivalent; only Hphonon during traversal changes physics here; finite Fock projection adds a separate numerical error'}
    json_write(folder/'results.json',result);return result


def audit_quantum_error_budget(output):
    """Keep initial Gibbs tail, final projection deficit and quadrature apart."""
    folder=output/'matched_material'/'quantum_refinement_p24'
    with np.load(folder/'operator_order40.npz') as data:basis=data['basis'];operator=data['operator']
    with np.load(folder/'operator_order32.npz') as data:previous=data['operator']
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz');bath=gamma_material_bath(model,2)
    refined=json.loads((folder/'results.json').read_text());rows=[]
    for temperature in [300,1000]:
        ell,all_line,tail,prob=transition_lines(operator,basis,bath.omega_rad_s,temperature,np.ones(len(operator),bool))
        final_deficit=max(0.,float(np.sum(prob)-np.sum(all_line)))
        for name,detector in {'all_beams':np.ones(len(operator),bool),'noncentral':np.arange(len(operator))!=len(operator)//2}.items():
            _,line,_,_=transition_lines(operator,basis,bath.omega_rad_s,temperature,detector)
            _,old_line,_,_=transition_lines(previous,basis,bath.omega_rad_s,temperature,detector)
            quadrature_error=float(np.sum(abs(line-old_line)));budget=tail+final_deficit+quadrature_error
            record=next(r for r in refined['comparisons'] if r['temperature_k']==temperature and r['detector']==name)
            rows.append({'temperature_k':temperature,'detector':name,'initial_thermal_tail':tail,
                'represented_final_projection_probability_deficit':final_deficit,'quadrature_line_L1_change':quadrature_error,
                'conservative_sum_of_thermal_tail_and_observed_numerical_errors':budget,
                'windows':[{'window':r['window'],'difference_exceeds3SE_plus_all_quantum_errors':abs(r['classical_minus_quantum'])>3*r['classical_independent_scramble_SE']+budget} for r in record['windows']]})
    result={'status':'EXECUTED_ERROR_BUDGET_AUDIT','rows':rows,
        'scope':'Gibbs tail is analytic; final norm deficit and32-to40quadrature change are numerical diagnostics, not rigorous a priori bounds; no electron orbulk bath convergence claimed'}
    json_write(folder/'error_budget.json',result);return result


def stream_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def run_metadata(output):
    folder=output/'external_metadata';folder.mkdir(parents=True,exist_ok=True)
    roots=[ROOT.parent/name for name in ['HiPhive_TRPMD','Si_LS','STEM_EELS','TACAW','test_TACAW_torched_TACAW']]
    exact_names=['Si_5x7x52_300K_beads9_Tersoff.npy','Si_5x7x52_1000K_beads4_Tersoff.npy']
    candidates=[];matches=[]
    for root in roots:
        if not root.exists():continue
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in ['.py','.ipynb','.json','.sh','.lmp','.in','.log'] or path.stat().st_size>20*1024**2:continue
            content=path.read_text(errors='replace')
            candidates.append({'path':str(path),'sha256':stream_sha256(path),'bytes':path.stat().st_size})
            named=[name for name in exact_names if name in content]
            if named:
                matches.append({'path':str(path),'sha256':stream_sha256(path),'exact_array_names':named,
                    'classification':'analysis-or-report reference unless a trustworthy generation output link is independently found'})
    datasets=[]
    for temperature,name in zip([300,1000],exact_names):
        original=ROOT/f'Si_EELS_{temperature}K_Tersoff.npy'; counterpart=ROOT.parent/'HiPhive_TRPMD'/name
        array=np.load(original,mmap_mode='r')
        hashes=[stream_sha256(path) for path in [original,counterpart]]
        datasets.append({'temperature_label_k':temperature,'path':str(original),'sha256':hashes[0],
            'shape':list(array.shape),'dtype':str(array.dtype),'counterpart':str(counterpart),'counterpart_sha256':hashes[1],
            'exact_bytes_match':hashes[0]==hashes[1],
            'recoverable_generation_metadata':{},'interpretation':'filename labels and downstream defaults do not prove original generation settings'})
    prior=ROOT/'benchmark_results/10_tacaw/tacaw_20261102_v2/metadata_manifest.json'
    result={'status':'BLOCKED_MISSING_GENERATION_METADATA','datasets':datasets,'search_roots':[str(p) for p in roots],
        'candidate_text_file_count':len(candidates),'candidate_text_files':candidates,'exact_name_references':matches,
        'prior_manifest':str(prior),'prior_manifest_sha256':stream_sha256(prior),
        'generation_link_found':False,'recovered_fields':[],
        'missing':['physical reciprocal axes and FFT ordering','original timing, transform, window and energy calibration',
            'verified structure, orientation, thickness and boundary conditions','exact potential and force model checksums',
            'actual temperature, ensemble, thermostat and trajectory/bead preparation','electron grid/probe/potential/slice spacing',
            'elastic/zero-energy subtraction','absolute incident-electron and Fourier normalization','independent statistical blocks','generation hardware/runtime'],
        'Kubo_policy':'user-supplied already applied; no further correction applied to old arrays',
        'scope':'all external files read-only; plausible similarly named structures and notebooks do not calibrate unmatched arrays'}
    confirmation=folder/'user_confirmation_20261102.json'
    if confirmation.exists():result['user_confirmation']=json.loads(confirmation.read_text())
    json_write(folder/'results.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['metadata','matched','refine','projection','error-budget','all'],default='all');parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name,method in [('metadata',run_metadata),('matched',run_matched),('refine',refine_matched_quantum),('projection',separate_frozen_projection),('error-budget',audit_quantum_error_budget)]:
        if args.stage in ['all',name]:
            result=method(args.output);print(name,result['status'],flush=True)


if __name__=='__main__':main()
