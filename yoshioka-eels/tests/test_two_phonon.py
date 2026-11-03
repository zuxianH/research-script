import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import constants

from yoshioka.free_propagation import TransverseGrid
from yoshioka.two_phonon import pair_nodes, pair_trace, SIGNS
from yoshioka.thermal_dyson_words import words_through_two, thermal_groups, label, net_occupation
from benchmarks.two_phonon_eels import run, require_contract, DEFAULT, PREVIOUS
from benchmarks.two_phonon_validation import dense_two_mode_control
from benchmarks.workflow_io import environment
from benchmarks.two_phonon_analysis import unbiased_return_square, estimate
from yoshioka.phonon_pair_importance import BornPairProposal


class TwoPhononTests(unittest.TestCase):
    def fixture(self):
        grid=TransverseGrid(1e-9,1.3e-9,3,3)
        words=words_through_two(2)
        omega=np.array([31.,67.])*(1e-3*constants.elementary_charge)/constants.hbar
        shift=np.array([[.2,.3],[-.4,.1]])*1e9
        shifts=np.array([sum((shift[k%2]*(1 if k<2 else -1) for k in w),start=np.zeros(2)) for w in words])
        rng=np.random.default_rng(17)
        amplitudes=(rng.normal(size=(len(words),3,3))+1j*rng.normal(size=(len(words),3,3)))*.03
        return grid,words,omega,shifts,amplitudes

    def test_independent_two_mode_dense_nonzero_Bloch_reference(self):
        self.assertTrue(dense_two_mode_control()['passed'])

    def test_nested_nodes_and_full_branch_support(self):
        a=pair_nodes(4,817); b=pair_nodes(8,817)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y[:16])
        self.assertEqual(set(b[1].ravel()),set(range(6)))
        self.assertTrue(np.all((b[0]>=-.5)&(b[0]<.5)))

    def test_importance_probability_and_branch_normalization(self):
        q,branch,w=pair_nodes(15,5791)
        self.assertLess(abs(np.mean(w[:,0])/6-1),.005)
        self.assertLess(abs(np.mean(np.prod(w,axis=1))/36-1),.005)
        # Integrate a known continuous BZ function and all36 discrete branches.
        value=(q[:,0,0]**2+q[:,1,2]**2)*np.prod(w,axis=1)
        self.assertLess(abs(np.mean(value)-6),.04)

    def test_exchange_coherence_and_unordered_pair_factor(self):
        grid,words,omega,shifts,a=self.fixture(); n=np.array([.2,.3])
        actual=pair_trace(words,a,shifts,omega,n,grid,[1,1])
        lines=thermal_groups(words,a,2,n,constants.hbar*omega)
        for j,sign in enumerate(SIGNS):
            ref=next(row for row in lines if row['net']==sign)
            self.assertAlmostEqual(actual['all_beam_weights'][j],.5*np.sum(ref['weight']),places=15)
        self.assertGreater(np.max(abs(actual['exchange_interference'])),0)

    def test_explicit_thermal_trace_including_identical_modes(self):
        grid,words,omega,shifts,a=self.fixture(); n=np.array([.2,.3])
        reference={}
        for i in range(32):
            for j in range(32):
                thermal=(n[0]/(1+n[0]))**i/(1+n[0])*(n[1]/(1+n[1]))**j/(1+n[1])
                by_final={}
                for word,amplitude in zip(words,a):
                    final=[i,j]; factor=1.
                    for k in reversed(word):
                        m,s=label(k,2)
                        if s==1: factor*=np.sqrt(final[m]+1); final[m]+=1
                        elif final[m]>0: factor*=np.sqrt(final[m]); final[m]-=1
                        else: factor=0.; break
                    key=tuple(np.array(final)-[i,j])
                    by_final[key]=by_final.get(key,0)+factor*amplitude
                for key,amplitude in by_final.items():
                    reference[key]=reference.get(key,0)+thermal*abs(amplitude.ravel())**2
        for line in thermal_groups(words,a,2,n,constants.hbar*omega):
            np.testing.assert_allclose(line['weight'],reference[line['net']],rtol=3e-13,atol=1e-16)

    def test_g_four_scaling_and_signed_return_scaling(self):
        grid,words,omega,shifts,a=self.fixture(); n=np.array([.2,.3])
        first=pair_trace(words,a,shifts,omega,n,grid,[1.2,3.4])
        g=.3
        scaled=a*np.array([g**len(w) for w in words])[:,None,None]
        second=pair_trace(words,scaled,shifts,omega,n,grid,[1.2,3.4])
        np.testing.assert_allclose(second['weights'],g**4*first['weights'],rtol=1e-13,atol=1e-25)
        np.testing.assert_allclose(second['elastic_interference'],g**2*first['elastic_interference'],rtol=1e-13,atol=1e-25)

    def test_sampling_weights_enter_probability_once(self):
        grid,words,omega,shifts,a=self.fixture(); n=np.array([.2,.3])
        x=pair_trace(words,a,shifts,omega,n,grid,[1,1])
        y=pair_trace(words,a,shifts,omega,n,grid,[2,3])
        np.testing.assert_allclose(y['weights'],6*x['weights'],rtol=1e-14)

    def test_cache_rejects_covariance_and_propagator_changes(self):
        old={'covariance':[1,2,3],'shifts':[.1,.2],'code':'abc'}
        for name in old:
            new=copy.deepcopy(old); new[name]='changed'
            with self.assertRaises(ValueError): require_contract({'contract':old},new)

    def test_return_square_estimator_removes_noisy_amplitude_bias(self):
        from itertools import product
        mu=np.array([.2+.1j,-.3+.7j]);noise=np.array([.5-.2j,.1+.2j])
        unbiased=[];naive=[]
        for signs in product((-1,1),repeat=3):
            batches=mu[None]+np.array(signs)[:,None]*noise[None]
            unbiased.append(unbiased_return_square(batches))
            naive.append(abs(np.mean(batches,axis=0))**2)
        np.testing.assert_allclose(np.mean(unbiased,axis=0),abs(mu)**2,rtol=1e-14)
        self.assertTrue(np.all(np.mean(naive,axis=0)>abs(mu)**2))
        with self.assertRaises(ValueError):unbiased_return_square(mu[None])

    def test_uncertainty_uses_independent_batch_integrals(self):
        batches=np.array([1.,2.,3.,4.])
        mean,sem,half=estimate(batches)
        self.assertEqual(mean,2.5)
        self.assertAlmostEqual(sem,np.std(batches,ddof=1)/2)
        self.assertGreater(half,1.96*sem)

    def test_Born_joint_proposal_is_unbiased_with_full_support(self):
        rng=np.random.default_rng(572)
        mass=rng.uniform(.01,1,size=(4**3,6));mass/=mass.sum()
        proposal=BornPairProposal(mass,4)
        q,branch,w=proposal.nodes(16,81392)
        np.testing.assert_allclose(proposal.density(q[:,0],branch[:,0])*w[:,0],1,rtol=1e-14)
        self.assertLess(abs(np.mean(w[:,0])-6),.02)
        self.assertLess(abs(np.mean(np.prod(w,axis=1))-36),.15)
        # Known smooth6D scattering-like intensity, integrated with actual PDF.
        intensity=(1+q[:,0,0]*q[:,1,2])*(branch[:,0]+1)*(branch[:,1]+1)
        self.assertLess(abs(np.mean(intensity*np.prod(w,axis=1))-441),2.)
        short=proposal.nodes(5,81392)
        for a,b in zip(short,(q,branch,w)):np.testing.assert_array_equal(a,b[:32])
        self.assertTrue(np.all(w<=30+1e-12))

    def test_finite_bath_pair_enumeration_sampling_and_identical_mode_terms(self):
        modes=3;words=words_through_two(modes);lookup={w:i for i,w in enumerate(words)}
        rng=np.random.default_rng(1912);n=np.array([.2,.5,.8])
        a=(rng.normal(size=len(words))+1j*rng.normal(size=len(words)))*.01
        a[0]=1
        lines=thermal_groups(words,a[:,None,None],modes,n,np.arange(1,4))
        return_mean=sum((n[m]+1)*a[lookup[(m+modes,m)]]+n[m]*a[lookup[(m,m+modes)]] for m in range(modes))
        expected=np.zeros(3)
        for line in lines:
            net=np.array(line['net']);value=line['weight'][0]
            if net.sum()==2:expected[0]+=value
            elif net.sum()==-2:expected[1]+=value
            elif np.sum(abs(net))==2 and net.sum()==0:expected[2]+=value
            elif not net.any():expected[2]+=value-abs(a[0]+return_mean)**2
        # Mixed diagonal terms are the finite-bath connected return variance,
        # not an extra orthogonal final state. They vanish in the continuum
        # double integral; the coherent single-integral return does not vanish.
        table=np.zeros((modes,modes,3))
        for m in range(modes):
            for k in range(modes):
                for sa,sb in SIGNS:
                    i=m if sa==1 else m+modes;j=k if sb==1 else k+modes
                    amplitude=a[lookup[(i,j)]]+a[lookup[(j,i)]]
                    value=.5*(n[m]+(sa==1))*(n[k]+(sb==1))*abs(amplitude)**2
                    table[m,k,0 if sa==sb==1 else 1 if sa==sb==-1 else 2]+=value
        np.testing.assert_allclose(table.sum(axis=(0,1)),expected,rtol=2e-11,atol=1e-15)
        probability=np.array([.2,.3,.5]);count=65536
        first=rng.choice(3,count,p=probability);second=rng.choice(3,count,p=probability)
        values=table[first,second]/(probability[first]*probability[second])[:,None]
        error=abs(values.mean(axis=0)-expected)
        self.assertTrue(np.all(error<5*np.std(values,axis=0,ddof=1)/np.sqrt(count)))

    def test_wider_energy_display_preserves_historical_TACAW_conditioning(self):
        from benchmarks.two_phonon_report import conditional_tacaw
        from benchmarks.two_phonon_analysis import AXIS
        from benchmarks.two_phonon_eels import CONTROLLED
        previous=json.loads((CONTROLLED/'comparison.json').read_text())
        expected=next(c for c in previous['curves'] if c['id']=='fixed_r4_T300_DW')['detectors'][0]['conditional_TACAW_shape_L1']
        with tempfile.TemporaryDirectory() as directory:
            result=conditional_tacaw(Path(directory),np.zeros((len(AXIS),4)))
        self.assertAlmostEqual(result['one_phonon_shape_L1'],expected,places=13)
        self.assertAlmostEqual(result['limited_order_shape_L1'],expected,places=13)

    def test_competing_peak_diagnostic_preserves_paired_spectrum_arrays(self):
        from benchmarks.two_phonon_report import numerical
        from benchmarks.two_phonon_analysis import AXIS
        gauss=lambda e:np.exp(-.5*((AXIS-e)/.8493)**2)
        base=gauss(20)+gauss(70)+.9*gauss(105)+.899*gauss(108.25)
        production=np.tile(base[:,None],(1,4))
        coarse=np.zeros((8,len(AXIS),4,4));coarse[:,:,0]=production[None]
        fine=coarse.copy();fine[:,:,0,1]+=.01*gauss(108.25)[None]
        def fake_load(output,job,n):
            return dict(tag=job,nodes=np.zeros((8,n,2,3)),branches=np.zeros((8,n,2)),contract={'settings':{}})
        with tempfile.TemporaryDirectory() as directory,patch('benchmarks.two_phonon_report.counts',return_value=[16]*8),\
             patch('benchmarks.two_phonon_report.load_job',side_effect=fake_load),\
             patch('benchmarks.two_phonon_report.spectra',side_effect=lambda d:coarse if d['tag']=='main' else fine),\
             patch('benchmarks.two_phonon_report.raw_batches',return_value={'all_loss':np.ones((8,4))}):
            rows=numerical(Path(directory),production)
        self.assertEqual(len(rows),4)
        for row in rows:
            self.assertEqual(len(row['paired_subset_comparison']),4)
            self.assertTrue(row['competing_peak_diagnosis'])
            self.assertTrue(row['primary_targets_met'])
            self.assertGreater(row['competing_peak_diagnosis'][0]['dominant_shift_mev'],1)

    def test_streamed_resume_matches_cold_prefix_and_finished_resume_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs=[Path(directory)/'resume',Path(directory)/'cold']
            config=dict(selected_repeats=4,padding_angstrom=12,selected_covariance_mesh=32,
                temperature_k=300,energy_ev=60000,runtime_limit_seconds=600,memory_limit_gib=2,seed_start=729001)
            for output in outputs:
                output.mkdir(); (output/'config.json').write_text(json.dumps(config))
                (output/'environment.json').write_text(json.dumps(environment()))
                (output/'covariance').mkdir()
                shutil.copy2(PREVIOUS/'covariance/T300_mesh32.json',output/'covariance/T300_mesh32.json')
            with contextlib.redirect_stdout(io.StringIO()):
                run(outputs[0],power=2,batches=2,grid=5,steps=17,max_new=3)
                run(outputs[0],power=2,batches=2,grid=5,steps=17)
                run(outputs[1],power=2,batches=2,grid=5,steps=17)
                before={str(p.relative_to(outputs[0])):p.stat().st_mtime_ns for p in outputs[0].rglob('*') if p.is_file()}
                run(outputs[0],power=2,batches=2,grid=5,steps=17)
                after={str(p.relative_to(outputs[0])):p.stat().st_mtime_ns for p in outputs[0].rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            for path in (outputs[0]/'pairs/main').rglob('*.npz'):
                with np.load(path) as a,np.load(outputs[1]/path.relative_to(outputs[0])) as b:
                    for key in a.files:
                        if key not in ('field_seconds','propagation_seconds'):
                            np.testing.assert_array_equal(a[key],b[key])
            # A real physical-input change must reject the completed cache too.
            path=outputs[0]/'covariance/T300_mesh32.json'
            original=path.read_text();row=json.loads(original);row['covariance_m2'][0][0]*=1.01
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):run(outputs[0],power=2,batches=2,grid=5,steps=17)
            path.write_text(original)


if __name__=='__main__': unittest.main()
