import unittest
import numpy as np
from yoshioka.phonon_pair_importance import BornPairProposal
from yoshioka.stratified_pairs import BranchStratifiedPairs


class StratifiedPairTests(unittest.TestCase):
    def proposal(self):
        rng=np.random.default_rng(4311)
        mass=rng.uniform(.01,1,size=(4**3,6));mass[:,0]*=3;mass/=mass.sum()
        return BornPairProposal(mass,4)

    def test_known_six_dimensional_branch_integral_and_full_support(self):
        proposal=self.proposal();sampler=BranchStratifiedPairs(proposal)
        q,b,w=sampler.nodes(9,86311)
        self.assertTrue(np.all(w>0));self.assertTrue(np.all(np.isfinite(w)))
        # Sum over 36 branch labels, integral of 1 over each BZ is36.
        self.assertLess(abs(np.mean(w.prod(axis=1))-36),.17)
        integrand=(1+q[:,0,0]*q[:,1,2])*(b[:,0]+1)*(b[:,1]+1)
        self.assertLess(abs(np.mean(integrand*w.prod(axis=1))-441),2.5)
        for branch in range(6):
            selected=b[:,0]==branch
            actual=proposal.density(q[selected,0],b[selected,0])
            np.testing.assert_allclose(w[selected,0]*actual,6*sampler.branch_probability[branch],rtol=3e-15)

    def test_all_strata_balanced_and_nested(self):
        sampler=BranchStratifiedPairs(self.proposal());small=sampler.nodes(2,9172);big=sampler.nodes(4,9172)
        for a,b in zip(small,big):np.testing.assert_array_equal(a,b[:len(a)])
        labels=6*big[1][:,0]+big[1][:,1]
        np.testing.assert_array_equal(np.bincount(labels),np.full(36,16))


if __name__=='__main__':unittest.main()
