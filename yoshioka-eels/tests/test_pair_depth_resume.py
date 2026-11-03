"""A resumed fixed-field hierarchy must preserve contact application and phase."""
import unittest
import numpy as np
from scipy import constants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.pair_depth_controls import contact_aligned_mesh,FrozenPlanarTable,propagate_table_pair


class PairDepthResumeTests(unittest.TestCase):
    def test_checkpoint_after_contact_reproduces_uninterrupted_hierarchy(self):
        grid=TransverseGrid(8e-10,9e-10,3,3);product=TransverseGrid(8e-10,9e-10,5,5)
        length=2e-9;plane=.731e-9;nodes=contact_aligned_mesh(length,257,[plane])
        rng=np.random.default_rng(731128);unit=constants.elementary_charge
        block=(rng.normal(size=(7,5,5))+1j*rng.normal(size=(7,5,5)))*unit
        block[0]=block[0].real;values=np.repeat(block[None],len(nodes)-1,axis=0)
        contacts=np.zeros((1,4,5,5),complex);contacts[0,:,0,0]=unit*1e-12*np.arange(1,5)
        table=FrozenPlanarTable(nodes,values,[plane],contacts)
        args=(grid,product,np.array([4e13,9e13]),np.array([[3e9,-4e9],[-2e9,1e9]]),table,nodes)
        uninterrupted=propagate_table_pair(*args,local_substeps=2)
        saved={}
        class DeliberateStop(Exception):pass
        def stop(index,coefficients):
            if index==128:
                saved.update(index=index,coefficients=coefficients.copy())
                raise DeliberateStop()
        with self.assertRaises(DeliberateStop):propagate_table_pair(*args,local_substeps=2,checkpoint=stop)
        self.assertGreater(nodes[saved['index']],plane)
        resumed=propagate_table_pair(*args,local_substeps=2,start_interval=saved['index'],initial_coefficients=saved['coefficients'])
        for key in ('word_coefficients','direct_amplitudes','sequential_amplitudes','total_amplitudes'):
            np.testing.assert_array_equal(resumed[key],uninterrupted[key])
        self.assertEqual(uninterrupted['contact_planes_applied'],1)
        self.assertEqual(resumed['contact_planes_applied'],1)


if __name__=='__main__':unittest.main()
