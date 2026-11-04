"""Scientific input changes must not silently reuse fields or checkpoints."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.momentum_material_action import MomentumMaterialAction
from yoshioka.active_modes.normal_order import NormalOrderedMomentumAction
from yoshioka.checkpointed_channel_action import checkpointed_action

class CacheInputIdentityTests(unittest.TestCase):
 def test_changed_potential_source_gets_fresh_cache(self):
  bath=SimpleNamespace(displacement_m=np.array([[[1e-12],[2e-12],[1e-12]]]),omega_rad_s=np.array([8e13]),inactive_covariance_m2=np.array([np.eye(3)*1e-22]),cell_m=np.diag([5.,4.,6.])*1e-10,positions_m=np.array([[.7,1.1,2.]])*1e-10,labels=['synthetic'])
  for action in (MomentumMaterialAction,NormalOrderedMomentumAction):
   with self.subTest(action=action.__name__),tempfile.TemporaryDirectory() as folder:
    table=Path(folder)/'potential.bin';table.write_text('synthetic potential source A')
    potential=LobatoSilicon(np.array([1.]),np.array([2.]),str(table))
    with patch.object(LobatoSilicon,'from_abtem_table',return_value=potential):
     first=action(bath,[[0,0,0]],1,3,3,5,folder,repetitions=(1,1),minimum_free_bytes=0)
     table.write_text('synthetic potential source B')
     second=action(bath,[[0,0,0]],1,3,3,5,folder,repetitions=(1,1),minimum_free_bytes=0)
    self.assertNotEqual(first.field_digest,second.field_digest)
    # Only the source identity changed in this synthetic case; physics matches.
    np.testing.assert_array_equal(first.profile,second.profile)
 def test_changed_operator_field_rejects_checkpoint(self):
  class H:
   shape=(1,1,1);n=1;steps=3
   def __init__(self,digest,phase):self.field_digest=digest;self.phase=phase
   def exponential_action(self,wave,index):return wave*np.exp(-1j*self.phase)
  class Limits:
   def __init__(self,count=99):self.count=count
   def check(self):
    self.count-=1
    if self.count<0:raise TimeoutError('intentional bounded interruption')
  with tempfile.TemporaryDirectory() as folder:
   wave,state=checkpointed_action(H('potential-A',.1),folder,{'same_caller_metadata':True},Limits(2),checkpoint_slices=1)
   self.assertEqual(state['completed_slices'],2)
   with self.assertRaises(ValueError):checkpointed_action(H('potential-B',.2),folder,{'same_caller_metadata':True},Limits(),checkpoint_slices=1)
if __name__=='__main__':unittest.main()
