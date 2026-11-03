"""Exercise the real Si diagnostic cache, including a JSON round trip."""
import argparse,contextlib,hashlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from benchmarks.pair_electron_diagnostics import run_case
from benchmarks.approximate_eels import PRIMITIVE,ORIENTED


@unittest.skipUnless(PRIMITIVE.exists() and ORIENTED.exists(),'local Si force-constant inputs required')
class PairDiagnosticResumeTests(unittest.TestCase):
    def test_real_cache_resumes_and_rejects_changed_config_or_array(self):
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=base/'node.npz'
            np.savez(source,q_fractional=np.array([[.173,.117,.081],[-.137,.211,.193]]),
                     branches=np.array([0,5]),inverse_marginals=np.array([6.,6.]))
            panel=base/'panel.json';panel.write_text(json.dumps({'pairs':[{'source_path':str(source),'source_sha256':digest(source)}]}))
            args=argparse.Namespace(output=base/'cases',case='round_trip',panel=panel,
                retained=3,product=5,steps=17,substeps=1,seconds=60,max_new=1,
                born=False,legacy=False,circular=False,field_halfwidth=None,
                planar=False,cell_average=False,background_only=True)
            with contextlib.redirect_stdout(io.StringIO()):run_case(args)
            folder=args.output/args.case;raw=folder/'pair000.npz';num=folder/'numerical_settings.json'
            initial_raw=digest(raw);initial_num=digest(num)
            args.max_new=0
            with contextlib.redirect_stdout(io.StringIO()):run_case(args)
            self.assertEqual(digest(raw),initial_raw)
            self.assertEqual(digest(num),initial_num)
            config=folder/'config.json';original=config.read_bytes()
            changed=json.loads(original);changed['temperature_k']=1000
            config.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'immutable configuration mismatch'):run_case(args)
            config.write_bytes(original)
            raw.write_bytes(raw.read_bytes()+b'guard corruption test')
            with self.assertRaisesRegex(ValueError,'changed completed output'):run_case(args)


if __name__=='__main__':unittest.main()
