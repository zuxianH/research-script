#!/usr/bin/env python3
"""Check the separately installed wheel from outside the research checkout."""
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import pkgutil
import sys
import zipfile

out=Path(sys.argv[1]).resolve()
target=out/'provenance/wheel_check_site'
sys.path.insert(0,str(target))
import yoshioka
assert Path(yoshioka.__file__).resolve().is_relative_to(target)

class NoBenchmarks(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='benchmarks' or fullname.startswith('benchmarks.'):
            raise AssertionError('installed core imported benchmark code: '+fullname)

sys.meta_path.insert(0,NoBenchmarks())
modules=[]
for module in pkgutil.iter_modules(yoshioka.__path__,'yoshioka.'):
    if module.name!='yoshioka.__main__':
        importlib.import_module(module.name);modules.append(module.name)
wheel=out/'provenance/wheel/yoshioka_eels-0.2.0-py3-none-any.whl'
root=Path(__file__).resolve().parents[2]
matched=[]
with zipfile.ZipFile(wheel) as archive:
    assert not any(n.startswith('benchmarks/') for n in archive.namelist())
    for name in archive.namelist():
        if name.startswith('yoshioka/') and name.endswith('.py'):
            assert archive.read(name)==(root/name).read_bytes(),name
            matched.append(name)
result={'passed':True,'imported_modules':modules,'module_count':len(modules),
        'source_files_matched':matched,'source_files_matched_count':len(matched),
        'import_path':str(Path(yoshioka.__file__).resolve()),'working_directory':str(Path.cwd()),
        'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'benchmarks_import_guard':True,'environment_modified':False,
        'installation':'pip --no-deps --no-index --target into isolated result-directory target'}
(out/'provenance/installed_wheel_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('passed','module_count','source_files_matched_count','wheel_sha256')}))
