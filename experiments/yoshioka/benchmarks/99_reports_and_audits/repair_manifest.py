"""Streamed content manifest for new repair evidence, with optional verification."""
import argparse,hashlib,json,time
from pathlib import Path
from .workflow_io import ROOT,json_write

OUT=ROOT/'benchmark_results/repair_20261102_v2'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    target=OUT/'manifest.json';started=time.monotonic()
    if args.verify:
        record=json.loads(target.read_text());changed=[]
        for name,value in record['files'].items():
            path=OUT/name
            if not path.exists() or path.stat().st_size!=value['bytes'] or digest(path)!=value['sha256']:changed.append(name)
        print(json.dumps({'checked_files':len(record['files']),'changed_or_missing':changed,'passed':not changed,'seconds':time.monotonic()-started}))
        if changed:raise SystemExit(1)
        return
    files={}
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or path==target or path.name.endswith('.tmp') or '__pycache__' in path.parts or 'reused_evidence' in path.parts:continue
        files[str(path.relative_to(OUT))]={'sha256':digest(path),'bytes':path.stat().st_size,**({'symlink_target':str(path.resolve())} if path.is_symlink() else {})}
    record={'files':files,'count':len(files),'total_bytes':sum(v['bytes'] for v in files.values()),'seconds':time.monotonic()-started,
        'scope':'all files below this new result directory except the manifest itself, temporary files and linked reference evidence; symlink paths can duplicate logical bytes; source paths/hashes are in changed_source_files.json',
        'partial_calculations':'B7 independent joint is a committed512q prefix, not a full physical result; per-case checkpoint statuses define valid prefixes',
        'history':'protected_before/after.json independently verify all1619pre-existing files',
        'resume_note':'a deliberate future resume updates checkpoints and requires a new manifest generation; preserve this finished report before extending it'}
    json_write(target,record);print(json.dumps({k:v for k,v in record.items() if k!='files'}))


if __name__=='__main__':main()
