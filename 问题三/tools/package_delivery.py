"""Create or verify the complete, data-free delivery manifest and archive (stdlib only)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'data', 'output', 'reproduced', 'work', '__pycache__', '.pytest_cache'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files():
    result = []
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & EXCLUDED or path.suffix == '.pyc' or rel.as_posix() == 'manifest.json':
            continue
        if path.is_symlink():
            raise ValueError(f'Delivery cannot depend on symlink: {rel}')
        if path.is_file():
            result.append(path)
    return result


def verify(manifest):
    actual = {p.relative_to(ROOT).as_posix(): p for p in files()}
    expected = {r['path']: r for r in manifest['files']}
    if actual.keys() != expected.keys():
        raise ValueError(f'File set differs: {sorted(actual.keys() ^ expected.keys())}')
    for name, record in expected.items():
        path = actual[name]
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Hash/size mismatch: {name}')
    print(json.dumps({'status': 'verified', 'files': len(expected), 'bytes': manifest['total_bytes']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--archive', type=Path, help='Archive output outside the delivery directory')
    args = parser.parse_args()
    target = ROOT / 'manifest.json'
    if args.verify:
        verify(json.loads(target.read_text()))
        return
    content = files()
    records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in content]
    by_name = {r['path']:r for r in records}
    for name in ('uncertainty', 'selfmm', 'tetfn'):
        selection = json.loads((ROOT/'weights'/name/'selection.json').read_text())
        if by_name[f'weights/{name}/best.pt']['sha256'] != selection['checkpoint_sha256']:
            raise ValueError(f'Expected full original checkpoint (run git lfs pull): {name}')
    manifest = dict(schema_version='q3-delivery-1', main_model='uncertainty', models=['uncertainty','selfmm','tetfn'],
                    excludes=sorted(EXCLUDED), raw_competition_data_included=False, fits_50mb_competition_limit=False,
                    hash_algorithm='SHA256', total_bytes=sum(r['bytes'] for r in records), files=records)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    if args.archive:
        out = args.archive.expanduser().resolve()
        if out.is_relative_to(ROOT):
            raise ValueError('Archive must be outside the packaged directory')
        out.parent.mkdir(parents=True, exist_ok=True)
        def normalized(info):
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            return info
        with out.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as archive:
                for p in sorted([*content,target]):
                    archive.add(p,arcname=str(Path('问题三')/p.relative_to(ROOT)),recursive=False,filter=normalized)
        sha = digest(out)
        out.with_name(out.name+'.sha256').write_text(f'{sha}  {out.name}\n')
        print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':sha},ensure_ascii=False))
    verify(manifest)


if __name__ == '__main__':
    main()
