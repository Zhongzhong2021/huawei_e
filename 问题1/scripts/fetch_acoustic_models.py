"""Fetch only fixed pretrained assets; verify every byte against recorded hashes."""
import hashlib
import json
import urllib.request
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    manifest = json.loads((ROOT / 'data/acoustic_comparison/models.json').read_text())
    downloaded=[]
    for item in manifest:
        target = args.cache / Path(item['path']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.download')
            with urllib.request.urlopen(item['url'], timeout=120) as source, temporary.open('wb') as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == item['sha256'], item['tag']
            temporary.replace(target)
        assert target.stat().st_size == item['bytes']
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'], item['tag']
        print(item['tag'], 'verified')
        downloaded.append({**item,'path':str(target.resolve())})
    (args.cache/'manifest.json').write_text(json.dumps(downloaded, indent=2) + '\n')


if __name__ == '__main__':
    main()
