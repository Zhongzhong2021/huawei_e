"""Fetch only pinned pretrained tool files from the recorded model manifest."""
import argparse
from pathlib import Path
import urllib.request
from common import read, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(Path(__file__).resolve().parents[1] / 'data/models.json')
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        target = (cache / item['path']).resolve()
        if not target.is_relative_to(cache):
            raise ValueError('Model path outside cache')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != item['sha256']:
                raise ValueError('Existing model digest mismatch: ' + item['path'])
        else:
            partial = target.with_name(target.name + '.partial')
            with urllib.request.urlopen(item['url'], timeout=120) as response, partial.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if digest(partial) != item['sha256']:
                raise ValueError('Downloaded digest mismatch: ' + item['path'])
            partial.replace(target)
        print('Verified', item['path'], flush=True)
    write(cache / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
