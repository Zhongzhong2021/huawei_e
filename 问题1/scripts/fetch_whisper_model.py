"""Fetch a pinned converted Whisper model; record every artifact digest."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from common import read, write, digest

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = read(ROOT / 'config/whisper_alignment.json')
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in cfg['files']:
        path = Path(hf_hub_download(cfg['repository'], name, revision=cfg['revision'], local_dir=args.output))
        files.append({'path': name, 'sha256': digest(path), 'bytes': path.stat().st_size})
        print(name, path.stat().st_size, flush=True)
    write(args.output / 'manifest.json', {'repository': cfg['repository'], 'revision': cfg['revision'], 'files': files})

if __name__ == '__main__':
    main()
