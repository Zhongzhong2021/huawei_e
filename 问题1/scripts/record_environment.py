"""Record runtime tool versions without hostnames, usernames or environment secrets."""
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from common import write, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    packages = ['numpy', 'scipy', 'torch', 'torchaudio', 'torchvision', 'transformers',
                'mediapipe', 'Pillow', 'matplotlib', 'openpyxl', 'num2words']
    tools = {}
    for name in ['ffmpeg', 'ffprobe']:
        result = subprocess.run([name, '-version'], capture_output=True, text=True, check=True)
        tools[name] = result.stdout.splitlines()[0]
    write(ROOT / 'data/runtime_environment.json', {
        'python': sys.version, 'system': platform.system(), 'machine': platform.machine(),
        'tools': tools, 'packages': {p: importlib.metadata.version(p) for p in packages},
        'configuration': 'CPU inference; 2 torch threads and seed 0 set explicitly by core generation scripts.',
        'host_or_user_identifiers_recorded': False,
        'environment_lock_sha256': digest(ROOT / 'config/environment.txt'),
        'scope': 'Version capture in the installed environment; not proof of fresh dependency installation.'})
    print(tools)


if __name__ == '__main__':
    main()
