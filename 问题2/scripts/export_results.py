"""Copy new results to Windows without touching earlier rounds."""
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
destination=Path(sys.argv[1]).resolve()
for folder in ['study','audit','final','evaluation','analysis','evidence','figures','scenarios','reports']:
    if not (ROOT/folder).exists():continue
    for source in (ROOT/folder).rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:
            target=destination/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
for source in (ROOT/'runs').rglob('*'):
    if source.is_file() and source.suffix in ['.json','.csv','.log','.txt']:
        target=destination/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
print(destination)
