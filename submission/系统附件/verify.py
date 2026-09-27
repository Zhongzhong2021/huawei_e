"""Verify extracted system files; optionally verify downloaded external resources."""
import argparse,hashlib,json
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parent
 records=json.loads((root/'manifest.json').read_text())['files']
 if a.external:records+=json.loads((root/'external_files.json').read_text())['files']
 for r in records:
  path=root/r['path']
  if not path.is_file() or path.stat().st_size!=r['bytes'] or digest(path)!=r['sha256']:raise RuntimeError('Missing or changed: '+str(path))
 print(json.dumps({'verified_files':len(records),'external_verified':a.external}))
if __name__=='__main__':main()
