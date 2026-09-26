"""Verify the checked-in Question 1 snapshot without third-party dependencies."""
from pathlib import Path
import hashlib,json,sys

def main():
    root=Path(__file__).resolve().parents[1]
    entries=json.loads((root/'manifest.sha256.json').read_text(encoding='utf-8'))
    bad=[]
    for name,expected in entries.items():
        path=root/name
        if not path.is_file(): bad.append((name,'missing'));continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:bad.append((name,'changed'))
    print(json.dumps({'files':len(entries),'passed':not bad,'differences':bad},ensure_ascii=False,indent=2))
    return 1 if bad else 0

if __name__=='__main__':sys.exit(main())
