import os
import sys,json,io,urllib.request,time,hashlib,gc,os
from pathlib import Path
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)))
sys.path.insert(0,str(R/'work/q1_v8/python_libs'))
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor
tree=json.loads((W/'mirror_tree.json').read_text())
revision='5f8d513c34278006d27f98e5609564e6d4b353d7'
name=sys.argv[1] if len(sys.argv)>1 else 'CMU_MOSEI_COVAREP.csd'
meta=next(x for x in tree if x['path'].endswith('/'+name));size=meta['size']
url=f'https://huggingface.co/datasets/reeha-parkar/cmu-mosei-comp-seq/resolve/{revision}/{meta["path"]}'
class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.n=1024*1024;self.cache=W/'range_cache'/name;self.cache.mkdir(parents=True,exist_ok=True);self.log=[];self.memory={}
        for attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url+'?download=true',headers={'Range':'bytes=0-0'}),timeout=30) as r:self.direct=r.url
                break
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else size+offset;return self.pos
    def read(self,n=-1):
        if n<0:n=size-self.pos
        end=min(size,self.pos+n);parts=[]
        while self.pos<end:
            block=self.pos//self.n;a=block*self.n;b=min(size,a+self.n)-1;p=self.cache/str(block)
            self.getblock(block)
            data=self.memory[block];take=min(end-self.pos,b-self.pos+1);parts.append(data[self.pos-a:self.pos-a+take]);self.pos+=take
        return b''.join(parts)
    def getblock(self,block):
            if block in self.memory:return
            a=block*self.n;b=min(size,a+self.n)-1;p=self.cache/str(block)
            if not p.exists() or p.stat().st_size!=b-a+1:
                for attempt in range(4):
                    try:
                        req=urllib.request.Request(self.direct,headers={'Range':f'bytes={a}-{b}'})
                        with urllib.request.urlopen(req,timeout=30) as r:
                            assert r.status==206,(r.status,dict(r.headers));assert r.headers['Content-Range']==f'bytes {a}-{b}/{size}',r.headers
                            data=r.read();assert len(data)==b-a+1
                        temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_bytes(data);temp.replace(p);break
                    except Exception as exc:
                        print('retry',block,attempt,type(exc).__name__,str(exc)[:80],flush=True)
                        if attempt==3:raise
                        time.sleep(2)
                self.log.append(dict(start=a,end=b,sha256=hashlib.sha256(data).hexdigest()))
                if len(self.log)%10==0:print('downloaded blocks',len(self.log),flush=True)
            self.memory[block]=p.read_bytes()
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
f=RangeFile();out=W/'csd_subset'/Path(name).stem;out.mkdir(parents=True,exist_ok=True)
def prefetch(ds,cols):
    blocks=set()
    if ds.chunks:
        chunk_cols=sorted({c//ds.chunks[1]*ds.chunks[1] for c in cols})
        spans=[]
        for i in range(0,ds.shape[0],ds.chunks[0]):
            for j in chunk_cols:
                info=ds.id.get_chunk_info_by_coord((i,j))
                if info.byte_offset is not None:spans.append((info.byte_offset,info.size))
    else:spans=[(ds.id.get_offset(),ds.id.get_storage_size())]
    for a,n in spans:
        if a is not None and n:blocks.update(range(a//f.n,(a+n-1)//f.n+1))
    missing=[i for i in sorted(blocks) if i not in f.memory]
    # All HDF5 metadata queries above finish before workers start. Avoid h5py
    # callbacks from workers/GC while the file-object global lock is held.
    gc.disable()
    try:
        with ThreadPoolExecutor(6) as pool:list(pool.map(f.getblock,missing))
    finally:gc.enable()
vids=sorted({json.loads(p.read_text(encoding='utf-8'))['sample_id'].split('$_$')[0] for p in (Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')))).glob('*.json')})
if len(sys.argv)>3:vids=vids[int(sys.argv[2])::int(sys.argv[3])]
with h5py.File(f,'r') as h:
    print('roots',list(h),flush=True);g=h[next(iter(h))]
    metadata={k:str(g['metadata'][k][()]) for k in g['metadata']};print('metadata loaded',flush=True)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for vid in vids:
        p=out/(vid+'.npz')
        if p.exists():continue
        if vid not in g['data']:print('missing',vid,flush=True);continue
        d=g['data'][vid];cols=[0,1,11] if 'COVAREP' in name else [0,1,2,3]
        print('reading',vid,d['features'].shape,d['features'].chunks,flush=True)
        prefetch(d['features'],cols);prefetch(d['intervals'],[0,1])
        x=d['features'][:,cols];t=d['intervals'][:];np.savez_compressed(p,features=x,intervals=t,column_indices=np.array(cols))
        print(vid,x.shape,flush=True)
(out/'source.json').write_text(json.dumps(dict(url=url,revision=revision,full_size=size,expected_full_sha256=meta['lfs']['oid'],full_file_hash_verified=False,method='HTTP 206 byte ranges with strict Content-Range; only matching source videos extracted',downloaded_new_blocks=f.log,subset_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')}),indent=2))
