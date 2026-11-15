"""Download the selected official checkpoint directly to D:, never the user profile cache."""
import hashlib,json,os,urllib.request
from pathlib import Path

ROOT=Path(os.environ.get('FER_ROOT',r'D:\FER2026task2'))
if ROOT.drive.lower()=='c:': raise RuntimeError('Pretrained weights must not be stored on C:')
DEST=ROOT/'pretrained'/'enet_b0_8_best_vgaf.pt'
URL='https://api.github.com/repos/sb-ai-lab/EmotiEffLib/git/blobs/7219cfb037209b253ea8b81cf969e60a00e9181c'
EXPECTED='95aafb39b8bb87964f45e208b9ab31e276e3e5278678db4961d18e6a1b42a141'

def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if not DEST.exists():
        op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req=urllib.request.Request(URL,headers={'Accept':'application/vnd.github.raw+json','User-Agent':'FER-course-project'})
        with op.open(req,timeout=60) as r,DEST.with_suffix('.part').open('wb') as f:
            while block:=r.read(1024*1024):
                f.write(block); print(f'downloaded {f.tell()} bytes',flush=True)
        actual=hashlib.sha256(DEST.with_suffix('.part').read_bytes()).hexdigest()
        if actual!=EXPECTED: raise RuntimeError('Downloaded checksum mismatch; incomplete file retained as .part')
        DEST.with_suffix('.part').replace(DEST)
    actual=hashlib.sha256(DEST.read_bytes()).hexdigest()
    if actual!=EXPECTED: raise RuntimeError('Checkpoint checksum mismatch')
    meta={'url':URL,'path':str(DEST),'bytes':DEST.stat().st_size,'sha256':actual}
    DEST.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)

if __name__=='__main__': main()
