"""Download a pinned public checkpoint; verify sizes and upstream LFS SHA-256."""
from pathlib import Path
import concurrent.futures,hashlib,json,time
import requests

ROOT=Path(__file__).resolve().parent/'rlad_swinv2_large'
REPO='JonathanFhima/RLAD-SwinV2-Large'
REV='cf1f116d6b9898aa7b2aa3b075e071d3460e3a6e'
META=json.loads((ROOT/'hub_metadata.json').read_text(encoding='utf-8'))
NAMES=['model.safetensors','LUNetv2.py','image_processing_lunetv2.py','config.json','requirements.txt','README.md','infer.py','preprocessor_config.json']

def download(name):
    item=next(x for x in META['siblings'] if x['rfilename']==name)
    p=ROOT/name;url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
    for attempt in range(4):
        try:
            if not p.exists():
                print(f'DOWNLOADING {name} attempt={attempt+1}',flush=True)
                tmp=p.with_suffix(p.suffix+'.part');start=tmp.stat().st_size if tmp.exists() else 0
                headers={'Range':f'bytes={start}-'} if start else {}
                with requests.get(url,stream=True,timeout=(20,60),headers=headers) as r:
                    r.raise_for_status()
                    if start and r.status_code!=206:start=0
                    count=start;stamp=time.monotonic()
                    with tmp.open('ab' if start else 'wb') as f:
                        for b in r.iter_content(1024*1024):
                            f.write(b);count+=len(b)
                            if time.monotonic()-stamp>25:
                                print(f'PROGRESS {name} {count/1e6:.1f}/{item["size"]/1e6:.1f} MB',flush=True)
                                stamp=time.monotonic()
                assert count==item['size'],(count,item['size'])
                tmp.replace(p)
            digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
            assert p.stat().st_size==item['size']
            if 'lfs' in item:assert digest==item['lfs']['sha256']
            print(f'COMPLETE {name} {p.stat().st_size} {digest}',flush=True)
            return dict(file=name,url=url,bytes=p.stat().st_size,sha256=digest)
        except Exception as e:
            print(f'RETRY {name}: {type(e).__name__}: {str(e)[:200]}',flush=True)
    raise RuntimeError('Download failed: '+name)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results=list(ex.map(download,NAMES))
    (ROOT/'download_manifest.json').write_text(json.dumps(dict(repo=REPO,revision=REV,files=results),ensure_ascii=False,indent=2),encoding='utf-8')
