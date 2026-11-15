"""Range-download pinned RLAD weights, then verify upstream SHA-256."""
from pathlib import Path
import concurrent.futures,hashlib,json,time
import requests

ROOT=Path(__file__).resolve().parent/'rlad_swinv2_large'
URL='https://huggingface.co/JonathanFhima/RLAD-SwinV2-Large/resolve/cf1f116d6b9898aa7b2aa3b075e071d3460e3a6e/model.safetensors'
SIZE=806056424
SHA='904c8c20ba38ff23f4350454349ce92e1cb7bbbcefe214243d3f458ecc8f3338'
PARTS=16
CHUNK=(SIZE+PARTS-1)//PARTS
DEST=ROOT/'model.safetensors'
CHUNKS=ROOT/'download_chunks'
CHUNKS.mkdir(exist_ok=True)

def get(i):
    start=i*CHUNK;end=min(SIZE,(i+1)*CHUNK)-1;p=CHUNKS/f'{i:02d}.bin'
    for attempt in range(5):
        try:
            done=p.stat().st_size if p.exists() else 0
            if done==end-start+1:return i
            pos=start+done
            with requests.get(URL,headers={'Range':f'bytes={pos}-{end}'},stream=True,timeout=(20,45)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {pos}-{end}/{SIZE}',r.headers.get('Content-Range')
                with p.open('ab') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
            assert p.stat().st_size==end-start+1
            print(f'CHUNK_COMPLETE {i+1}/{PARTS}',flush=True)
            return i
        except Exception as e:print(f'CHUNK_RETRY {i} {type(e).__name__} {str(e)[:100]}',flush=True)
    raise RuntimeError(f'Failed chunk {i}')

if __name__=='__main__':
    if not DEST.exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARTS) as ex:list(ex.map(get,range(PARTS)))
        tmp=ROOT/'assembled.safetensors.part'
        with tmp.open('wb') as f:
            for i in range(PARTS):
                with (CHUNKS/f'{i:02d}.bin').open('rb') as part:
                    while b:=part.read(4*1024*1024):f.write(b)
        assert tmp.stat().st_size==SIZE
        assert hashlib.file_digest(tmp.open('rb'),'sha256').hexdigest()==SHA
        tmp.replace(DEST)
    assert hashlib.file_digest(DEST.open('rb'),'sha256').hexdigest()==SHA
    files=[]
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix not in ('.part',) and p.name!='download_manifest.json':
            files.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
    result=dict(repo='JonathanFhima/RLAD-SwinV2-Large',revision='cf1f116d6b9898aa7b2aa3b075e071d3460e3a6e',weights_sha256_verified=True,files=files)
    (ROOT/'download_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RLAD_WEIGHTS_VERIFIED '+SHA,flush=True)
