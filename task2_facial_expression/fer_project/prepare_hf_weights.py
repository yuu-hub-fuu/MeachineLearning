"""Download three task-finetuned ViTs from pinned Hugging Face revisions to D:."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
from prepare_upgrade_weights import ROOT,digest

MODELS={'hf_dima':'dima806/facial_emotions_image_detection',
        'hf_mo':'mo-thecreator/vit-Facial-Expression-Recognition',
        'hf_trpakov':'trpakov/vit-face-expression'}


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url,timeout=60) as r:return r.read()
        except Exception:
            if attempt==3:raise
            time.sleep(2)


def download(kind,repo):
    folder=ROOT/'pretrained'/kind;folder.mkdir(parents=True,exist_ok=True)
    api=json.loads(get(f'https://huggingface.co/api/models/{repo}'));revision=api['sha']
    tree=json.loads(get(f'https://huggingface.co/api/models/{repo}/tree/{revision}'))
    item=next(x for x in tree if x['path']=='model.safetensors');size=item['size'];expected=item['lfs']['oid']
    if expected.startswith('sha256:'):expected=expected[7:]
    for name in ['README.md','config.json','preprocessor_config.json']:
        (folder/name).write_bytes(get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}'))
    target=folder/'model.safetensors'
    if not target.exists():
        workers=6;stride=(size+workers-1)//workers
        url=f'https://huggingface.co/{repo}/resolve/{revision}/model.safetensors'
        def segment(index):
            begin=index*stride;end=min(size,(index+1)*stride)-1;part=folder/f'model.range{index}'
            for attempt in range(6):
                offset=part.stat().st_size if part.exists() else 0
                if offset==end-begin+1:return part
                req=urllib.request.Request(url,headers={'Range':f'bytes={begin+offset}-{end}'})
                try:
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=60) as r:
                        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {begin+offset}-{end}/{size}':
                            raise RuntimeError(f'Unexpected range {r.status} {r.headers.get("Content-Range")}')
                        with part.open('ab') as f:
                            while b:=r.read(65536):f.write(b)
                    if part.stat().st_size==end-begin+1:
                        print(json.dumps({'event':'hf_range_complete','kind':kind,'index':index}),flush=True);return part
                except Exception as error:print(json.dumps({'event':'hf_retry','kind':kind,'index':index,'error':repr(error)}),flush=True)
            raise RuntimeError(f'Incomplete {kind} range {index}')
        print(json.dumps({'event':'hf_download_start','kind':kind,'repo':repo,'bytes':size,'revision':revision}),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:parts=list(pool.map(segment,range(workers)))
        temp=folder/'model.safetensors.assembled'
        with temp.open('wb') as out:
            for part in parts:
                with part.open('rb') as f:
                    while b:=f.read(1<<20):out.write(b)
        assert temp.stat().st_size==size and digest(temp)==expected
        temp.replace(target)
    assert target.stat().st_size==size and digest(target)==expected
    report={'kind':kind,'repo':repo,'revision':revision,'bytes':size,'sha256':expected,'path':str(target),'status':'verified',
            'id2label':json.loads((folder/'config.json').read_text())['id2label'],
            'preprocessor':json.loads((folder/'preprocessor_config.json').read_text())}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    out=Path(__file__).resolve().parent.parent; (out/f'{kind}_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(download,k,r):k for k,r in MODELS.items()}
        for f in concurrent.futures.as_completed(pending):
            try:f.result()
            except Exception as e:print(json.dumps({'event':'hf_download_failed','kind':pending[f],'error':repr(e)}),flush=True)
