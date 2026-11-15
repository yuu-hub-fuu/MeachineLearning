"""Download verified author-provided FER checkpoints only into D:.

The Google Drive file IDs come from the authors' READMEs. The RMN SHA256
comes from its official Hugging Face LFS metadata. This does not download
external image labels or reconstruct competition test labels.
"""
import concurrent.futures, hashlib, json, os, time, urllib.request
from pathlib import Path

ROOT=Path(os.environ.get('FER_ROOT',r'D:\FER2026task2'))
if ROOT.drive.lower()=='c:': raise RuntimeError('Weights must not be on C:')
DEST=ROOT/'pretrained';DEST.mkdir(parents=True,exist_ok=True)
ITEMS=[
 {'name':'nla_rafdb_best.pth','url':'https://drive.usercontent.google.com/download?id=12NY75DwMUnXFbRYDQso4eXIxLAV2PP63&export=download&confirm=t',
  'bytes':46894543,'source':'https://github.com/jungyu0413/NLA','pretraining':'MS-Celeb-1M then RAF-DB'},
 {'name':'qcs_ferplus_best.pth','url':'https://drive.usercontent.google.com/download?id=1nepj3oPB670p03ReF9yFGxD9JwK_RGhk&export=download&confirm=t',
  'bytes':331781045,'source':'https://github.com/birdwcp/QCS','pretraining':'MS-Celeb-1M then FERPlus'},
 {'name':'resmasking_fer2013.pth','url':'https://huggingface.co/phamquiluan/ResidualMaskingNetwork/resolve/main/Z_resmasking_dropout1_rot30_2019Nov30_13.32',
  'bytes':551634217,'sha256':'1aa0454662df6c59ab8bc4db80a25be5a109acfda970d32e7593cde2d3aeae68',
  'source':'https://huggingface.co/phamquiluan/ResidualMaskingNetwork','pretraining':'FER2013; potential source-image overlap with this competition, not an independent holdout claim'},
]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def download(item):
    path=DEST/item['name'];partial=path.with_suffix(path.suffix+'.part')
    if not path.exists():
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(5):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==item['bytes']:break
            headers={'User-Agent':'FER-course-project'}
            if offset:headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(item['url'],headers=headers)
            try:
                with opener.open(request,timeout=60) as response:
                    if 'text/html' in response.headers.get('Content-Type',''):raise RuntimeError('Received HTML, not a checkpoint')
                    resume=offset>0 and response.status==206
                    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Range response starts at the wrong offset')
                    count=offset if resume else 0;last=time.time()
                    with partial.open('ab' if resume else 'wb') as f:
                        while True:
                            chunk=response.read(65536)
                            if not chunk:break
                            f.write(chunk);count+=len(chunk)
                            if time.time()-last>20:
                                print(json.dumps({'event':'download_progress','name':item['name'],'bytes':count,'total':item['bytes']}),flush=True);last=time.time()
                if partial.stat().st_size==item['bytes']:break
            except Exception as error:
                print(json.dumps({'event':'download_retry','name':item['name'],'attempt':attempt+1,'error':repr(error)}),flush=True)
        if partial.stat().st_size!=item['bytes']:raise RuntimeError(f'Wrong size for {path.name}')
        partial.replace(path)
    if path.stat().st_size!=item['bytes']:raise RuntimeError(f'Existing file has wrong size: {path}')
    sha=digest(path)
    if item.get('sha256') and item['sha256']!=sha:raise RuntimeError(f'Wrong SHA256: {path}')
    metadata={**item,'path':str(path),'sha256':sha,'status':'downloaded_verified_size'}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)
    return metadata


if __name__=='__main__':
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(download,item):item for item in ITEMS}
        for future in concurrent.futures.as_completed(futures):
            try:results.append(future.result())
            except Exception as e:
                results.append({'name':futures[future]['name'],'status':'failed','error':repr(e)})
                print(json.dumps(results[-1]),flush=True)
    output=Path(__file__).resolve().parent.parent/'upgrade_pretrained_manifest.json'
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
