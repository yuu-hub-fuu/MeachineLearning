"""Resumable parallel ranges from the original author checkpoint URL."""
import argparse,concurrent.futures,json,time,urllib.request
from prepare_upgrade_weights import DEST,ITEMS,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    item=next(x for x in ITEMS if x['name']==args.name)
    assert item.get('sha256'),'Parallel recovery requires a published expected SHA256'
    target=DEST/item['name']
    if target.exists() and digest(target)==item['sha256']:return
    prefix=target.with_suffix(target.suffix+'.part');start=prefix.stat().st_size if prefix.exists() else 0
    folder=DEST/(item['name']+'.ranges');folder.mkdir(exist_ok=True)
    chunk=(item['bytes']-start+args.workers-1)//args.workers
    def fetch(index):
        begin=start+index*chunk;end=min(item['bytes'],begin+chunk)-1
        path=folder/f'{begin}-{end}.part'
        for attempt in range(6):
            have=path.stat().st_size if path.exists() else 0
            if have==end-begin+1:return path
            request=urllib.request.Request(item['url'],headers={'Range':f'bytes={begin+have}-{end}','User-Agent':'FER-course-project'})
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request,timeout=60) as r:
                    expected=f'bytes {begin+have}-{end}/{item["bytes"]}'
                    if r.status!=206 or r.headers.get('Content-Range')!=expected:
                        raise RuntimeError(f'Range mismatch: {r.status}, {r.headers.get("Content-Range")}')
                    with path.open('ab') as f:
                        while b:=r.read(65536):f.write(b)
            except Exception as error:print(json.dumps({'event':'range_retry','index':index,'attempt':attempt,'error':repr(error)}),flush=True)
            if path.exists() and path.stat().st_size==end-begin+1:
                print(json.dumps({'event':'range_complete','index':index,'bytes':path.stat().st_size}),flush=True);return path
        raise RuntimeError(f'Range {index} did not finish')
    print(json.dumps({'event':'parallel_recovery','prefix_bytes':start,'total':item['bytes'],'workers':args.workers}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:parts=list(pool.map(fetch,range(args.workers)))
    assembled=target.with_suffix(target.suffix+'.assembled')
    with assembled.open('wb') as out:
        for part in ([prefix] if start else [])+parts:
            with part.open('rb') as f:
                while b:=f.read(1<<20):out.write(b)
    assert assembled.stat().st_size==item['bytes']
    assert digest(assembled)==item['sha256'],'Checkpoint SHA256 mismatch'
    assembled.replace(target)
    metadata={**item,'path':str(target),'status':'downloaded_verified_sha256','download_method':'resumed_prefix_parallel_ranges'}
    target.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
