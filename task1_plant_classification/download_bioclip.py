"""Download the pinned public BioCLIP release with verified HTTP ranges.

Uses official Hugging Face URLs, checks each range length and the official
whole-file SHA-256. Existing complete chunks can be resumed after a failure.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse,parse_qs

SIZE=3944517804
SHA='ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97'
URL='https://huggingface.co/imageomics/bioclip-2.5-vith14/resolve/6e3d04e3d6522012c88181085c5ae666e14c45cd/open_clip_model.safetensors'
ROOT=Path('work/bioclip25');PARTS=ROOT/'parts8';CHUNK=8*1024*1024
CDN_URL=None
header_file=ROOT/'direct_headers.txt'
if header_file.exists():
    for line in header_file.read_text().splitlines():
        if line.lower().startswith('location:'):
            candidate=line.split(':',1)[1].strip();u=urlparse(candidate)
            expiry=int(parse_qs(u.query).get('Expires',['0'])[0])
            if u.hostname=='us.aws.cdn.hf.co' and expiry>time.time()+60:CDN_URL=candidate

def part(i):
    start=i*CHUNK;end=min(SIZE,start+CHUNK)-1
    target=PARTS/f'{i:04d}.bin';expected=end-start+1
    if target.exists() and target.stat().st_size==expected:return i,expected
    tmp=target.with_suffix('.partial')
    prefix=tmp.stat().st_size if tmp.exists() else 0
    if prefix>expected:raise RuntimeError(f'Unexpected partial length for chunk {i}')
    while prefix<expected:
        tail=target.with_suffix('.tail')
        length=min(4*1024*1024,expected-prefix)
        args=['curl.exe','--fail','--location','--silent','--show-error','--connect-timeout','15',
              '--max-time','90','--retry','3','--retry-all-errors','--retry-delay','2',
              '--range',f'{start+prefix}-{start+prefix+length-1}','--output',str(tail),
              '--write-out','%{http_code}',CDN_URL or URL+f'?plant_chunk8={i}&tail={prefix}&length={length}']
        r=subprocess.run(args,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if r.returncode==0 and r.stdout.strip()=='206' and tail.exists() and tail.stat().st_size==length:
            with tmp.open('ab') as f:f.write(tail.read_bytes())
            prefix+=length
        else:raise RuntimeError(f'Chunk {i}: code={r.returncode} HTTP={r.stdout.strip()} retained_prefix={prefix}')
    os.replace(tmp,target)
    return i,expected

def main():
    existing=ROOT/'open_clip_model.safetensors'
    if existing.exists() and existing.stat().st_size==SIZE:
        check=hashlib.sha256()
        with existing.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):check.update(block)
        if check.hexdigest()==SHA:
            print('Reusing SHA-256 verified official BioCLIP weights',flush=True);return
    PARTS.mkdir(parents=True,exist_ok=True);begin=time.time();done=0
    count=(SIZE+CHUNK-1)//CHUNK
    # Reuse the one fully validated 32 MB HTTP range from the first transport.
    seed=ROOT/'parts/0000.bin'
    if seed.exists() and seed.stat().st_size==32*1024*1024:
        with seed.open('rb') as f:
            for i in range(4):
                block=f.read(CHUNK);target=PARTS/f'{i:04d}.bin'
                if not target.exists():target.write_bytes(block)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        failures=[]
        for future in concurrent.futures.as_completed([pool.submit(part,i) for i in range(count)]):
            try:i,n=future.result()
            except Exception as e:
                failures.append(str(e));print(json.dumps({'failed_chunk':str(e)}),flush=True);continue
            done+=n
            if done%(64*1024*1024)==0 or done==SIZE:
                print(json.dumps(dict(chunk=i,complete_mb=round(done/2**20),total_mb=round(SIZE/2**20),seconds=round(time.time()-begin,1))),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} ranges failed; rerun to resume completed ranges')
    h=hashlib.sha256();target=ROOT/'open_clip_model.safetensors';tmp=target.with_suffix('.assembling')
    with tmp.open('wb') as out:
        for i in range(count):
            with (PARTS/f'{i:04d}.bin').open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);out.write(block)
    if h.hexdigest()!=SHA:raise RuntimeError(f'Full SHA-256 mismatch: {h.hexdigest()}')
    assert tmp.stat().st_size==SIZE
    os.replace(tmp,target)
    report=dict(file=str(target),bytes=SIZE,sha256=h.hexdigest(),official_match=True,seconds=time.time()-begin)
    (ROOT/'download_verified.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
