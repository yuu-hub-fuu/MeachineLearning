"""Official Xet download, with curl for Hub metadata on this Windows proxy.

Only public repository read metadata is requested; transient access tokens stay
in process memory and are never printed or written by this helper.
"""
import os
from pathlib import Path
import subprocess
import hashlib
import json
import time

os.environ['HF_HOME']=str(Path('work/hf_bioclip').resolve())
os.environ['HF_XET_CACHE']=str(Path('work/hf_bioclip/xet').resolve())
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
from hf_xet import PyXetDownloadInfo,download_files

ROOT=Path('work/bioclip25')
REV='6e3d04e3d6522012c88181085c5ae666e14c45cd'
BASE='https://huggingface.co/imageomics/bioclip-2.5-vith14/resolve/'+REV+'/open_clip_model.safetensors'

def headers(url,head=False):
    args=['curl.exe','--fail','--silent','--show-error','--connect-timeout','15','--max-time','45','--dump-header','-','--output','NUL']
    if head:args.append('--head')
    r=subprocess.run(args+[url],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if r.returncode:raise RuntimeError(f'Public metadata curl failed: {r.returncode}')
    return {line.split(':',1)[0].lower():line.split(':',1)[1].strip() for line in r.stdout.splitlines() if ':' in line}

def connection():
    h=headers('https://huggingface.co/api/models/imageomics/bioclip-2.5-vith14/xet-read-token/'+REV)
    return h['x-xet-cas-url'],h['x-xet-access-token'],int(h['x-xet-token-expiration'])

if __name__=='__main__':
    h=headers(BASE,head=True);file_hash=h['x-xet-hash'];endpoint,token,expiration=connection()
    target=ROOT/'open_clip_model.safetensors';temporary=target.with_suffix('.xet-incomplete')
    begin=time.time();state={'bytes':0,'last':0}
    def refresh():
        _,token,expiration=connection();return token,expiration
    def update(delta):
        state['bytes']+=delta
        if time.time()-state['last']>20:
            print(json.dumps({'xet_progress_mb':round(state['bytes']/2**20),'seconds':round(time.time()-begin)}),flush=True)
            state['last']=time.time()
    print('Official Xet metadata ready; starting transfer',flush=True)
    download_files([PyXetDownloadInfo(destination_path=str(temporary.resolve()),hash=file_hash,file_size=3944517804)],
                   endpoint=endpoint,token_info=(token,expiration),token_refresher=refresh,progress_updater=[update])
    checksum=hashlib.sha256()
    with temporary.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):checksum.update(block)
    assert checksum.hexdigest()=='ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97'
    os.replace(temporary,target)
    r=dict(bytes=target.stat().st_size,sha256=checksum.hexdigest(),official_match=True,seconds=time.time()-begin)
    (ROOT/'download_verified.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r),flush=True)
