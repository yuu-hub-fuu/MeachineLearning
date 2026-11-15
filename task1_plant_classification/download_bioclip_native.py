"""Use Hugging Face's native Xet transport for the pinned public release."""
import os
from pathlib import Path
import hashlib
import json
import time

os.environ['HF_HOME']=str(Path('work/hf_bioclip').resolve())
os.environ['HF_HUB_DISABLE_XET']='0'
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_ETAG_TIMEOUT']='30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'

from huggingface_hub import hf_hub_download

if __name__=='__main__':
    begin=time.time()
    path=Path(hf_hub_download(repo_id='imageomics/bioclip-2.5-vith14',filename='open_clip_model.safetensors',
                            revision='6e3d04e3d6522012c88181085c5ae666e14c45cd',local_dir='work/bioclip25',token=False))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()=='ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97'
    r=dict(file=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),official_match=True,seconds=time.time()-begin)
    (path.parent/'download_verified.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r),flush=True)
