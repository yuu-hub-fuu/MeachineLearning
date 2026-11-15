"""Extract only changed crop images; verify and reuse pixel-identical features."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from plant_v3_features import FeatureImages,fingerprint
from plant_bioclip_probe import data_info

def main():
    original,original_meta=data_info()
    root=Path('outputs/stronger_v4_20260929');out=root/'dino_crop';out.mkdir(exist_ok=True)
    manifest=json.loads((root/'crop_manifest.json').read_text())
    assert manifest['original_data_hash']==original_meta['data_hash']
    source=Path('outputs/stronger_20260929/dino_foreground')
    meta=json.loads((source/'feature_metadata.json').read_text());meta.pop('fingerprint')
    assert meta['train_ids']==manifest['train_ids'] and meta['test_ids']==manifest['test_ids']
    paths=[Path('work/task2_crops_v4')/r['id'] for r in manifest['records']]
    changed=[]
    for i,(path,r) in enumerate(zip(paths,manifest['records'])):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['crop_sha256']
        if r['area_fraction']<1:changed.append(i)
        else:
            with Image.open(original[i]) as a,Image.open(path) as b:
                assert np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB')))
    meta['fingerprint']=fingerprint(paths,meta)
    if (out/'features.npz').exists():
        assert json.loads((out/'feature_metadata.json').read_text())==meta
        print('Reusing verified cropped feature cache',flush=True);return
    z=np.load(source/'features.npz');X=np.concatenate([z['train'],z['test']]);z.close()
    os.environ.setdefault('TORCH_HOME',str(Path('work/torch_cache').resolve()));torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=torch.hub.load('facebookresearch/dinov2',meta['model'],trust_repo=True).eval().to(device)
    model.requires_grad_(False)
    batches=DataLoader(FeatureImages([paths[i] for i in changed],meta['size']),batch_size=2,shuffle=False,num_workers=0)
    seen=0;begin=time.time()
    with torch.inference_mode():
        for images,masks in batches:
            images,masks=images.to(device),masks.to(device);views=[]
            for j in range(8):
                x,mask=images.rot90(j//2,[2,3]),masks.rot90(j//2,[2,3])
                if j%2:x,mask=x.flip(3),mask.flip(3)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    layers=model.get_intermediate_layers(x,n=4,return_class_token=True)
                cls=[F.normalize(c.float(),dim=1) for _,c in layers];patches=layers[-1][0].float()
                side=int(patches.shape[1]**.5);w=F.adaptive_avg_pool2d(mask,(side,side)).flatten(1)
                w=torch.where(w.sum(1,keepdim=True)>.5,w,torch.ones_like(w))
                region=(patches*w[...,None]).sum(1)/w.sum(1,keepdim=True)
                views.append(torch.stack(cls+[F.normalize(patches.mean(1),dim=1),F.normalize(region,dim=1)],dim=1).cpu().numpy())
            batch=np.stack(views,axis=1);X[np.array(changed[seen:seen+len(images)])]=batch;seen+=len(images)
            if seen%40==0 or seen==len(changed):print(json.dumps({'changed_images':seen,'total':len(changed),'seconds':time.time()-begin}),flush=True)
    assert np.isfinite(X).all()
    np.savez_compressed(out/'features.npz',train=X[:500],test=X[500:])
    (out/'feature_metadata.json').write_text(json.dumps(meta,indent=2))
    (out/'crop_feature_reuse.json').write_text(json.dumps(dict(extracted=len(changed),reused=len(paths)-len(changed),
             original_data_hash=original_meta['data_hash'],source=str(source/'features.npz'),unchanged_images_verified_pixel_equal=True),indent=2))
    print('Cropped features ready',flush=True)

if __name__=='__main__':main()
