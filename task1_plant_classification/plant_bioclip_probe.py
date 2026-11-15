"""BioCLIP 2.5 Huge feature adaptation with nested, image-level held-out folds.

Official vision weights only. No species text prompts or external class labels
are used: all supervised fitting uses the supplied training subset of each fold.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from threadpoolctl import threadpool_limits
from plant_classifier import discover_training_images,read_submission
from plant_v3_features import save_submission


ROOT=Path('outputs/stronger_v4_20260929')
CACHE=Path('work/bioclip25')
WEIGHT_SHA='ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97'
REVISION='6e3d04e3d6522012c88181085c5ae666e14c45cd'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def data_info():
    cfg=json.loads(Path('outputs/stronger_20260929/dino_lpft/config.json').read_text())
    rows,counts,root=discover_training_images(cfg['train_dir'])
    sample=read_submission(cfg['sample'])
    paths=[root/r[0] for r in rows]+[Path(cfg['test_dir'])/r['ID'] for r in sample]
    h=hashlib.sha256()
    for path in paths:h.update(path.name.encode());h.update(hashlib.sha256(path.read_bytes()).digest())
    assert h.hexdigest()==cfg['data_hash']
    meta=dict(train_ids=[r[0] for r in rows],test_ids=[r['ID'] for r in sample],
              labels=[r[1] for r in rows],classes=sorted(counts),data_hash=h.hexdigest(),
              model='imageomics/bioclip-2.5-vith14',revision=REVISION,weights_sha256=WEIGHT_SHA,
              size=224,views=8,components='projected1024,last4CLS5120,mean_patch1280,green_patch1280')
    return paths,meta


def load_visual(device):
    from open_clip.model import _build_vision_tower
    from safetensors import safe_open
    config=json.loads((CACHE/'open_clip_config.json').read_text())
    checkpoint=CACHE/'open_clip_model.safetensors'
    actual=sha(checkpoint)
    if actual!=WEIGHT_SHA:raise ValueError(f'Official weight hash mismatch: {actual}')
    dtype=torch.float16 if device.type=='cuda' else torch.float32
    # Meta construction + per-tensor loading avoids a second 2.5 GB CPU model.
    with torch.device('meta'):
        model=_build_vision_tower(config['model_cfg']['embed_dim'],config['model_cfg']['vision_cfg'],cast_dtype=dtype)
    state={}
    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        for key in f.keys():
            if key.startswith('visual.'):
                name=key[len('visual.'):]
                tensor=f.get_tensor(key)
                # LayerNormFp32 expects its affine weights to stay float32.
                cast=torch.float32 if ('ln_' in name) else dtype
                state[name]=tensor.to(device=device,dtype=cast)
    model.load_state_dict(state,strict=True,assign=True)
    del state
    model.eval().requires_grad_(False)
    return model,config


class Images(Dataset):
    def __init__(self,paths,preprocess):self.paths=paths;self.preprocess=preprocess
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:return self.preprocess(im.convert('RGB'))


def extract(out,batch_size):
    from open_clip.transform import PreprocessCfg,image_transform_v2
    paths,meta=data_info();out.mkdir(parents=True,exist_ok=True)
    if (out/'features.npz').exists():
        assert json.loads((out/'feature_metadata.json').read_text())==meta
        print('Reusing verified BioCLIP feature cache',flush=True);return
    torch.set_num_threads(3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,config=load_visual(device)
    preprocess=image_transform_v2(PreprocessCfg(size=224,**config['preprocess_cfg']),is_train=False)
    batches=DataLoader(Images(paths,preprocess),batch_size=batch_size,shuffle=False,num_workers=0)
    mean=torch.tensor(config['preprocess_cfg']['mean'],device=device).view(1,3,1,1)
    std=torch.tensor(config['preprocess_cfg']['std'],device=device).view(1,3,1,1)
    result=[];begin=time.time();seen=0
    with torch.inference_mode():
        for x in batches:
            x=x.to(device);views=[]
            for j in range(8):
                v=x.rot90(j//2,[2,3]);v=v.flip(3) if j%2 else v
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    outputs=model.forward_intermediates(v,indices=4,normalize_intermediates=True,
                                                       output_fmt='NLC',output_extra_tokens=True)
                rgb=v.float()*std+mean;r,g,b=rgb.unbind(1)
                green=F.relu(2*g-r-b-.035)*((g>r)&(g>b))
                patch=outputs['image_intermediates'][-1].float();side=int(patch.shape[1]**.5)
                w=F.adaptive_avg_pool2d(green[:,None],(side,side)).flatten(1)
                w=torch.where(w.sum(1,keepdim=True)>.05,w,torch.ones_like(w))
                fg=(patch*w[...,None]).sum(1)/w.sum(1,keepdim=True)
                parts=[outputs['image_features'].float()]+[c[:,0].float() for c in outputs['image_intermediates_prefix']]
                parts+=[patch.mean(1),fg]
                views.append(torch.cat([F.normalize(p,dim=1) for p in parts],dim=1).cpu().numpy())
            result.append(np.stack(views,axis=1));seen+=len(x)
            if seen%40==0 or seen==len(paths):
                print(json.dumps(dict(images=seen,total=len(paths),seconds=round(time.time()-begin,1),gpu_peak_gb=round(torch.cuda.max_memory_allocated()/2**30,2))),flush=True)
    X=np.concatenate(result);assert X.shape==(878,8,8704) and np.isfinite(X).all()
    n=len(meta['labels']);np.savez_compressed(out/'features.npz',train=X[:n],test=X[n:])
    (out/'feature_metadata.json').write_text(json.dumps(meta,indent=2))
    print('BioCLIP feature extraction complete',flush=True)


class BioProbe(ClassifierMixin,BaseEstimator):
    def __init__(self,recipe='clip',per_view=False,C=1.):self.recipe=recipe;self.per_view=per_view;self.C=C
    def features(self,X):
        if self.recipe=='clip':z=X[:,:,:1024].copy()
        elif self.recipe=='global':z=X[:,:,1024:7424].copy()
        elif self.recipe=='foreground':z=np.concatenate([X[:,:,1024:6144],X[:,:,7424:]],axis=-1)
        else:raise ValueError(self.recipe)
        z/=np.linalg.norm(z,axis=-1,keepdims=True).clip(1e-8)
        if self.per_view:return z.reshape(-1,z.shape[-1])
        z=z.mean(1);return z/np.linalg.norm(z,axis=-1,keepdims=True).clip(1e-8)
    def fit(self,X,y):
        self.classes_=np.unique(y);v=X.shape[1] if self.per_view else 1
        self.estimator_=make_pipeline(StandardScaler(),LogisticRegression(C=self.C/v,max_iter=2000))
        self.estimator_.fit(self.features(X),np.repeat(y,v) if self.per_view else y)
        return self
    def predict_proba(self,X):
        p=self.estimator_.predict_proba(self.features(X))
        return p.reshape(len(X),X.shape[1],-1).mean(1) if self.per_view else p
    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]


def evaluate(out):
    _,current=data_info();meta=json.loads((out/'feature_metadata.json').read_text());assert meta==current
    z=np.load(out/'features.npz');X=z['train'];Xt=z['test'];y=np.array(meta['labels']);classes=np.array(meta['classes'])
    from plant_bioclip_probe import BioProbe as SerializableProbe
    grid=dict(recipe=['clip','global','foreground'],per_view=[False,True],C=[.1,1.,10.])
    oof=np.zeros((len(y),len(classes)));folds=np.zeros(len(y),int);details=[]
    with threadpool_limits(2):
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y),1):
            begin=time.time();search=GridSearchCV(SerializableProbe(),grid,cv=StratifiedKFold(4,shuffle=True,random_state=42+fold),n_jobs=1,error_score='raise',scoring='accuracy')
            search.fit(X[tr],y[tr]);assert np.array_equal(search.classes_,classes)
            oof[va]=search.predict_proba(X[va]);folds[va]=fold
            row=dict(fold=fold,accuracy=float(accuracy_score(y[va],classes[oof[va].argmax(1)])),params=search.best_params_,seconds=time.time()-begin)
            details.append(row);print(json.dumps(row),flush=True)
        search=GridSearchCV(SerializableProbe(),grid,cv=StratifiedKFold(5,shuffle=True,random_state=47),n_jobs=1,error_score='raise',scoring='accuracy')
        search.fit(X,y);tp=search.predict_proba(Xt)
    report=dict(accuracy=float(accuracy_score(y,classes[oof.argmax(1)])),folds=details,final_params=search.best_params_,
                confusion_matrix=confusion_matrix(y,classes[oof.argmax(1)],labels=classes).tolist(),
                note='Nested image-level folds; biological backbone remains frozen; classifiers fit only fold training labels.')
    np.savez_compressed(out/'probabilities.npz',oof=oof,test=tp,labels=y,classes=classes,folds=folds)
    joblib.dump(dict(estimator=search.best_estimator_,metadata=meta),out/'classifier.joblib')
    (out/'cv_report.json').write_text(json.dumps(report,indent=2))
    save_submission(out/'submission.csv',meta['test_ids'],classes[tp.argmax(1)])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','evaluate'])
    p.add_argument('--run-dir',default=str(ROOT/'bioclip25'));p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();out=Path(a.run_dir)
    if a.mode=='extract':extract(out,a.batch_size)
    else:evaluate(out)
