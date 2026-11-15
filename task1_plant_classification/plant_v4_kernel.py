"""Nested RBF support-vector probes on frozen DINO/BioCLIP features.

Kernels are pairwise functions of independently normalized descriptors. Every
SVC fit, including its internal probability calibration, sees fold-train only.
No statistics are fitted to the test set; no test labels are available.
"""
import itertools
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from plant_v3_features import Recipe,save_submission
from plant_bioclip_probe import BioProbe,data_info

ROOT=Path('outputs/stronger_v4_20260929')
OUT=ROOT/'kernel'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    _,meta=data_info();y=np.array(meta['labels']);classes=np.array(meta['classes']);n=len(y)
    paths=[Path('outputs/stronger_20260929/dino_foreground'),ROOT/'bioclip25']
    meta.update(model='DINOv2-L + BioCLIP 2.5 Huge RBF kernels',size=[518,224],
                components='DINO foreground5120 + BioCLIP foreground6400',
                source_backbones=[json.loads((p/'feature_metadata.json').read_text()) for p in paths])
    banks=[]
    for i,p in enumerate(paths):
        m=json.loads((p/'feature_metadata.json').read_text())
        assert m['train_ids']==meta['train_ids'] and m['test_ids']==meta['test_ids']
        z=np.load(p/'features.npz');x=np.concatenate([z['train'],z['test']])
        x=Recipe(recipe='foreground',views=8).transform(x) if i==0 else BioProbe(recipe='foreground',per_view=False).features(x)
        x=x.astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)
        banks.append(x)
    distances=[np.maximum(0,2-2*x@x.T) for x in banks]
    # Fixed, modest grid; all choices selected within the outer training fold.
    grid=list(itertools.product([0.,.5,1.],[1.,10.,50.],[1.,10.,100.]))
    kernels={(w,g):(1-w)*np.exp(-g*distances[0])+w*np.exp(-g*distances[1])
             for w,g in itertools.product([0.,.5,1.],[1.,10.,50.])}
    def fit_predict(params,tr,va,seed):
        w,g,c=params;k=kernels[w,g]
        model=SVC(C=c,kernel='precomputed',probability=True,random_state=seed)
        model.fit(k[np.ix_(tr,tr)],y[tr]);assert np.array_equal(model.classes_,classes)
        return model,model.predict_proba(k[np.ix_(va,tr)])
    def choose(indices,splits,seed):
        inner=list(StratifiedKFold(splits,shuffle=True,random_state=seed).split(indices,y[indices]))
        scored=[]
        for params in grid:
            correct=0
            for j,(it,iv) in enumerate(inner):
                _,p=fit_predict(params,indices[it],indices[iv],seed+j)
                correct+=int((classes[p.argmax(1)]==y[indices[iv]]).sum())
            scored.append((correct,params))
        # Deterministic first-in-grid tie rule, not outer-fold performance.
        return max(scored,key=lambda row:row[0])[1]
    oof=np.zeros((n,len(classes)));folds=np.zeros(n,int);details=[]
    with threadpool_limits(2):
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y),1):
            start=time.time();params=choose(tr,4,42+fold)
            _,p=fit_predict(params,tr,va,42+fold);oof[va]=p;folds[va]=fold
            row=dict(fold=fold,accuracy=float(np.mean(classes[p.argmax(1)]==y[va])),
                     params=dict(bioclip_weight=params[0],gamma=params[1],C=params[2]),seconds=round(time.time()-start,2))
            details.append(row);print(json.dumps(row),flush=True)
        params=choose(np.arange(n),5,47)
        model,tp=fit_predict(params,np.arange(n),np.arange(n,len(banks[0])),47)
    report=dict(accuracy=float(np.mean(classes[oof.argmax(1)]==y)),folds=details,
                final_params=dict(bioclip_weight=params[0],gamma=params[1],C=params[2]),
                protocol='Nested image-level CV; fixed RBF kernels; SVC calibration uses training subset only.')
    np.savez_compressed(OUT/'probabilities.npz',oof=oof,test=tp,labels=y,classes=classes,folds=folds)
    joblib.dump(dict(model=model,train_features=[x[:n] for x in banks],params=params,metadata=meta),OUT/'classifier.joblib')
    (OUT/'cv_report.json').write_text(json.dumps(report,indent=2))
    (OUT/'feature_metadata.json').write_text(json.dumps(meta,indent=2))
    save_submission(OUT/'submission.csv',meta['test_ids'],classes[tp.argmax(1)])
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
