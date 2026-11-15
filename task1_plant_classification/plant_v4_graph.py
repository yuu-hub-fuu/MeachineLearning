"""Fixed clamped graph-propagation ablations on frozen deep descriptors.

This borrows the neighborhood-consistency motivation of LaplacianShot; it is
NOT a reproduction of LaplacianShot or TransCLIP. Outer query labels are never
read by the refinement function. Training labels stay clamped; query features
and probabilities are jointly refined, including unlabeled test images.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
from plant_v3_features import Recipe,save_submission
from plant_bioclip_probe import data_info

ROOT=Path('outputs/stronger_v4_20260929')

def refine(support,ys,query,prior,alpha):
    x=np.concatenate([support,query]);x=x.astype(np.float64)
    x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)
    sim=x@x.T;np.fill_diagonal(sim,-np.inf)
    neighbors=np.argsort(-sim,axis=1)[:,:5]
    values=np.take_along_axis(sim,neighbors,axis=1)
    weights=np.exp((values-values.max(1,keepdims=True))/.1)
    weights/=weights.sum(1,keepdims=True)
    targets=np.eye(prior.shape[1])[ys]
    p0=np.concatenate([targets,prior]);p=p0.copy();n=len(support)
    for step in range(100):
        new=(1-alpha)*p0+alpha*(p[neighbors]*weights[:,:,None]).sum(1)
        new[:n]=targets
        change=np.max(np.abs(new-p));p=new
        if change<1e-8:break
    return p[n:]/p[n:].sum(1,keepdims=True)

def main():
    _,meta=data_info();source=Path('outputs/stronger_20260929/dino_foreground')
    z=np.load(source/'features.npz');transform=Recipe(recipe='foreground',views=8)
    x=transform.transform(z['train']);xt=transform.transform(z['test'])
    baseline=np.load('outputs/stronger_20260929/dino_lpft/probabilities.npz')
    classes=np.array(meta['classes']);labels=np.array(meta['labels'])
    assert np.array_equal(classes,baseline['classes']) and np.array_equal(labels,baseline['labels'])
    sm=json.loads((source/'feature_metadata.json').read_text())
    assert sm['train_ids']==meta['train_ids'] and sm['test_ids']==meta['test_ids']
    meta.update(model='dinov2_vitl14',size=518,components='last4CLS,green_patch',
                feature_source_fingerprint=sm['fingerprint'],method='clamped graph with DINO LP-FT probability prior')
    meta.pop('revision');meta.pop('weights_sha256')
    y=np.array([list(classes).index(label) for label in labels]);folds=baseline['folds']
    with threadpool_limits(2):
        for alpha in (.2,.5):
            out=ROOT/f'graph_a{int(alpha*10)}';out.mkdir(exist_ok=True)
            p=np.zeros_like(baseline['oof']);scores=[]
            for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y),1):
                assert np.all(folds[va]==fold)
                p[va]=refine(x[tr],y[tr],x[va],baseline['oof'][va],alpha)
                scores.append(float(accuracy_score(labels[va],classes[p[va].argmax(1)])))
            tp=refine(x,y,xt,baseline['test'],alpha)
            report=dict(accuracy=float(accuracy_score(labels,classes[p.argmax(1)])),fold_scores=scores,
                        alpha=alpha,k=5,temperature=.1,iterations=100,tolerance=1e-8,
                        note='Fixed clamped graph adaptation, transductive unlabeled queries; no query labels in refinement. Not a paper reproduction.')
            np.savez_compressed(out/'probabilities.npz',oof=p,test=tp,labels=labels,classes=classes,folds=folds)
            (out/'cv_report.json').write_text(json.dumps(report,indent=2))
            (out/'feature_metadata.json').write_text(json.dumps(meta,indent=2))
            save_submission(out/'submission.csv',meta['test_ids'],classes[tp.argmax(1)])
            print(json.dumps(report),flush=True)

if __name__=='__main__':main()
