"""Foreground-aware DINOv2 evaluation. Test labels are never loaded.

Research ablations: original two-view descriptor; D4 orientation averaging;
green-region patch pooling. Feature recipe and C are selected in INNER folds.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from plant_classifier import discover_training_images, read_submission, MEAN, STD


def green_mask(rgb):
    """Fixed colour heuristic, inspired by seedling segmentation literature.

    This is NOT a learned or annotated segmentation ground truth. Leaves which
    are not green can be missed, hence global descriptors remain candidates.
    """
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    hsv = np.asarray(rgb.convert('HSV'))
    exg = 2 * a[..., 1] - a[..., 0] - a[..., 2]
    return ((hsv[..., 0] >= 24) & (hsv[..., 0] <= 130) &
            (hsv[..., 1] >= 40) & (hsv[..., 2] >= 20) & (exg > .035)).astype(np.float32)


class FeatureImages(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.spatial = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)])
        self.tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = self.spatial(im.convert('RGB'))
            mask = torch.from_numpy(green_mask(im))[None]
            return self.tensor(im), mask


def fingerprint(paths, config):
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for p in paths:
        h.update(str(p.resolve()).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def extract(args):
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    rows, counts, root = discover_training_images(args.train_dir)
    sample = read_submission(args.sample)
    paths = [root / r[0] for r in rows] + [Path(args.test_dir) / r['ID'] for r in sample]
    meta = dict(model=args.model, size=args.size, views=['r0','r0flip','r90','r90flip','r180','r180flip','r270','r270flip'],
                classes=sorted(counts), train_ids=[r[0] for r in rows], labels=[r[1] for r in rows],
                test_ids=[r['ID'] for r in sample], mask='HSV24:130_S40_V20_ExG0.035', blocks=4)
    meta['fingerprint'] = fingerprint(paths, meta)
    target = out / 'features.npz'
    if target.exists() and (out / 'feature_metadata.json').exists():
        previous = json.loads((out / 'feature_metadata.json').read_text())
        if previous['fingerprint'] == meta['fingerprint']:
            print('Reusing verified feature cache', flush=True); return
        raise RuntimeError('Cache exists but data/config fingerprint changed; use a different run directory')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/dinov2', args.model, trust_repo=True).eval().to(device)
    model.requires_grad_(False)
    loader = DataLoader(FeatureImages(paths, args.size), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    feats = []; begin = time.time(); seen = 0
    with torch.inference_mode():
        for batch, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            views = []
            for rotation in range(4):
                for flip in (False, True):
                    x, mask = images.rot90(rotation, [2,3]), masks.rot90(rotation, [2,3])
                    if flip: x, mask = x.flip(3), mask.flip(3)
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                        layers = model.get_intermediate_layers(x, n=4, return_class_token=True)
                    cls = [F.normalize(c.float(),dim=1) for _, c in layers]
                    patches = layers[-1][0].float()
                    side = int(patches.shape[1] ** .5)
                    w = F.adaptive_avg_pool2d(mask, (side,side)).flatten(1)
                    # Fallback for images without enough detected green pixels.
                    w = torch.where(w.sum(1,keepdim=True) > .5, w, torch.ones_like(w))
                    region = (patches * w[...,None]).sum(1) / w.sum(1,keepdim=True)
                    view = torch.stack(cls + [F.normalize(patches.mean(1),dim=1), F.normalize(region,dim=1)], dim=1)
                    views.append(view.cpu().numpy())
            feats.append(np.stack(views,axis=1).astype(np.float32))
            seen += len(images)
            if (batch+1) % 20 == 0 or seen == len(paths):
                print(f'features {seen}/{len(paths)} elapsed={time.time()-begin:.0f}s gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)
    x = np.concatenate(feats)
    np.savez_compressed(target, train=x[:len(rows)], test=x[len(rows):])
    (out/'feature_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved {x.shape} features to {target}', flush=True)


class Recipe(BaseEstimator, TransformerMixin):
    def __init__(self, recipe='global', views=8):
        self.recipe = recipe; self.views = views

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        parts = {'global':[0,1,2,3,4], 'foreground':[0,1,2,3,5], 'combined':[0,1,2,3,4,5]}[self.recipe]
        x = X[:, :self.views, parts, :].copy()
        x = x.reshape(len(x),self.views,-1)
        x /= np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-8)
        x = x.mean(1)
        x /= np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8)
        return x


def make_probe():
    # Use a stable import path so joblib checkpoints load from other scripts.
    from plant_v3_features import Recipe as SerializableRecipe
    return Pipeline([('recipe',SerializableRecipe()),('scale',StandardScaler()),
                     ('classifier',LogisticRegression(max_iter=2000,C=1,solver='lbfgs'))])


def save_submission(path, ids, predictions):
    assert len(ids)==len(predictions) and len(set(ids))==len(ids)
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['ID','Category']); w.writerows(zip(ids,predictions))


def evaluate(args):
    out=Path(args.run_dir)
    meta=json.loads((out/'feature_metadata.json').read_text())
    data=np.load(out/'features.npz'); X=data['train']; Xt=data['test']; y=np.array(meta['labels'])
    classes=np.array(meta['classes']); cv=StratifiedKFold(5,shuffle=True,random_state=42)
    grid={'recipe__recipe':['global','foreground','combined'], 'recipe__views':[2,8],
          'classifier__C':[.01,.1,1.,10.]}
    oof=np.zeros((len(y),len(classes))); baseline_oof=np.zeros_like(oof); folds=np.zeros(len(y),dtype=int)
    results=[]; baseline=make_probe().set_params(recipe__recipe='global',recipe__views=2,classifier__C=10)
    with threadpool_limits(limits=2):
        for fold,(tr,va) in enumerate(cv.split(X,y),1):
            search=GridSearchCV(make_probe(),grid,cv=StratifiedKFold(4,shuffle=True,random_state=42+fold),
                                scoring='accuracy',n_jobs=1,error_score='raise')
            start=time.time(); search.fit(X[tr],y[tr]); p=search.predict_proba(X[va])
            assert np.array_equal(search.classes_,classes)
            oof[va]=p; folds[va]=fold
            baseline.fit(X[tr],y[tr]); baseline_oof[va]=baseline.predict_proba(X[va])
            result=dict(fold=fold,accuracy=accuracy_score(y[va],classes[p.argmax(1)]),
                        best_params=search.best_params_,seconds=time.time()-start)
            results.append(result); print(json.dumps(result),flush=True)
        final=GridSearchCV(make_probe(),grid,cv=StratifiedKFold(5,shuffle=True,random_state=47),
                           scoring='accuracy',n_jobs=1,error_score='raise')
        final.fit(X,y); test_prob=final.predict_proba(Xt)
        # Fixed old-recipe refit is also saved for paired comparison / audit.
        baseline.fit(X,y); baseline_test=baseline.predict_proba(Xt)
    pred=classes[oof.argmax(1)]
    report=dict(folds=results,accuracy=accuracy_score(y,pred),
                fold_sd=float(np.std([r['accuracy'] for r in results],ddof=1)),
                baseline_fixed_C10_accuracy=accuracy_score(y,classes[baseline_oof.argmax(1)]),
                final_params=final.best_params_,confusion_matrix=confusion_matrix(y,pred,labels=classes).tolist(),
                classification_report=classification_report(y,pred,output_dict=True),
                note='Inner CV selects feature recipe and C. Repeated experiment selection across rounds can bias CV. No test labels used.')
    (out/'cv_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(out/'probabilities.npz',oof=oof,test=test_prob,baseline_oof=baseline_oof,
                        baseline_test=baseline_test,labels=y,classes=classes,folds=folds)
    with (out/'oof_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['ID','true','pred','fold']+list(classes))
        for i in range(len(y)): w.writerow([meta['train_ids'][i],y[i],pred[i],folds[i]]+oof[i].tolist())
    joblib.dump({'estimator':final.best_estimator_,'metadata':meta},out/'classifier.joblib')
    save_submission(out/'submission.csv',meta['test_ids'],classes[test_prob.argmax(1)])
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['extract','evaluate','predict'])
    p.add_argument('--run-dir',default='outputs/stronger_20260929/dino_foreground')
    p.add_argument('--train-dir',default='work/task2_dataset/dataset-for-task2/dataset-for-task2/train')
    p.add_argument('--test-dir',default='work/task2_dataset/dataset-for-task2/dataset-for-task2/test')
    p.add_argument('--sample',default='work/task2_dataset/submission-for-task2.csv')
    p.add_argument('--model',default='dinov2_vitl14');p.add_argument('--size',type=int,default=518)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); os.environ.setdefault('TORCH_HOME',str(Path('work/torch_cache').resolve()))
    if a.mode=='extract': extract(a)
    elif a.mode=='evaluate': evaluate(a)
    else:
        out=Path(a.run_dir); saved=joblib.load(out/'classifier.joblib')
        cached=np.load(out/'features.npz')['test']
        save_submission(out/'submission.csv',saved['metadata']['test_ids'],saved['estimator'].predict(cached))


if __name__=='__main__': main()
