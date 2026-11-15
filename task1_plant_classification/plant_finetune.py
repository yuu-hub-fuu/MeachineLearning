"""LP-FT-inspired ConvNeXt adaptation with fixed-epoch, held-out 5-fold evaluation.

The linear head is fitted only on the training part of each fold, then both
backbone and head are optimized. The final epoch count is fixed before training;
outer-fold validation scores do not pick a checkpoint. Test predictions average
the five final EMA models. This is an adaptation, not a paper reproduction.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from plant_classifier import MEAN, STD, discover_training_images, read_submission
from plant_v3_features import save_submission


class QuarterTurn:
    def __call__(self, image):
        return image.rotate(random.choice((0,90,180,270)))


class Images(Dataset):
    def __init__(self, images, indices, labels, size, augment=False):
        self.images=images; self.indices=indices; self.labels=labels
        if augment:
            ops=[T.RandomResizedCrop(size,scale=(.75,1.),ratio=(.9,1.1),interpolation=T.InterpolationMode.BICUBIC),
                 QuarterTurn(), T.RandomHorizontalFlip(),
                 T.RandomApply([T.ColorJitter(.15,.15,.12,.015)],p=.7)]
        else:
            ops=[T.Resize(size,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(size)]
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize(MEAN,STD)])

    def __len__(self): return len(self.indices)

    def __getitem__(self,i):
        k=self.indices[i]
        return self.transform(self.images[k]),int(self.labels[k])


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def load_images(paths):
    images=[]
    for p in paths:
        with Image.open(p) as im:
            images.append(ImageOps.contain(im.convert('RGB'),(640,640)))
    return images


def loader(images,idx,y,size,batch,augment=False):
    # RAM-cached PIL images avoid re-decoding source PNGs on every epoch.
    return DataLoader(Images(images,idx,y,size,augment),batch_size=batch,
                      shuffle=augment,num_workers=0,pin_memory=True)


def features_model(device):
    model=convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1]=nn.Identity()
    return model.to(device).eval()


@torch.inference_mode()
def probabilities(model, batches, device, views=8):
    model.eval(); result=[]
    for x,_ in batches:
        x=x.to(device); prob=[]
        for j in range(views):
            rotated=x.rot90(j//2,[2,3]); rotated=rotated.flip(3) if j%2 else rotated
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                logits=model(rotated)
            prob.append(logits.float().softmax(1))
        result.append(torch.stack(prob).mean(0).cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def initial_features(model,batches,device):
    feats=[]
    for x,_ in batches:
        x=x.to(device)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            f=(model(x)+model(x.flip(3)))/2
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def initialize_head(model,features,y,device):
    # Convert StandardScaler + logistic regression into an equivalent Linear.
    with threadpool_limits(2):
        scale=StandardScaler().fit(features)
        lr=LogisticRegression(C=1,max_iter=2000).fit(scale.transform(features),y)
    w=lr.coef_/scale.scale_[None,:]
    b=lr.intercept_-w@scale.mean_
    head=nn.Linear(features.shape[1],len(lr.classes_))
    with torch.no_grad():
        head.weight.copy_(torch.tensor(w,dtype=torch.float32));head.bias.copy_(torch.tensor(b,dtype=torch.float32))
    converted=head(torch.from_numpy(features)).detach().numpy()
    expected=lr.decision_function(scale.transform(features))
    if not np.allclose(converted,expected,atol=2e-4,rtol=1e-4):
        raise RuntimeError('Linear-head conversion failed equivalence check')
    model.classifier[-1]=head.to(device)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default='outputs/stronger_20260929/convnext_lpft')
    p.add_argument('--train-dir',default='work/task2_dataset/dataset-for-task2/dataset-for-task2/train')
    p.add_argument('--test-dir',default='work/task2_dataset/dataset-for-task2/dataset-for-task2/test')
    p.add_argument('--sample',default='work/task2_dataset/submission-for-task2.csv')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--size',type=int,default=288)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--backbone-lr',type=float,default=5e-5);p.add_argument('--head-lr',type=float,default=5e-4)
    a=p.parse_args();os.environ.setdefault('TORCH_HOME',str(Path('work/torch_cache').resolve()))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True)
    rows,counts,root=discover_training_images(a.train_dir);classes=np.array(sorted(counts))
    labels=np.array([list(classes).index(r[1]) for r in rows]); sample=read_submission(a.sample)
    train_paths=[root/r[0] for r in rows];test_paths=[Path(a.test_dir)/r['ID'] for r in sample]
    h=hashlib.sha256()
    for path in train_paths+test_paths: h.update(path.name.encode());h.update(hashlib.sha256(path.read_bytes()).digest())
    config=dict(vars(a),classes=classes.tolist(),seed=42,folds=5,protocol='fixed final EMA; no outer-val epoch selection',
                data_hash=h.hexdigest(),script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise RuntimeError('Config differs from existing run; use new run-dir')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    def log(event):
        print(json.dumps(event),flush=True)
        with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    train_images=load_images(train_paths);test_images=load_images(test_paths); n=len(rows)
    fp=out/'initial_features.npy'
    if fp.exists(): feats=np.load(fp)
    else:
        seed_all(42);model=features_model(device)
        feats=initial_features(model,loader(train_images,np.arange(n),labels,a.size,a.batch_size),device)
        np.save(fp,feats);del model;torch.cuda.empty_cache()
    oof=np.zeros((n,len(classes)));test_preds=[];scores=[];fold_ids=np.zeros(n,int)
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),labels),1):
        done=out/f'fold{fold}_predictions.npz';fold_ids[va]=fold
        if done.exists():
            cached=np.load(done);assert np.array_equal(cached['val_indices'],va)
            oof[va]=cached['val'];test_preds.append(cached['test'])
            scores.append(float(accuracy_score(labels[va],cached['val'].argmax(1))));continue
        seed_all(42+fold);model=features_model(device);initialize_head(model,feats[tr],labels[tr],device)
        model.requires_grad_(True)
        optimizer=torch.optim.AdamW([{'params':model.features.parameters(),'lr':a.backbone_lr},
                                    {'params':model.classifier.parameters(),'lr':a.head_lr}],weight_decay=.05)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        train=loader(train_images,tr,labels,a.size,a.batch_size,True)
        valid=loader(train_images,va,labels,a.size,a.batch_size)
        test=loader(test_images,np.arange(len(test_images)),np.zeros(len(test_images)),a.size,a.batch_size)
        ce=nn.CrossEntropyLoss(label_smoothing=.03);begin=time.time()
        for epoch in range(a.epochs):
            # Two warmup epochs; subsequent cosine decay. This schedule is fixed.
            factor=(epoch+1)/2 if epoch<2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/max(1,a.epochs-3)))
            for group,base in zip(optimizer.param_groups,[a.backbone_lr,a.head_lr]):group['lr']=base*factor
            model.train();loss_sum=0.;seen=0;optimizer.zero_grad(set_to_none=True)
            for step,(x,y) in enumerate(train):
                x=x.to(device);y=y.to(device);mix=random.random()<.25
                lam=float(np.random.beta(.2,.2)) if mix else 1.; perm=torch.randperm(len(x),device=device)
                x=lam*x+(1-lam)*x[perm] if mix else x
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    logits=model(x);loss=lam*ce(logits,y)+(1-lam)*ce(logits,y[perm])
                loss_sum+=loss.item()*len(y);seen+=len(y)
                # Last accumulation group may have fewer microbatches.
                group_size=min(a.accumulate,len(train)-(step//a.accumulate)*a.accumulate)
                scaler.scale(loss/group_size).backward()
                if (step+1)%a.accumulate==0 or step+1==len(train):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale()>=old_scale:
                        with torch.no_grad():
                            for e,m in zip(ema.parameters(),model.parameters()):e.lerp_(m,.02)
                            for e,m in zip(ema.buffers(),model.buffers()):e.copy_(m)
            event=dict(fold=fold,epoch=epoch+1,loss=loss_sum/seen,seconds=round(time.time()-begin,1))
            if (epoch+1)%5==0:
                vp=probabilities(ema,valid,device,views=2)
                event['diagnostic_val_acc']=float(accuracy_score(labels[va],vp.argmax(1)))
            log(event)
        vp=probabilities(ema,valid,device);tp=probabilities(ema,test,device)
        oof[va]=vp;test_preds.append(tp);score=float(accuracy_score(labels[va],vp.argmax(1)));scores.append(score)
        torch.save(dict(state_dict={k:v.cpu() for k,v in ema.state_dict().items()},classes=classes.tolist(),config=config,fold=fold),out/f'fold{fold}.pth')
        np.savez_compressed(done,val_indices=va,val=vp,test=tp)
        log(dict(fold=fold,final_accuracy=score,elapsed=round(time.time()-begin,1)))
        del model,ema,optimizer,scaler;torch.cuda.empty_cache()
    test=np.mean(test_preds,axis=0);pred=oof.argmax(1)
    np.savez_compressed(out/'probabilities.npz',oof=oof,test=test,labels=classes[labels],classes=classes,folds=fold_ids)
    report=dict(accuracy=float(accuracy_score(labels,pred)),fold_scores=scores,fold_sd=float(np.std(scores,ddof=1)),
                confusion_matrix=confusion_matrix(labels,pred).tolist(),classification_report=classification_report(labels,pred,target_names=classes,output_dict=True),
                protocol=config['protocol'])
    (out/'cv_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save_submission(out/'submission.csv',[r['ID'] for r in sample],classes[test.argmax(1)])
    log(report)


if __name__=='__main__':main()
