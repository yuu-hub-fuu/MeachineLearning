"""Six-class FER: one seed, one grouped holdout, CCF-A-inspired ablations."""
import os
from pathlib import Path

# Set before importing any ML library. No pretrained weights/cache on C:.
ROOT = Path(os.environ.get('FER_ROOT', r'D:\FER2026task2'))
if ROOT.drive.lower() == 'c:':
    raise RuntimeError('FER_ROOT must not be on C:')
for key, part in {'TORCH_HOME':'torch','HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg',
                  'TEMP':'tmp','TMP':'tmp','TORCHINDUCTOR_CACHE_DIR':'inductor'}.items():
    os.environ[key] = str(ROOT/'cache'/part)
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import argparse, copy, csv, hashlib, io, json, math, random, time, zipfile
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

CLASSES = ['Angry','Fear','Happy','Sad','Surprise','Neutral']
LABELS = {name:i for i,name in enumerate(CLASSES)}
PRETRAINED = ROOT/'pretrained'/'enet_b0_8_best_vgaf.pt'

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare(zip_path, sample_path):
    dest=ROOT/'data'; dest.mkdir(parents=True,exist_ok=True)
    images=[]; labels=[]; ids=[]; groups=[]; test={}
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith('.jpg'): continue
            arr=np.array(Image.open(io.BytesIO(z.read(name))).convert('L'))
            if arr.shape != (48,48): raise ValueError((name, arr.shape))
            if '/train/' in name:
                images.append(arr); labels.append(LABELS[name.split('/')[-2]]); ids.append(name)
                groups.append(hashlib.sha256(arr.tobytes()).hexdigest())
            else: test[name.split('/')[-1]]=arr
    with open(sample_path,encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); assert r.fieldnames==['ID','Emotion']; test_ids=[row['ID'] for row in r]
    assert len(test_ids)==len(set(test_ids)) and set(test_ids)==set(test)
    y=np.array(labels,dtype=np.int64); groups=np.array(groups); ids=np.array(ids)
    group_labels=defaultdict(list)
    for g,label in zip(groups,labels): group_labels[g].append(label)
    unique=sorted(group_labels)
    majority=[Counter(group_labels[g]).most_common(1)[0][0] for g in unique]
    # One stratified group holdout, NOT k-fold training. Mixed-label groups stay together.
    train_groups,val_groups=train_test_split(unique,test_size=0.15,random_state=42,stratify=majority)
    val_mask=np.isin(groups,val_groups); train_idx=np.flatnonzero(~val_mask); val_idx=np.flatnonzero(val_mask)
    assert not set(groups[train_idx]) & set(groups[val_idx])
    np.savez(dest/'dataset.npz',train_images=np.stack(images),labels=y,train_ids=ids,groups=groups,
             train_idx=train_idx,val_idx=val_idx,test_images=np.stack([test[k] for k in test_ids]),test_ids=np.array(test_ids))
    report={'label_mapping':LABELS,'train_total':len(y),'holdout_train':len(train_idx),'holdout_val':len(val_idx),
            'train_counts':np.bincount(y[train_idx],minlength=6).tolist(),'val_counts':np.bincount(y[val_idx],minlength=6).tolist(),
            'test_total':len(test_ids),'split_seed':42,'folds_trained':0,'split_policy':'one 15% stratified pixel-hash-group holdout',
            'split_sha256':hashlib.sha256(ids[val_idx].tobytes()).hexdigest(),
            'source_zip_sha256':hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()}
    dump(dest/'split.json',report)
    with (dest/'split.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['image','label','pixel_group','partition'])
        w.writerows(zip(ids,y,groups,np.where(val_mask,'validation','train')))
    print(json.dumps(report),flush=True)

class Faces(Dataset):
    def __init__(self,images,labels,size,augment=False):
        self.images=images; self.labels=labels
        ops=[T.Resize((size,size),interpolation=T.InterpolationMode.BILINEAR)]
        if augment:
            ops += [T.RandomHorizontalFlip(),T.RandomAffine(10,translate=(.04,.04),scale=(.92,1.08),
                    interpolation=T.InterpolationMode.BILINEAR,fill=128)]
        ops += [T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]
        if augment: ops += [T.RandomErasing(p=.1,scale=(.02,.08),ratio=(.5,2.))]
        self.transform=T.Compose(ops)
    def __len__(self): return len(self.images)
    def __getitem__(self,i):
        x=self.transform(Image.fromarray(self.images[i]).convert('RGB'))
        return x,int(self.labels[i]) if self.labels is not None else i

class LocalRegions(nn.Module):
    """EfficientFace-inspired four-region depthwise features; a custom adaptation."""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(in_channels,128,1,bias=False),nn.BatchNorm2d(128),nn.SiLU())
        self.regions=nn.ModuleList([nn.Sequential(nn.Conv2d(128,128,3,padding=1,groups=128,bias=False),
                         nn.BatchNorm2d(128),nn.SiLU(),nn.AdaptiveAvgPool2d(1)) for _ in range(4)])
        self.fuse=nn.Sequential(nn.Linear(512,out_channels),nn.LayerNorm(out_channels),nn.SiLU())
        self.scale=nn.Parameter(torch.tensor(.05))
    def forward(self,x):
        x=self.project(x); h,w=x.shape[-2:]; a,b=h//2,w//2
        patches=[x[:,:,:a,:b],x[:,:,:a,b:],x[:,:,a:,:b],x[:,:,a:,b:]]
        return self.scale*self.fuse(torch.cat([m(p).flatten(1) for m,p in zip(self.regions,patches)],dim=1))

class FER(nn.Module):
    def __init__(self,variant,initialize_pretrained=True):
        super().__init__(); self.variant=variant
        # Official EmotiEffLib supplies a serialized timm module; loaded from the explicit D: path.
        # Current timm module rebuild avoids depending on old pickled forward implementations.
        self.backbone=timm.create_model('tf_efficientnet_b0.ns_jft_in1k',pretrained=False,num_classes=8)
        if initialize_pretrained:
            original=torch.load(PRETRAINED,map_location='cpu',weights_only=False)
            state=original.state_dict()
            if isinstance(original.classifier,nn.Sequential):
                state={k.replace('classifier.0.','classifier.'):v for k,v in state.items()}
            self.backbone.load_state_dict(state,strict=True)
        old=self.backbone.classifier; channels=old.in_features
        self.classifier=nn.Linear(channels,6)
        # Official 8-class order: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.
        order=[0,3,4,6,7,5]
        with torch.no_grad():
            self.classifier.weight.copy_(old.weight[order]); self.classifier.bias.copy_(old.bias[order])
        self.backbone.classifier=nn.Identity()
        self.local=LocalRegions(40,channels) if variant!='baseline' else None
        self.confidence=nn.Linear(channels,1) if variant=='scn' else None
        self.log_variance=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,64)) if variant=='rul' else None
        self.dropout=nn.Dropout(.2)
    def features(self,x):
        if self.local is None:
            maps=self.backbone.forward_features(x)
            return maps.mean((2,3))
        # Capture the stride-8 map (40x20x20 at input 160) before coarse global features.
        x=self.backbone.bn1(self.backbone.conv_stem(x))
        local_map=None
        for index,stage in enumerate(self.backbone.blocks):
            x=stage(x)
            if index==2: local_map=x
        maps=self.backbone.bn2(self.backbone.conv_head(x))
        f=maps.mean((2,3))
        f=f+self.local(local_map)
        return f
    def forward(self,x):
        f=self.features(x); logits=self.classifier(self.dropout(f))
        if not self.training and self.confidence is not None:
            logits=logits*torch.sigmoid(self.confidence(f))
        return logits,f

def objective(model,logits,features,y,epoch):
    ce=F.cross_entropy(logits,y,label_smoothing=.05)
    if model.variant=='scn':
        alpha=torch.sigmoid(model.confidence(features))
        weighted=F.cross_entropy(alpha*logits,y,label_smoothing=.05)
        ranked=alpha.flatten().sort(descending=True).values
        k=max(1,min(len(y)-1,int(.7*len(y))))
        rr=F.relu(.07-ranked[:k].mean()+ranked[k:].mean())
        # SCN importance weighting + rank regularization; no hard relabeling in this adaptation.
        return weighted+rr
    if model.variant=='rul' and epoch>=2:
        # RUL official mixup_data uses uncertainty/(u_i+u_j), and equal add-up CE targets.
        u=model.log_variance(features).float().clamp(-5,5).exp().mean(1,keepdim=True)
        perm=torch.randperm(len(y),device=y.device); a=u/(u+u[perm]+1e-8)
        mixed=a*features+(1-a)*features[perm]
        mixed_logits=model.classifier(model.dropout(mixed.to(features.dtype)))
        add_up=.5*F.cross_entropy(mixed_logits,y)+.5*F.cross_entropy(mixed_logits,y[perm])
        return ce+.5*add_up
    return ce

def loader(dataset,batch,shuffle,workers,seed=42):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,
                      persistent_workers=workers>0,generator=generator,drop_last=shuffle)

@torch.inference_mode()
def predict(model,dl,device,tta=False):
    model.eval(); probabilities=[]; targets=[]
    for x,y in dl:
        x=x.to(device,non_blocking=True,memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            z,_=model(x); p=z.float().softmax(1)
            if tta:
                z,_=model(x.flip(-1)); p=(p+z.float().softmax(1))*.5
        probabilities.append(p.cpu().numpy()); targets.append(y.numpy())
    return np.concatenate(probabilities),np.concatenate(targets)

def metrics(p,y):
    pred=p.argmax(1)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
            'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-8,1)).mean()),
            'confusion_matrix':confusion_matrix(y,pred,labels=list(range(6))).tolist(),
            'classification_report':classification_report(y,pred,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0)}

def checkpoint(path,model,args,epoch,score):
    torch.save({'model':model.state_dict(),'config':vars(args),'epoch':epoch,'score':score,'label_mapping':LABELS},path)

def train(args):
    seed_everything(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('GPU training expected; CUDA unavailable')
    packed=np.load(ROOT/'data'/'dataset.npz'); x=packed['train_images']; y=packed['labels']
    tr=np.arange(len(y)) if args.full else packed['train_idx']; va=packed['val_idx']
    run=ROOT/'runs'/args.name; run.mkdir(parents=True,exist_ok=True)
    if (run/'history.jsonl').exists(): raise RuntimeError('Run already exists; use a new --name to avoid overwriting')
    dump(run/'config.json',vars(args)); dump(run/'split.json',json.loads((ROOT/'data'/'split.json').read_text()))
    model=FER(args.variant).to(device,memory_format=torch.channels_last)
    model_params=sum(p.numel() for p in model.parameters())
    backbone=list(model.backbone.parameters()); backbone_ids={id(p) for p in backbone}
    added=[p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':args.lr},{'params':added,'lr':args.lr*5}],weight_decay=1e-4)
    train_dl=loader(Faces(x[tr],y[tr],args.size,True),args.batch,True,args.workers,args.seed)
    val_dl=loader(Faces(x[va],y[va],args.size),args.batch*2,False,args.workers,args.seed) if not args.full else None
    print(json.dumps({'event':'started','variant':args.variant,'params':model_params,'train_samples':len(tr),
                      'val_samples':0 if args.full else len(va),'device':str(device),'seed':args.seed}),flush=True)
    best=-1.; best_epoch=0; started=time.time()
    if val_dl is not None:
        p,truth=predict(model,val_dl,device); initial=metrics(p,truth)
        dump(run/'initial_validation.json',initial)
        dump(run/'validation_best.json',initial)
        checkpoint(run/'best.pt',model,args,0,initial['accuracy']); best=initial['accuracy']
        np.savez(run/'validation_best.npz',probabilities=p,labels=truth,indices=va)
        print(json.dumps({'event':'initial_validation','accuracy':best}),flush=True)
    for epoch in range(1,args.epochs+1):
        epoch_start=time.time(); model.train(); total_loss=0.; correct=0; count=0
        # Warmup followed by cosine decay. Same schedule for every ablation.
        schedule_epochs=args.schedule_epochs or args.epochs
        ratio=min(1.,epoch/2)*(.05+.95*.5*(1+math.cos(math.pi*(epoch-1)/schedule_epochs)))
        optimizer.param_groups[0]['lr']=args.lr*ratio; optimizer.param_groups[1]['lr']=args.lr*5*ratio
        for step,(batch_x,batch_y) in enumerate(train_dl):
            batch_x=batch_x.to(device,non_blocking=True,memory_format=torch.channels_last); batch_y=batch_y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,features=model(batch_x); loss=objective(model,logits,features,batch_y,epoch)
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss at {epoch}/{step}')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            n=len(batch_y); count+=n; total_loss+=loss.item()*n; correct+=(logits.argmax(1)==batch_y).sum().item()
            if args.max_steps and step+1>=args.max_steps: break
        record={'epoch':epoch,'train_loss':total_loss/count,'train_accuracy':correct/count,'lr':args.lr*ratio}
        if val_dl is not None:
            p,truth=predict(model,val_dl,device); score=metrics(p,truth)
            record.update({'val_accuracy':score['accuracy'],'val_macro_f1':score['macro_f1'],'val_nll':score['nll']})
            if score['accuracy']>best:
                best=score['accuracy']; best_epoch=epoch
                checkpoint(run/'best.pt',model,args,epoch,best); dump(run/'validation_best.json',score)
                np.savez(run/'validation_best.npz',probabilities=p,labels=truth,indices=va)
        else:
            checkpoint(run/'best.pt',model,args,epoch,None); best_epoch=epoch
        checkpoint(run/'last.pt',model,args,epoch,record.get('val_accuracy'))
        record.update({'seconds':time.time()-epoch_start,'elapsed_seconds':time.time()-started,'best_accuracy':best,'best_epoch':best_epoch})
        with (run/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=True); model.load_state_dict(ckpt['model'])
    if val_dl is not None:
        p,truth=predict(model,val_dl,device,True)
        dump(run/'validation_tta.json',metrics(p,truth)); np.savez(run/'validation_tta.npz',probabilities=p,labels=truth,indices=va)
    test_dl=loader(Faces(packed['test_images'],None,args.size),args.batch*2,False,args.workers,args.seed)
    p,_=predict(model,test_dl,device,True)
    assert p.shape==(len(packed['test_ids']),6) and np.isfinite(p).all()
    np.savez(run/'test_probabilities.npz',probabilities=p,ids=packed['test_ids'])
    with (run/'submission.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(packed['test_ids'],p.argmax(1)))
    dump(run/'completed.json',{'best_epoch':best_epoch,'best_accuracy':None if args.full else best,
          'elapsed_seconds':time.time()-started,'submission_rows':len(p),'label_mapping':LABELS,'full_train':args.full})
    print(json.dumps({'event':'completed','run':str(run),'best_epoch':best_epoch,'best_accuracy':best}),flush=True)

def infer(args):
    seed_everything(42); packed=np.load(ROOT/'data'/'dataset.npz')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); all_probabilities=[]
    for path in args.checkpoint:
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['label_mapping']!=LABELS: raise ValueError('Checkpoint label mapping differs')
        cfg=ckpt['config']; model=FER(cfg['variant'],initialize_pretrained=False)
        model.load_state_dict(ckpt['model'],strict=True);model.to(device,memory_format=torch.channels_last)
        dl=loader(Faces(packed['test_images'],None,cfg['size']),args.batch,False,args.workers)
        p,_=predict(model,dl,device,tta=True); all_probabilities.append(p)
        del model
    p=np.mean(all_probabilities,axis=0)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(packed['test_ids'],p.argmax(1)))
    print(json.dumps({'output':str(output),'rows':len(p),'checkpoints':args.checkpoint}),flush=True)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--zip',default=r'D:\下载\2026task2.zip'); a.add_argument('--sample',default=r'D:\下载\submission.csv')
    a=sub.add_parser('train'); a.add_argument('--variant',choices=['baseline','local','scn','rul'],required=True)
    a.add_argument('--name',required=True); a.add_argument('--epochs',type=int,default=12); a.add_argument('--size',type=int,default=160)
    a.add_argument('--batch',type=int,default=64); a.add_argument('--workers',type=int,default=0); a.add_argument('--lr',type=float,default=1e-4)
    a.add_argument('--seed',type=int,choices=[42],default=42); a.add_argument('--full',action='store_true');a.add_argument('--max-steps',type=int,default=0)
    a.add_argument('--schedule-epochs',type=int,default=0,help='Preserve the validation learning-rate schedule in full-data refit')
    a=sub.add_parser('infer');a.add_argument('--checkpoint',nargs='+',required=True);a.add_argument('--output',required=True)
    a.add_argument('--batch',type=int,default=128);a.add_argument('--workers',type=int,default=0)
    args=p.parse_args()
    if args.command=='prepare': prepare(args.zip,args.sample)
    elif args.command=='train': train(args)
    else: infer(args)

if __name__=='__main__': main()
