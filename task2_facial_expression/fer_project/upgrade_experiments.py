"""Frozen author representations -> six-class heads and feature adapters.

One seed and the existing grouped holdout. Source-image overlap may exist for
FERPlus/FER2013 pretrained models; scores are NOT independent generalization.
All caches/checkpoints stay on D:. No external labels or test-label matching.
"""
import argparse, copy, csv, gc, hashlib, json, math, shutil, time
from pathlib import Path
from fer import ROOT, torch, nn, np, F, seed_everything, dump, CLASSES
from sklearn.metrics import f1_score, confusion_matrix
from upgrade_models import UpgradeFER, UpgradeImages, SPECS

HERE=Path(__file__).resolve().parent; OUT=HERE.parent; MIN_GAIN=.0025


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def score(p,y):
    return {'accuracy':float((p.argmax(1)==y).mean()),
            'macro_f1':float(f1_score(y,p.argmax(1),average='macro')),
            'nll':float(-np.log(p[np.arange(len(y)),y].clip(1e-8)).mean())}


def emit(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)


@torch.inference_mode()
def extract(kind,packed,batch=16):
    source=ROOT/'pretrained'/SPECS[kind]['filename']
    identity={'checkpoint_sha256':sha(source),'split_sha256':sha(ROOT/'data'/'split.json'),
              'spec':SPECS[kind],'implementation_sha256':sha(HERE/'upgrade_models.py'),
              'views':['original','horizontal_flip'],'dtype':'float16_features_float32_probabilities'}
    if kind.startswith('hf_'):identity['processor_sha256']=sha(source.parent/'preprocessor_config.json')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    cache=ROOT/'upgrade_cache'/f'{kind}_{key}';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'complete.json').exists():
        emit('reuse_features',kind=kind,cache=str(cache));return cache
    model=UpgradeFER(kind).eval().to('cuda')
    dump(cache/'load_report.json',model.load_report)
    np.savez(cache/'head.npz',weight=model.initial_weight.numpy(),bias=model.initial_bias.numpy())
    for part,images in [('train',packed['train_images']),('test',packed['test_images'])]:
        features=np.lib.format.open_memmap(cache/f'{part}_features.npy',mode='w+',dtype=np.float16,shape=(len(images),2,model.dim))
        probs=np.lib.format.open_memmap(cache/f'{part}_probs.npy',mode='w+',dtype=np.float32,shape=(len(images),2,6))
        image_order=np.concatenate([packed['val_idx'],packed['train_idx']]) if part=='train' else np.arange(len(images))
        loader=torch.utils.data.DataLoader(UpgradeImages(images[image_order],kind),batch_size=batch,num_workers=0,shuffle=False)
        start=time.time();last=start
        screen_reported=False
        for x,index in loader:
            x=x.cuda()
            for view in range(2):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z,f=model(x if view==0 else x.flip(-1))
                dest=image_order[index.numpy()]
                features[dest,view]=f.float().cpu().numpy().astype(np.float16)
                probs[dest,view]=z.float().softmax(1).cpu().numpy()
            if part=='train' and not screen_reported and int(index[-1])+1>=len(packed['val_idx']):
                va=packed['val_idx'];v=np.asarray(probs[va]);target=packed['labels'][va]
                report={'kind':kind,'single':score(v[:,0],target),'flip':score(v.mean(1),target),
                        'note':'Frozen pretrained checkpoint on fixed holdout; source overlap possible; not Kaggle score'}
                dump(OUT/f'upgrade_{kind}_screen.json',report);emit('pretrained_screen',**report);screen_reported=True
            if time.time()-last>=30:
                emit('feature_progress',kind=kind,part=part,done=int(index[-1])+1,total=len(images),seconds=time.time()-start);last=time.time()
        features.flush();probs.flush();del features,probs
    del model;gc.collect();torch.cuda.empty_cache()
    dump(cache/'complete.json',identity);return cache


class FeatureHead(nn.Module):
    """Identity-initialized residual adapter; pretrained linear head is retained."""
    def __init__(self,weight,bias,mode):
        super().__init__();dim=weight.shape[1]
        self.adapter=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),nn.Linear(128,dim)) if mode!='linear' else None
        if self.adapter is not None:
            nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        self.fc=nn.Linear(dim,6)
        with torch.no_grad():self.fc.weight.copy_(weight);self.fc.bias.copy_(bias)
    def forward(self,x):
        f=x if self.adapter is None else x+.1*self.adapter(x)
        return self.fc(f),f


def supcon(features,labels,reliability,temperature=.1):
    """Two views per image; confidence-weighted anchors, no label replacement."""
    batch,views,dim=features.shape
    z=F.normalize(features.flatten(0,1),dim=1)
    labels=labels.repeat_interleave(views);weights=reliability.repeat_interleave(views)
    similarity=z@z.T/temperature
    eye=torch.eye(batch*views,device=z.device,dtype=torch.bool)
    logp=similarity-similarity.masked_fill(eye,-torch.inf).logsumexp(1,keepdim=True)
    positive=labels[:,None].eq(labels[None,:])&~eye
    loss=-(logp.masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1))
    return (loss*weights).sum()/weights.sum().clamp_min(1e-8)


def probability_js(logits):
    p=logits.softmax(-1);m=p.mean(1,keepdim=True)
    return (p*(p.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(-1).mean()


@torch.inference_mode()
def predict_head(model,features,indices,device='cuda'):
    model.eval();result=[]
    for begin in range(0,len(indices),512):
        x=torch.from_numpy(np.array(features[indices[begin:begin+512]],dtype=np.float32)).to(device)
        z,_=model(x);result.append(z.softmax(-1).mean(1).cpu().numpy())
    return np.concatenate(result)


def train_head(kind,mode,cache,packed,epochs=20,full=False,chosen_epoch=None,device='cuda'):
    seed_everything(42)
    features=np.load(cache/'train_features.npy',mmap_mode='r')
    teacher=np.load(cache/'train_probs.npy',mmap_mode='r')
    head=np.load(cache/'head.npz');weight=torch.from_numpy(head['weight']);bias=torch.from_numpy(head['bias'])
    model=FeatureHead(weight,bias,mode).to(device)
    optimizer=torch.optim.AdamW([{'params':model.fc.parameters(),'lr':1e-4}]+(
        [{'params':model.adapter.parameters(),'lr':1e-3}] if model.adapter else []),weight_decay=1e-3)
    y=packed['labels'];tr=np.arange(len(y)) if full else packed['train_idx'];va=packed['val_idx']
    name=f'upgrade_{kind}_{mode}_{"full" if full else "holdout"}_s42'
    run=ROOT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
    config={'kind':kind,'mode':mode,'seed':42,'epochs':epochs,'full':full,'source_cache':str(cache),
            'batch':256,'head_lr':1e-4,'adapter_lr':1e-3,'weight_decay':1e-3,
            'supcon_weight':.05 if mode=='supcon' else 0.,'standard_probability_js_weight':.1 if mode=='supcon' else 0.,
            'label_smoothing':.02,'feature_backbone_frozen':True,'source_image_overlap_possible':kind!='nla'}
    dump(run/'config.json',config)
    best_p=predict_head(model,features,va,device) if not full else None
    best_acc=score(best_p,y[va])['accuracy'] if not full else -1
    best_epoch=0;best_state=copy.deepcopy(model.state_dict());hist=[]
    count=chosen_epoch if full else epochs
    for epoch in range(1,count+1):
        model.train();order=tr[np.random.permutation(len(tr))];loss_sum=0.;seen=0
        for group in optimizer.param_groups:
            base=1e-4 if group is optimizer.param_groups[0] else 1e-3
            group['lr']=base*(.05+.95*(1+math.cos(math.pi*(epoch-1)/epochs))/2)
        for begin in range(0,len(order),256):
            index=order[begin:begin+256]
            x=torch.from_numpy(np.array(features[index],dtype=np.float32)).to(device)
            target=torch.from_numpy(y[index]).to(device)
            z,f=model(x);loss=F.cross_entropy(z.flatten(0,1),target.repeat_interleave(2),label_smoothing=.02)
            if mode=='supcon':
                # Hard examples retain full CE; reliability only attenuates contrastive anchors.
                tp=np.asarray(teacher[index]).mean(1)
                reliability=torch.from_numpy(.25+.75*tp[np.arange(len(index)),y[index]]).to(device)
                loss=loss+.1*probability_js(z)
                if epoch>2:loss=loss+.05*supcon(f,target,reliability)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            loss_sum+=loss.item()*len(index);seen+=len(index)
        row={'epoch':epoch,'loss':loss_sum/seen}
        if not full:
            p=predict_head(model,features,va,device);row.update(score(p,y[va]))
            if row['accuracy']>best_acc:
                best_acc=row['accuracy'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());best_p=p
        hist.append(row);emit('feature_fit',kind=kind,mode=mode,full=full,**row)
    if full:best_state=model.state_dict();best_epoch=count
    model.load_state_dict(best_state)
    checkpoint=run/'best.pt';torch.save({'model':best_state,'config':config,'best_epoch':best_epoch},checkpoint)
    dump(run/'history.json',hist)
    if full:
        test_features=np.load(cache/'test_features.npy',mmap_mode='r')
        best_p=predict_head(model,test_features,np.arange(len(test_features)),device)
        np.savez(run/'test_probabilities.npz',ids=packed['test_ids'],probabilities=best_p)
    else:np.savez(run/'validation_probabilities.npz',indices=va,labels=y[va],probabilities=best_p)
    dump(run/'completed.json',{'best_epoch':best_epoch,'checkpoint':str(checkpoint),'full':full})
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return best_p,best_epoch,str(checkpoint)


def smoke(kind):
    seed_everything(42);packed=np.load(ROOT/'data'/'dataset.npz')
    idx=np.concatenate([np.flatnonzero(packed['labels'][packed['train_idx']]==c)[:2] for c in range(6)])
    indices=packed['train_idx'][idx]
    model=UpgradeFER(kind).eval();data=UpgradeImages(packed['train_images'][indices],kind)
    # CPU check can run while the separate GPU queue is busy.
    with torch.inference_mode():
        zs=[];fs=[]
        for start in range(0,len(data),2):
            x=torch.stack([data[i][0] for i in range(start,min(start+2,len(data)))])
            z,f=model(x);zs.append(z);fs.append(f)
        z=torch.cat(zs);f=torch.cat(fs)
        adapter=FeatureHead(model.initial_weight,model.initial_bias,'supcon').eval()
        logits,adapted=adapter(f)
        assert torch.allclose(z,logits,atol=2e-5), 'Head mapping or feature extraction mismatch'
        assert torch.isfinite(logits).all() and z.shape==(12,6)
    paired=torch.randn(8,2,model.dim,requires_grad=True);labels=torch.arange(8)%6
    loss=supcon(paired,labels,torch.ones(8));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(paired.grad).all()
    report={**model.load_report,'original_head_matches_identity_adapter':True,'supcon_gradients_finite':True,
            'training_only_sample_indices':indices.tolist(),'sample_predictions':z.argmax(1).tolist(),
            'sample_labels':packed['labels'][indices].tolist(),'note':'12-image implementation smoke check, not an accuracy estimate'}
    dump(OUT/f'upgrade_{kind}_verification.json',report);emit('smoke_passed',**report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kinds',nargs='+',default=['nla','qcs','rmn'])
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--wait-weights-seconds',type=int,default=0)
    parser.add_argument('--smoke',choices=list(SPECS));args=parser.parse_args()
    if args.smoke:return smoke(args.smoke)
    seed_everything(42);packed=np.load(ROOT/'data'/'dataset.npz');va=packed['val_idx'];y=packed['labels'][va]
    results=[];candidates=[];failures=[]
    pending=list(args.kinds)
    while pending:
        ready=[k for k in pending if (ROOT/'pretrained'/SPECS[k]['filename']).exists()]
        kind=ready[0] if ready else pending[0];pending.remove(kind)
        try:
            deadline=time.monotonic()+args.wait_weights_seconds
            source=ROOT/'pretrained'/SPECS[kind]['filename']
            while not source.exists() and time.monotonic()<deadline:
                # This manifest is written only after the original concurrent
                # downloader exits its pool, so retry cannot race its writers.
                if not kind.startswith('hf_') and (OUT/'upgrade_pretrained_manifest.json').exists():
                    from prepare_upgrade_weights import ITEMS, download
                    item=next(item for item in ITEMS if item['name']==source.name)
                    download(item)
                    break
                emit('waiting_for_weight_download',kind=kind,remaining_seconds=round(deadline-time.monotonic()))
                time.sleep(30)
            smoke(kind)
            cache=extract(kind,packed,args.batch)
            native=np.load(cache/'train_probs.npy',mmap_mode='r')[va]
            single,flip=native[:,0],native.mean(1)
            view='flip' if score(flip,y)['accuracy']>=score(single,y)['accuracy']+MIN_GAIN else 'single'
            chosen_p=flip if view=='flip' else single
            winner={'kind':kind,'mode':'native','tta':view,'epoch':0,'cache':str(cache),**score(chosen_p,y)}
            results.extend([{'kind':kind,'mode':'native','tta':v,'epoch':0,**score(p,y)} for v,p in [('single',single),('flip',flip)]])
            for mode in ['linear','adapter_ce','supcon']:
                p,epoch,checkpoint=train_head(kind,mode,cache,packed,args.epochs)
                row={'kind':kind,'mode':mode,'tta':'flip','epoch':epoch,'cache':str(cache),'checkpoint':checkpoint,**score(p,y)}
                results.append(row)
                if row['accuracy']>=winner['accuracy']+MIN_GAIN:winner=row;chosen_p=p
            candidates.append({'record':winner,'probabilities':chosen_p})
            emit('model_selected',**winner)
            dump(OUT/'upgrade_results.json',{'results':results,'failures':failures,'completed':False})
        except Exception as error:
            import traceback;traceback.print_exc()
            failures.append({'kind':kind,'error':repr(error)})
            dump(OUT/'upgrade_results.json',{'results':results,'failures':failures,'completed':False})
    if not candidates:raise RuntimeError('No upgrade model completed; preserve the core submission')
    ranked=sorted(candidates,key=lambda x:x['record']['accuracy'],reverse=True)
    chosen=ranked[:1];val=chosen[0]['probabilities'];fusion=None
    if len(ranked)>1:
        mean=(ranked[0]['probabilities']+ranked[1]['probabilities'])/2
        fusion=score(mean,y)
        if fusion['accuracy']>=score(val,y)['accuracy']+MIN_GAIN:chosen=ranked[:2];val=mean
    selected_score=score(val,y)
    result={'results':results,'failures':failures,'selected':[r['record'] for r in chosen],
            'selected_metrics':selected_score,'top2_equal_fusion_metrics':fusion,'seed':42,'minimum_gain':MIN_GAIN,
            'source_image_overlap_possible':any(r['record']['kind']!='nla' for r in chosen),
            'caveat':'External pretrained source images may overlap local holdout/test. This is not an independent generalization or Kaggle score.',
            'completed':True}
    core=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8')) if (OUT/'model_selection.json').exists() else None
    core_default=(core['top2_ensemble_validation_accuracy'] if len(core['selected_variants'])>1 else max(r['accuracy'] for r in core['results'])) if core else -1
    core_acc=core.get('final_validation_accuracy',core_default) if core else -1
    result['adopted_over_core']=selected_score['accuracy']>=core_acc+MIN_GAIN
    if result['adopted_over_core']:
        test=[];artifacts=[]
        for entry in chosen:
            r=entry['record'];cache=Path(r['cache'])
            artifacts.append(str(ROOT/'pretrained'/SPECS[r['kind']]['filename']))
            if r['mode']=='native':
                p=np.load(cache/'test_probs.npy',mmap_mode='r');p=p.mean(1) if r['tta']=='flip' else p[:,0]
            else:
                p,_,checkpoint=train_head(r['kind'],r['mode'],cache,packed,args.epochs,True,r['epoch'])
                artifacts.append(checkpoint)
            test.append(p)
        final=np.mean(test,axis=0);assert final.shape==(7065,6) and np.isfinite(final).all()
        assert np.allclose(final.sum(1),1,atol=1e-5)
        for name in ['submission.csv','final_probabilities.npz','model_selection.json']:
            if (OUT/name).exists() and not (OUT/('before_upgrade_'+name)).exists():shutil.copy2(OUT/name,OUT/('before_upgrade_'+name))
        with (OUT/'submission.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(packed['test_ids'],final.argmax(1)))
        np.savez(OUT/'final_probabilities.npz',ids=packed['test_ids'],probabilities=final)
        np.savez(OUT/'upgrade_validation_probabilities.npz',indices=va,labels=y,probabilities=val)
        result['final_checkpoint_paths']=artifacts
        if core:
            core['upgrade']=result;core['final_validation_accuracy']=selected_score['accuracy'];core['final_validation_macro_f1']=selected_score['macro_f1']
            dump(OUT/'model_selection.json',core)
    dump(OUT/'upgrade_results.json',result)
    lines=['# 预训练与特征适配实测结果','',result['caveat'],'','|模型|适配|TTA|轮数|验证准确率|Macro-F1|','|---|---|---|---:|---:|---:|']
    for r in results:lines.append(f"|{r['kind']}|{r['mode']}|{r['tta']}|{r['epoch']}|{r['accuracy']:.2%}|{r['macro_f1']:.2%}|")
    lines.extend(['',f"是否替换原基线提交：{result['adopted_over_core']}。所选升级模型验证准确率：{selected_score['accuracy']:.2%}。",'',
                  '此阶段冻结图像骨干，训练分类头/特征适配器；并非已经完成端到端微调或完整复现 QCS/NLA。',
                  '原有 error_analysis 属于 B0 阶段；升级模型的错误表另见 upgrade_validation_errors.csv。'])
    if failures:lines.extend(['','未完成的模型：'+json.dumps(failures,ensure_ascii=False)])
    (OUT/'预训练升级结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'upgrade_validation_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['image','given_label','prediction','confidence'])
        for i in np.flatnonzero(val.argmax(1)!=y):w.writerow([packed['train_ids'][va[i]],int(y[i]),int(val[i].argmax()),float(val[i].max())])
    dump(OUT/'upgrade_confusion_matrix.json',{'classes':CLASSES,'matrix':confusion_matrix(y,val.argmax(1),labels=range(6)).tolist()})
    emit('upgrade_completed',**result)


if __name__=='__main__':main()
