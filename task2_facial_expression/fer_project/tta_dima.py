"""Unlabeled episodic LayerNorm adaptation on shuffled batches.

MEMO (NeurIPS 2022) marginal augmentation entropy, and EATA (ICML 2022)
inspired reliable-sample entropy filtering. Batch-wise, last-four-block LN
adaptations, NOT full reproductions of either paper. No labels enter adapt().
"""
import argparse,csv,gc,json,math,shutil,time
from pathlib import Path
from finetune_dima import ROOT,OUT,Model,load_delta,torch,nn,np,F,seed_everything,emit,dump
from upgrade_models import UpgradeImages
from upgrade_experiments import score

CONFIGS=[{'name':'reliable_entropy_ln','objective':'reliable_entropy','lr':1e-3,'steps':2,'batch':16},
         {'name':'memo_ln_small','objective':'memo','lr':1e-4,'steps':2,'batch':16},
         {'name':'memo_ln','objective':'memo','lr':1e-3,'steps':2,'batch':16}]

def configure(model):
    model.eval();model.requires_grad_(False);names=[];params=[]
    for name,module in model.named_modules():
        if not isinstance(module,nn.LayerNorm):continue
        last=name=='vit.layernorm' or (name.startswith('vit.encoder.layer.') and int(name.split('.')[3])>=8)
        if last:
            for key,p in module.named_parameters(recurse=False):
                p.requires_grad_(True);names.append(name+'.'+key);params.append(p)
    assert len(params)==18 and sum(p.numel() for p in params)==13824
    return params,names

def restore(params,snapshot):
    with torch.no_grad():
        for p,initial in zip(params,snapshot):p.copy_(initial)

def forward(model,x):
    with torch.autocast('cuda',dtype=torch.bfloat16):z,_=model(x)
    return z.float()

@torch.no_grad()
def prediction(model,x,tta):
    p=forward(model,x).softmax(-1)
    return (p+forward(model,x.flip(-1)).softmax(-1))/2 if tta=='flip' else p

def adapt(model,images,config,tta='single'):
    """Images/config only; label access is intentionally outside this function."""
    params,names=configure(model);snapshot=[p.detach().clone() for p in params]
    protected=model.head.weight.detach().clone()
    order=np.random.default_rng(42).permutation(len(images))
    dl=torch.utils.data.DataLoader(UpgradeImages(images[order],'hf_dima'),batch_size=config['batch'],num_workers=0,shuffle=False)
    original=np.zeros((len(images),6),np.float32);result=np.zeros_like(original)
    stats={'parameters':sum(p.numel() for p in params),'parameter_names':names,'eligible_samples_steps':0,
           'skipped_steps':0,'episodes':0,'max_parameter_change':0.,'shuffle_seed':42,'reset_each_batch':True}
    start=time.time();last=start
    try:
        for x,idx in dl:
            restore(params,snapshot);opt=torch.optim.Adam(params,lr=config['lr'])
            x=x.cuda();dest=order[idx.numpy()];base=prediction(model,x,tta);original[dest]=base.cpu().numpy()
            if config['objective']=='memo':
                crop=F.interpolate(x[:,:,4:-4,4:-4],size=(224,224),mode='bilinear',align_corners=False,antialias=True)
                # Mean over the four views of EACH image, never over different images.
                views=torch.stack([x,x.flip(-1),crop,(x*.96-.04).clamp(-1,1)],dim=1).flatten(0,1)
            for step in range(config['steps']):
                opt.zero_grad(set_to_none=True)
                if config['objective']=='memo':
                    p=forward(model,views).softmax(-1).reshape(len(x),4,6).mean(1)
                    entropy=-(p*p.clamp_min(1e-8).log()).sum(-1);loss=entropy.mean();eligible=len(x)
                else:
                    z=forward(model,x);p=z.softmax(-1);entropy=-(p*z.log_softmax(-1)).sum(-1)
                    mask=entropy.detach()<.4*math.log(6);eligible=int(mask.sum())
                    if not eligible:stats['skipped_steps']+=1;continue
                    loss=entropy[mask].mean()
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                stats['eligible_samples_steps']+=eligible
            result[dest]=prediction(model,x,tta).cpu().numpy();stats['episodes']+=1
            change=max(float((p-initial).abs().max()) for p,initial in zip(params,snapshot))
            stats['max_parameter_change']=max(change,stats['max_parameter_change'])
            del opt
            if time.time()-last>35:
                emit('tta_progress',name=config['name'],processed=min(stats['episodes']*config['batch'],len(images)),total=len(images),seconds=time.time()-start);last=time.time()
    finally:restore(params,snapshot)
    assert all(torch.equal(p,s) for p,s in zip(params,snapshot))
    assert torch.equal(protected,model.head.weight)
    assert np.isfinite(result).all() and np.allclose(result.sum(1),1,atol=1e-5)
    stats.update(seconds=time.time()-start,reset_verified=True,classification_head_unchanged=True)
    return result,original,stats

def load_model(full=False):
    r=json.loads((OUT/'dima_finetune_results.json').read_text());s=r['selected']
    assert s['mode'] in ['ce','supcon'],'This implementation requires the selected fine-tuned backbone'
    model=Model(s['mode']).cuda()
    path=r['final_checkpoint_paths'][-1] if full else s['checkpoint']
    ck=torch.load(path,map_location='cpu',weights_only=True);load_delta(model,ck['model'])
    return model,s

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'submission_dima_tta_reproduced.csv');args=parser.parse_args();seed_everything(42)
    data=np.load(ROOT/'data'/'dataset.npz')
    if args.smoke:
        model=Model('ce').cuda();ck=torch.load(ROOT/'runs'/'dima_ft_ce_holdout_s42'/'best.pt',map_location='cpu',weights_only=True);load_delta(model,ck['model'])
        # Training-side implementation check; no validation sample used for diagnostics.
        images=data['train_images'][data['train_idx'][:32]];cfg=CONFIGS[-1]
        p,b,stats=adapt(model,images,cfg);q,_,stats2=adapt(model,images,cfg)
        difference=float(np.abs(p-q).max());same_labels=bool(np.array_equal(p.argmax(1),q.argmax(1)))
        emit('tta_repeat_check',maximum_probability_difference=difference,same_labels=same_labels)
        assert np.allclose(p,q,atol=1e-5,rtol=1e-5) and same_labels,'Repeated adaptation exceeds numerical tolerance'
        assert stats['max_parameter_change']>0
        report={**stats,'repeat_labels_identical':same_labels,'repeat_max_probability_difference':difference,'training_only_smoke':True,'label_argument_absent':True}
        dump(OUT/'dima_tta_verification.json',report);emit('tta_smoke_passed',**report);return
    if args.reproduce:
        meta=json.loads((OUT/'dima_tta_results.json').read_text());assert meta['adopted']
        model,s=load_model(full=True);prob,_,stats=adapt(model,data['test_images'],meta['selected_config'],s['tta'])
        with args.output.open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(data['test_ids'],prob.argmax(1)))
        saved=np.load(OUT/'final_probabilities.npz')['probabilities'];report={'rows':len(prob),'same_labels':bool(np.array_equal(prob.argmax(1),saved.argmax(1))),
            'maximum_probability_difference':float(np.abs(prob-saved).max()),'output':str(args.output)}
        dump(OUT/'dima_inference_verification.json',report);emit('tta_reproduced',**report);return
    model,s=load_model();va=data['val_idx'];y=data['labels'][va];images=data['train_images'][va]
    baseline=np.load(OUT/'dima_finetune_validation.npz')['probabilities'];winner=score(baseline,y);best_p=baseline;best_config=None;records=[]
    for cfg in CONFIGS:
        prob,original,stats=adapt(model,images,cfg,s['tta']);metrics=score(prob,y)
        row={'config':cfg,'metrics':metrics,'unadapted_same_batch_metrics':score(original,y),'stats':stats};records.append(row)
        np.savez(ROOT/'runs'/f"dima_tta_{cfg['name']}_validation.npz",probabilities=prob,indices=va,labels=y)
        emit('tta_validation',**row)
        if metrics['accuracy']>=winner['accuracy']+.0025:winner=metrics;best_p=prob;best_config=cfg
    del model;gc.collect();torch.cuda.empty_cache()
    report={'completed':True,'baseline':score(baseline,y),'candidates':records,'adopted':best_config is not None,
            'selected_config':best_config,'selected_metrics':winner,'minimum_gain':.0025,'source_image_overlap_possible':True,
            'label_usage':'No labels passed into adaptation. Validation labels used only for configuration selection; test labels unavailable.',
            'method_scope':'Episodic shuffled batches; last four ViT blocks LayerNorm only. Inspired by MEMO/EATA, not full paper reproductions.'}
    if best_config:
        model,s=load_model(full=True);prob,_,stats=adapt(model,data['test_images'],best_config,s['tta']);report['test_adaptation_stats']=stats
        for name in ['submission.csv','final_probabilities.npz']:
            if not (OUT/('before_dima_tta_'+name)).exists():shutil.copy2(OUT/name,OUT/('before_dima_tta_'+name))
        with (OUT/'submission.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(data['test_ids'],prob.argmax(1)))
        np.savez(OUT/'final_probabilities.npz',ids=data['test_ids'],probabilities=prob)
    np.savez(OUT/'dima_final_validation.npz',indices=va,labels=y,probabilities=best_p)
    with (OUT/'dima_validation_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['image','given_label','prediction','confidence'])
        for i in np.flatnonzero(best_p.argmax(1)!=y):w.writerow([data['train_ids'][va[i]],int(y[i]),int(best_p[i].argmax()),float(best_p[i].max())])
    from sklearn.metrics import confusion_matrix
    dump(OUT/'dima_confusion_matrix.json',{'classes':['Angry','Fear','Happy','Sad','Surprise','Neutral'],'matrix':confusion_matrix(y,best_p.argmax(1)).tolist()})
    dump(OUT/'dima_tta_results.json',report)
    selection=json.loads((OUT/'model_selection.json').read_text());selection['dima_tta']=report
    selection['final_validation_accuracy']=winner['accuracy'];selection['final_validation_macro_f1']=winner['macro_f1'];dump(OUT/'model_selection.json',selection)
    emit('tta_complete',**report)

if __name__=='__main__':main()
