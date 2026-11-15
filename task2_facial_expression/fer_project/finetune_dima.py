"""Dima ViT last-four-block fine-tuning: paired CE vs confidence-aware SupCon.

Seed 42, existing grouped holdout; no test labels. External pretraining overlap
is acknowledged. This is an adaptation of SupCon and ambiguity/consistency ideas,
not a reproduction of NLA. Weights and resumable state are stored only on D:.
"""
import argparse, copy, csv, gc, hashlib, json, math, time
from pathlib import Path
from fer import ROOT, torch, nn, F, np, seed_everything, dump, CLASSES
from upgrade_models import UpgradeFER, UpgradeImages
from upgrade_experiments import score, supcon, probability_js, train_head

OUT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'upgrade_cache'/'hf_dima_419cba1016473c0d'
CFG={'seed':42,'epochs':5,'patience':3,'batch_images':32,'accumulation':1,
     'unfrozen_blocks':4,'backbone_lr':3e-6,'head_lr':3e-5,'projection_lr':1e-4,
     'weight_decay':.01,'label_smoothing':.02,'supcon_weight':.03,
     'js_weight':.1,'temperature':.1,'source_image_overlap_possible':True,
     'minimum_gain':.0025,'augmentation':'independent views: +/-8deg, scale .94-1.06, translation +/- .04, flip, brightness .9-1.1',
     'pretrained_revision':'747cf16692eea925b54b7b543cf436848128b68d'}

def emit(event,**kw):
    print(json.dumps({'event':event,**kw}),flush=True)

class Model(nn.Module):
    def __init__(self,mode):
        super().__init__()
        source=UpgradeFER('hf_dima')
        self.vit=source.source_model.vit
        self.head=nn.Linear(768,6)
        with torch.no_grad():
            self.head.weight.copy_(source.initial_weight);self.head.bias.copy_(source.initial_bias)
        for p in self.vit.parameters():p.requires_grad_(False)
        for layer in self.vit.encoder.layer[-CFG['unfrozen_blocks']:]:
            for p in layer.parameters():p.requires_grad_(True)
        for p in self.vit.layernorm.parameters():p.requires_grad_(True)
        self.projector=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,128)) if mode=='supcon' else None
    def forward(self,x):
        f=self.vit(x).last_hidden_state[:,0]
        return self.head(f), self.projector(f) if self.projector is not None else f

def delta(model):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in names}

def load_delta(model,state):
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(state)==expected,(len(state),len(expected))
    model.load_state_dict(state,strict=False)

def optimizer_for(model):
    groups=[]
    for i,layer in enumerate(model.vit.encoder.layer[-CFG['unfrozen_blocks']:]):
        groups.append({'params':list(layer.parameters()),'base_lr':CFG['backbone_lr']*(.8**(3-i))})
    groups.extend([{'params':list(model.vit.layernorm.parameters()),'base_lr':CFG['backbone_lr']},
                   {'params':list(model.head.parameters()),'base_lr':CFG['head_lr']}])
    if model.projector is not None:groups.append({'params':list(model.projector.parameters()),'base_lr':CFG['projection_lr']})
    return torch.optim.AdamW(groups,lr=CFG['backbone_lr'],weight_decay=CFG['weight_decay'])

def augment(x):
    """Per-image geometry on the GPU, operating on 48px gray images before resize."""
    n=len(x);x=x.float().unsqueeze(1)/255
    angle=(torch.rand(n,device=x.device)*2-1)*(8*math.pi/180)
    scale=.94+.12*torch.rand(n,device=x.device)
    flip=torch.where(torch.rand(n,device=x.device)<.5,-1.,1.)
    a=torch.zeros(n,2,3,device=x.device)
    a[:,0,0]=angle.cos()*scale*flip;a[:,0,1]=-angle.sin()*scale
    a[:,1,0]=angle.sin()*scale*flip;a[:,1,1]=angle.cos()*scale
    a[:,:,2]=(torch.rand(n,2,device=x.device)*2-1)*.04
    grid=F.affine_grid(a,x.shape,align_corners=False)
    x=F.grid_sample(x,grid,align_corners=False,padding_mode='border')
    x=(x*(.9+.2*torch.rand(n,1,1,1,device=x.device))).clamp(0,1)
    x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
    return ((x-.5)/.5).expand(-1,3,-1,-1).contiguous()

@torch.inference_mode()
def predict(model,images,batch=32):
    model.eval();result=[]
    dl=torch.utils.data.DataLoader(UpgradeImages(images,'hf_dima'),batch_size=batch,shuffle=False,num_workers=0)
    for x,_ in dl:
        x=x.cuda();views=[]
        for v in range(2):
            with torch.autocast('cuda',dtype=torch.bfloat16):z,_=model(x if not v else x.flip(-1))
            views.append(z.float().softmax(-1).cpu().numpy())
        result.append(np.stack(views,axis=1))
    return np.concatenate(result)

def best_view(prob,y):
    a=score(prob[:,0],y);b=score(prob.mean(1),y)
    view='flip' if b['accuracy']>=a['accuracy']+CFG['minimum_gain'] else 'single'
    return {'tta':view,**(b if view=='flip' else a)}, {'single':a,'flip':b}

def train(mode,packed,full=False,chosen_epoch=None,smoke=False):
    seed_everything(42)
    run=ROOT/'runs'/f'dima_ft_{mode}_{"full" if full else "holdout"}_s42'
    if smoke:run=ROOT/'runs'/'dima_ft_smoke'
    run.mkdir(parents=True,exist_ok=True)
    cfg={**CFG,'mode':mode,'full':full,'chosen_epoch':chosen_epoch,'smoke':smoke}
    if (run/'completed.json').exists() and not smoke:
        assert json.loads((run/'config.json').read_text())==cfg,'Refuse reuse of changed experiment'
        return json.loads((run/'completed.json').read_text())
    dump(run/'config.json',cfg)
    model=Model(mode).cuda();opt=optimizer_for(model)
    tr=np.arange(len(packed['labels'])) if full else packed['train_idx'];va=packed['val_idx']
    assert full or not set(tr)&set(va)
    if smoke:tr=tr[:32]
    raw=torch.from_numpy(np.array(packed['train_images'])).cuda()
    labels=torch.from_numpy(np.array(packed['labels'])).cuda()
    teacher=np.load(CACHE/'train_probs.npy',mmap_mode='r').mean(1)
    given=teacher[np.arange(len(labels)),packed['labels']]
    other=teacher.copy();other[np.arange(len(labels)),packed['labels']]=-1
    # Bounded reliability from true-vs-competing teacher probability. Only training indices are gathered.
    reliable=torch.from_numpy((.25+.75*np.clip((given-other.max(1)+1)/2,0,1)).astype(np.float32)).cuda()
    epochs=chosen_epoch if full else CFG['epochs'];start=1;history=[]
    best={'mode':mode,'epoch':0,'tta':'single','accuracy':.8197838345864662,
          'macro_f1':.8125589143210248,'nll':.5258791446685791}
    if not full and not smoke and not (run/'last.pt').exists():
        native=np.load(CACHE/'train_probs.npy',mmap_mode='r')[va]
        b,_=best_view(native,packed['labels'][va]);best.update(b)
        np.savez(run/'validation_best.npz',probabilities=native,indices=va,labels=packed['labels'][va])
    if not (run/'initial.pt').exists():torch.save({'model':delta(model),'config':cfg,'selected':best},run/'initial.pt')
    if (run/'last.pt').exists() and not smoke:
        ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert ck['config']==cfg
        load_delta(model,ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1;best=ck['best'];history=ck['history']
        np.random.set_state(ck['np_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        emit('resumed',mode=mode,full=full,epoch=start)
    elif not smoke:torch.save({'model':delta(model),'config':cfg,'selected':best},run/'best.pt')
    frozen_before=model.vit.embeddings.patch_embeddings.projection.weight.detach().clone() if smoke else None
    train_before=model.vit.encoder.layer[-1].output.dense.weight.detach().clone() if smoke else None
    emit('finetune_start',mode=mode,full=full,samples=len(tr),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),config=cfg)
    total_steps=math.ceil(len(tr)/CFG['batch_images']);clock=time.time()
    for epoch in range(start,epochs+1):
        model.train();order=tr[np.random.permutation(len(tr))];opt.zero_grad(set_to_none=True)
        loss_sum=0.;ce_sum=0.;sc_sum=0.;js_sum=0.;seen=0;last=time.time();epstart=last
        for step,begin in enumerate(range(0,len(order),CFG['batch_images'])):
            ids=torch.as_tensor(order[begin:begin+CFG['batch_images']],device='cuda');n=len(ids)
            x=augment(raw[ids].repeat_interleave(2,dim=0));y=labels[ids]
            with torch.autocast('cuda',dtype=torch.bfloat16):z,f=model(x)
            z=z.float().reshape(n,2,6);f=f.float().reshape(n,2,-1)
            ce=F.cross_entropy(z.flatten(0,1),y.repeat_interleave(2),label_smoothing=CFG['label_smoothing'])
            sc=ce.new_zeros(());js=ce.new_zeros(())
            if mode=='supcon':
                sc=supcon(f,y,reliable[ids],CFG['temperature']);js=probability_js(z)
            loss=ce+CFG['supcon_weight']*min(epoch/2,1)*sc+CFG['js_weight']*js
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            window_start=(step//CFG['accumulation'])*CFG['accumulation']
            window=min(CFG['accumulation'],total_steps-window_start)
            (loss/window).backward()
            if (step+1)%CFG['accumulation']==0 or step+1==total_steps:
                progress=(epoch-1+step/total_steps)/CFG['epochs']
                warm=min(1.,(epoch-1+step/total_steps+.02)/.25)
                factor=warm*(.1+.9*(1+math.cos(math.pi*progress))/2)
                for g in opt.param_groups:g['lr']=g['base_lr']*factor
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
            loss_sum+=loss.item()*n;ce_sum+=ce.item()*n;sc_sum+=sc.item()*n;js_sum+=js.item()*n;seen+=n
            if time.time()-last>35:
                emit('finetune_progress',mode=mode,full=full,epoch=epoch,images=seen,total=len(tr),loss=loss_sum/seen,seconds=time.time()-epstart);last=time.time()
            if smoke and step>=1:break
        row={'epoch':epoch,'loss':loss_sum/seen,'ce':ce_sum/seen,'supcon':sc_sum/seen,'js':js_sum/seen,'seconds':time.time()-epstart}
        if not full and not smoke:
            p=predict(model,packed['train_images'][va]);selected,views=best_view(p,packed['labels'][va]);row.update(selected);row['views']=views
            if selected['accuracy']>best['accuracy']:
                best={'mode':mode,'epoch':epoch,**selected}
                torch.save({'model':delta(model),'config':cfg,'selected':best},run/'best.pt')
                np.savez(run/'validation_best.npz',probabilities=p,indices=va,labels=packed['labels'][va])
        history.append(row);dump(run/'history.json',history);emit('finetune_epoch',mode=mode,full=full,**row,best=best)
        if smoke:
            assert torch.equal(frozen_before,model.vit.embeddings.patch_embeddings.projection.weight)
            assert not torch.equal(train_before,model.vit.encoder.layer[-1].output.dense.weight)
            assert model.vit.encoder.layer[-1].output.dense.weight.requires_grad
            state=delta(model);original=model.head.weight.detach().clone()
            with torch.no_grad():model.head.weight.zero_()
            load_delta(model,state);assert torch.equal(original,model.head.weight)
            report={'frozen_embedding_unchanged':True,'last_transformer_block_updated':True,'delta_reload_exact':True,'loss_finite':True,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'seconds':time.time()-clock}
            dump(OUT/'dima_finetune_verification.json',report);emit('smoke_passed',**report)
            return report
        torch.save({'model':delta(model),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'history':history,'config':cfg,
                    'np_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},run/'last.pt')
        if not full and epoch-best['epoch']>=CFG['patience']:
            emit('early_stop',mode=mode,epoch=epoch,best=best);break
    if full:
        best={'mode':mode,'epoch':chosen_epoch}
        torch.save({'model':delta(model),'config':cfg,'selected':best},run/'best.pt')
        p=predict(model,packed['test_images']);np.savez(run/'test_probabilities.npz',probabilities=p,ids=packed['test_ids'])
    result={**best,'checkpoint':str(run/'best.pt'),'run':str(run),'full':full,'elapsed_seconds':time.time()-clock,'completed':True}
    dump(run/'completed.json',result)
    del model,opt,raw,labels,reliable;gc.collect();torch.cuda.empty_cache()
    return result

def finalize(packed,records):
    va=packed['val_idx'];y=packed['labels'][va]
    old=np.load(ROOT/'runs'/'upgrade_hf_dima_linear_holdout_s42'/'validation_probabilities.npz')['probabilities']
    winner={'mode':'linear','epoch':13,'tta':'flip',**score(old,y)};val=old
    for r in records:
        if r['accuracy']>=winner['accuracy']+CFG['minimum_gain']:
            winner=r;p=np.load(Path(r['run'])/'validation_best.npz')['probabilities'];val=p.mean(1) if r['tta']=='flip' else p[:,0]
    emit('finetune_selected',selected=winner)
    if winner['mode']=='linear':
        final,_,path=train_head('hf_dima','linear',CACHE,packed,20,True,13)
        checkpoints=[str(ROOT/'pretrained'/'hf_dima'/'model.safetensors'),path]
    else:
        r=train(winner['mode'],packed,full=True,chosen_epoch=winner['epoch'])
        p=np.load(Path(r['run'])/'test_probabilities.npz')['probabilities'];final=p.mean(1) if winner['tta']=='flip' else p[:,0]
        checkpoints=[str(ROOT/'pretrained'/'hf_dima'/'model.safetensors'),r['checkpoint']]
    assert final.shape==(len(packed['test_ids']),6) and np.isfinite(final).all() and np.allclose(final.sum(1),1,atol=1e-5)
    with (OUT/'submission.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(packed['test_ids'],final.argmax(1)))
    np.savez(OUT/'final_probabilities.npz',ids=packed['test_ids'],probabilities=final)
    np.savez(OUT/'dima_finetune_validation.npz',indices=va,labels=y,probabilities=val)
    result={'completed':True,'config':CFG,'experiments':records,'selected':winner,'selected_metrics':score(val,y),
            'final_checkpoint_paths':checkpoints,'full_refit':True,'source_image_overlap_possible':True,
            'note':'Local validation, not Kaggle or independent generalization; source overlap accepted by user.'}
    dump(OUT/'dima_finetune_results.json',result)
    core=json.loads((OUT/'model_selection.json').read_text(encoding='utf8'))
    core['dima_finetune']=result;core['final_validation_accuracy']=result['selected_metrics']['accuracy'];core['final_validation_macro_f1']=result['selected_metrics']['macro_f1'];core['full_refit']=True
    dump(OUT/'model_selection.json',core)
    with (OUT/'dima_validation_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['image','given_label','prediction','confidence'])
        for i in np.flatnonzero(val.argmax(1)!=y):w.writerow([packed['train_ids'][va[i]],int(y[i]),int(val[i].argmax()),float(val[i].max())])
    from sklearn.metrics import confusion_matrix
    dump(OUT/'dima_confusion_matrix.json',{'classes':CLASSES,'matrix':confusion_matrix(y,val.argmax(1)).tolist()})
    emit('finetune_complete',**result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['ce','supcon']);p.add_argument('--all',action='store_true');args=p.parse_args()
    packed=np.load(ROOT/'data'/'dataset.npz')
    if args.smoke:return train('supcon',packed,smoke=True)
    if args.mode:return train(args.mode,packed)
    if args.all:
        records=[train(m,packed) for m in ['ce','supcon']];finalize(packed,records)
    else:p.error('Choose --smoke, --mode or --all')

if __name__=='__main__':main()
