"""Descriptive CLS-feature diagnostics, never used for checkpoint selection."""
import gc,json
from pathlib import Path
from finetune_dima import ROOT,OUT,Model,load_delta,torch,np,seed_everything,F
from upgrade_models import UpgradeImages

@torch.inference_mode()
def extract(model,images):
    model.eval();features=[]
    dl=torch.utils.data.DataLoader(UpgradeImages(images,'hf_dima'),batch_size=32,num_workers=0,shuffle=False)
    for x,_ in dl:
        with torch.autocast('cuda',dtype=torch.bfloat16):f=model.vit(x.cuda()).last_hidden_state[:,0]
        features.append(F.normalize(f.float(),dim=-1).cpu().numpy())
    return np.concatenate(features)

def main():
    seed_everything(42);data=np.load(ROOT/'data'/'dataset.npz');y=data['labels'];rng=np.random.default_rng(42)
    def subset(idx):return np.concatenate([rng.choice(idx[y[idx]==c],128,replace=False) for c in range(6)])
    tr=subset(data['train_idx']);va=subset(data['val_idx']);assert not set(tr)&set(va)
    yt,yv=y[tr],y[va];results=[];all_features={}
    for name in ['public','ce','supcon']:
        model=Model('ce' if name=='public' else name).cuda()
        if name!='public':
            ck=torch.load(ROOT/'runs'/f'dima_ft_{name}_holdout_s42'/'best.pt',map_location='cpu',weights_only=True)
            load_delta(model,ck['model'])
        ft=extract(model,data['train_images'][tr]);fv=extract(model,data['train_images'][va]);sim=fv@fv.T
        same=yv[:,None]==yv[None,:];np.fill_diagonal(same,False)
        diff=yv[:,None]!=yv[None,:]
        prototypes=np.stack([ft[yt==c].mean(0) for c in range(6)]);prototypes/=np.linalg.norm(prototypes,axis=1,keepdims=True)
        p=fv@prototypes.T;within=float(sim[same].mean());between=float(sim[diff].mean())
        results.append({'model':name,'within_class_cosine':within,'between_class_cosine':between,'cosine_gap':within-between,
                        'train_prototype_validation_accuracy':float((p.argmax(1)==yv).mean()),
                        'fear_sad_prototype_cosine':float(prototypes[1]@prototypes[3]),'sad_neutral_prototype_cosine':float(prototypes[3]@prototypes[5])})
        all_features[name+'_train']=ft;all_features[name+'_val']=fv
        print(json.dumps(results[-1]),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    np.savez(ROOT/'runs'/'dima_feature_diagnostics.npz',train_indices=tr,val_indices=va,**all_features)
    report={'seed':42,'per_class_per_partition':128,'training_samples':len(tr),'validation_samples':len(va),
            'representation':'L2-normalized CLS before classification/projector; original view only',
            'used_for_model_selection':False,'source_image_overlap_possible':True,'results':results}
    (OUT/'dima_feature_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines=['# Dima 主干特征诊断','','在训练和验证两侧分别按类抽取 128 张图，均使用 seed=42。提取归一化 CLS 特征；类别原型只由训练侧构造。结果仅作机制诊断，不用于挑选模型，也不替代完整验证分数。','',
           '|表示来源|同类平均余弦|异类平均余弦|两者差值|训练原型对验证子集分类准确率|','|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['model']}|{r['within_class_cosine']:.4f}|{r['between_class_cosine']:.4f}|{r['cosine_gap']:.4f}|{r['train_prototype_validation_accuracy']:.2%}|")
    lines+=['','同类/异类余弦差值与原型分类准确率描述这批样本上的特征结构。它们不构成因果证明，外部预训练重叠风险仍存在。']
    (OUT/'Dima特征诊断.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':main()
