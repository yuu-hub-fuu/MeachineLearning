"""One fixed crop/neighbor comparison plus feature-grounded error inspection."""
import argparse,base64,csv,gc,html,io,json,shutil
from collections import defaultdict
from pathlib import Path
from upgrade_experiments import ROOT,OUT,torch,np,F,FeatureHead,score,sha,dump,seed_everything,MIN_GAIN
from upgrade_models import UpgradeFER,UpgradeImages
from optimize_predictions import neighbors
from fer import CLASSES,Image

def write_csv(path,header,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def load_head(record,full=False):
    if record['mode']=='native':return None
    h=np.load(Path(record['cache'])/'head.npz')
    model=FeatureHead(torch.from_numpy(h['weight']),torch.from_numpy(h['bias']),record['mode'])
    path=ROOT/'runs'/f"upgrade_{record['kind']}_{record['mode']}_full_s42"/'best.pt' if full else Path(record['checkpoint'])
    cp=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True)
    return model.cuda().eval()

@torch.inference_mode()
def cached_representation(record,indices,part='train',full=False):
    cache=Path(record['cache']);x=np.load(cache/f'{part}_features.npy',mmap_mode='r')
    native=np.load(cache/f'{part}_probs.npy',mmap_mode='r');head=load_head(record,full);probs=[];features=[]
    for start in range(0,len(indices),512):
        ids=indices[start:start+512];f=torch.from_numpy(np.array(x[ids],dtype=np.float32)).cuda()
        if head is None:p=torch.from_numpy(np.array(native[ids])).cuda()
        else:z,f=head(f);p=z.softmax(-1)
        if record['tta']=='single':p=p[:,0];f=f[:,0]
        else:p=p.mean(1);f=f.mean(1)
        probs.append(p.cpu().numpy());features.append(F.normalize(f,dim=-1).cpu().numpy())
    del head;gc.collect();torch.cuda.empty_cache()
    return np.concatenate(probs),np.concatenate(features)

@torch.inference_mode()
def crop_predictions(record,images,full=False):
    cache=Path(record['cache'])
    identity=sha(Path(__file__))[:10]
    if record['mode']!='native':
        checkpoint=ROOT/'runs'/f"upgrade_{record['kind']}_{record['mode']}_full_s42"/'best.pt' if full else Path(record['checkpoint'])
        identity+='_'+sha(checkpoint)[:10]
    file=cache/f'crop1_{"test" if full else "validation"}_{record["tta"]}_{identity}.npy'
    if file.exists():return np.load(file)
    source=UpgradeFER(record['kind']).cuda().eval();head=load_head(record,full);probs=[]
    dl=torch.utils.data.DataLoader(UpgradeImages(images,record['kind'],crop=1),batch_size=16,num_workers=0)
    for x,_ in dl:
        x=x.cuda();views=[]
        for view in range(2 if record['tta']=='flip' else 1):
            with torch.autocast('cuda',dtype=torch.bfloat16):z,f=source(x if view==0 else x.flip(-1))
            if head is not None:z,_=head(f.float().half().float())
            views.append(z.float().softmax(1))
        probs.append(torch.stack(views).mean(0).cpu().numpy())
    p=np.concatenate(probs);np.save(file,p);del source,head;gc.collect();torch.cuda.empty_cache();return p

def thumbnail(arr):
    b=io.BytesIO();Image.fromarray(arr).save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def error_report(data,record,p,train_p,train_f,val_f,idx,sim):
    folder=OUT/'upgrade_error_analysis';folder.mkdir(exist_ok=True)
    tr,va,y=data['train_idx'],data['val_idx'],data['labels'];pred=p.argmax(1)
    errors=np.flatnonzero(pred!=y[va]);errors=errors[np.argsort(-p[errors].max(1))]
    write_csv(folder/'validation_errors.csv',['image','label','prediction','confidence','nearest_training_image','nearest_label','cosine'],
        ([data['train_ids'][va[i]],int(y[va[i]]),int(pred[i]),float(p[i].max()),data['train_ids'][tr[idx[i,0]]],int(y[tr[idx[i,0]]]),float(sim[i,0])] for i in errors))
    centers=np.stack([train_f[y[tr]==c].mean(0) for c in range(6)])
    centers/=np.linalg.norm(centers,axis=1,keepdims=True).clip(1e-8)
    similarity=train_f@centers.T;own=similarity[np.arange(len(tr)),y[tr]]
    other=similarity.copy();other[np.arange(len(tr)),y[tr]]=-np.inf
    margin=own-other.max(1)
    review=np.flatnonzero((train_p.argmax(1)!=y[tr])&(train_p.max(1)>=.9)&(margin<0))
    write_csv(folder/'training_review_candidates.csv',['image','label','prediction','confidence','own_minus_other_prototype_similarity'],
        ([data['train_ids'][tr[i]],int(y[tr[i]]),int(train_p[i].argmax()),float(train_p[i].max()),float(margin[i])] for i in review))
    group=defaultdict(list)
    for i,g in enumerate(data['groups']):group[str(g)].append(i)
    conflicts=[{'pixel_sha256':g,'images':[str(data['train_ids'][i]) for i in ids],'labels':[int(y[i]) for i in ids]}
               for g,ids in group.items() if len(set(y[ids]))>1]
    dump(folder/'label_conflicts.json',conflicts)
    summary={'model':record['kind'],'mode':record['mode'],'validation_errors':len(errors),'training_review_candidates':len(review),
             'conflicting_pixel_groups':len(conflicts),'feature_bank':'holdout training rows only','labels_changed':0,
             'note':'Disagreement and prototype margin are review signals, not proof of wrong labels; external pretraining may overlap source images.'}
    dump(folder/'summary.json',summary)
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>升级模型误分类复查</title>',
          '<style>body{font:16px system-ui;margin:36px auto;max-width:1100px;background:#f2f4f7;color:#172438}section{background:white;padding:20px;margin:20px 0}.row{display:flex;gap:18px}figure{margin:0;width:23%}img{width:144px;height:144px}figcaption{font-size:12px;overflow-wrap:anywhere}p{line-height:1.6}</style>',
          f'<h1>升级模型误分类与训练近邻</h1><p>特征模型：{html.escape(record["kind"])} / {html.escape(record["mode"])}。仅用训练子集建立近邻库。下列分歧不能证明错标，原标签未修改。可能存在外部预训练源图像重叠。</p>']
    for i in errors[:36]:
        page.append(f'<section><h2>{CLASSES[y[va[i]]]} → {CLASSES[pred[i]]}，置信度 {p[i].max():.1%}</h2><div class="row">')
        page.append(f'<figure><img src="{thumbnail(data["train_images"][va[i]])}"><figcaption>验证图<br>{html.escape(str(data["train_ids"][va[i]]))}</figcaption></figure>')
        for rank in range(3):
            j=tr[idx[i,rank]];page.append(f'<figure><img src="{thumbnail(data["train_images"][j])}"><figcaption>训练近邻 {rank+1}: {CLASSES[y[j]]}<br>cos={sim[i,rank]:.3f}<br>{html.escape(str(data["train_ids"][j]))}</figcaption></figure>')
        page.append('</div></section>')
    page.append('</html>');(folder/'error_gallery.html').write_text('\n'.join(page),encoding='utf-8')

def final_prediction(selection,data,post):
    chosen=selection['upgrade']['selected'];predictions=[]
    for r in chosen:
        p,_=cached_representation(r,np.arange(len(data['test_ids'])),'test',True)
        if post['crop']:p=(p+crop_predictions(r,data['test_images'],True))/2
        predictions.append(p)
    result=np.mean(predictions,axis=0)
    if post['knn_weight']:
        anchor=chosen[0]
        _,bank=cached_representation(anchor,np.arange(len(data['labels'])),'train',True)
        _,query=cached_representation(anchor,np.arange(len(data['test_ids'])),'test',True)
        knn,_,_=neighbors(bank,data['labels'],query)
        a=post['knn_weight'];result=(1-a)*result+a*knn
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inference-only',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();seed_everything(42);selection=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    assert selection.get('upgrade',{}).get('adopted_over_core')
    data=np.load(ROOT/'data'/'dataset.npz');chosen=selection['upgrade']['selected'];va=data['val_idx'];tr=data['train_idx'];y=data['labels'][va]
    if args.inference_only:
        p=final_prediction(selection,data,selection['upgrade_postprocessing']['selected'])
        output=args.output or ROOT/'submission_upgrade_post_reproduced.csv'
        write_csv(output,['ID','Emotion'],zip(data['test_ids'],p.argmax(1)));return
    original=np.load(OUT/'upgrade_validation_probabilities.npz');assert np.array_equal(original['indices'],va)
    base=original['probabilities'];baseline={'crop':False,'knn_weight':0.,**score(base,y)}
    crop=np.mean([crop_predictions(r,data['train_images'][va]) for r in chosen],axis=0)
    averaged=(base+crop)/2;candidate={'crop':True,'knn_weight':0.,**score(averaged,y)}
    best=candidate if candidate['accuracy']>=baseline['accuracy']+MIN_GAIN else baseline
    probability=averaged if best['crop'] else base;table=[baseline,candidate]
    anchor=chosen[0];train_p,train_f=cached_representation(anchor,tr)
    _,val_f=cached_representation(anchor,va);knn,idx,sim=neighbors(train_f,data['labels'][tr],val_f)
    for a in [.1,.2]:table.append({'crop':best['crop'],'knn_weight':a,**score((1-a)*probability+a*knn,y)})
    blend=max(table[-2:],key=lambda r:r['accuracy'])
    if blend['accuracy']>=best['accuracy']+MIN_GAIN:best=blend
    final_val=(1-best['knn_weight'])*probability+best['knn_weight']*knn
    post={'baseline':baseline,'candidates':table,'selected':best,'minimum_gain_per_stage':MIN_GAIN,'k':15,'temperature':.07,
          'validation_corrected':int(((base.argmax(1)!=y)&(final_val.argmax(1)==y)).sum()),
          'validation_new_errors':int(((base.argmax(1)==y)&(final_val.argmax(1)!=y)).sum()),
          'source_image_overlap_possible':selection['upgrade']['source_image_overlap_possible']}
    error_report(data,anchor,final_val,train_p,train_f,val_f,idx,sim)
    final=final_prediction(selection,data,best)
    assert final.shape==(7065,6) and np.isfinite(final).all() and np.allclose(final.sum(1),1,atol=1e-5)
    for name in ['submission.csv','final_probabilities.npz','model_selection.json']:
        backup=OUT/('before_upgrade_post_'+name)
        if not backup.exists():shutil.copy2(OUT/name,backup)
    write_csv(OUT/'submission.csv',['ID','Emotion'],zip(data['test_ids'],final.argmax(1)))
    np.savez(OUT/'final_probabilities.npz',ids=data['test_ids'],probabilities=final)
    np.savez(OUT/'upgrade_post_validation.npz',indices=va,labels=y,probabilities=final_val)
    selection['upgrade_postprocessing']=post;selection['final_validation_accuracy']=best['accuracy'];selection['final_validation_macro_f1']=best['macro_f1']
    dump(OUT/'model_selection.json',selection);dump(OUT/'upgrade_postprocessing.json',post)
    print(json.dumps({'event':'upgrade_postprocessing_completed',**post}),flush=True)

if __name__=='__main__':main()
