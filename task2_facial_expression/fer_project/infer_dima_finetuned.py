"""Reload the selected public backbone plus trained delta, and reproduce CSV."""
import argparse,csv,json
from pathlib import Path
from finetune_dima import OUT,ROOT,Model,load_delta,predict,torch,np,seed_everything
from upgrade_experiments import FeatureHead,predict_head

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'submission_dima_reproduced.csv');args=p.parse_args()
    seed_everything(42);record=json.loads((OUT/'dima_finetune_results.json').read_text());selected=record['selected']
    tta_meta=OUT/'dima_tta_results.json'
    if tta_meta.exists() and json.loads(tta_meta.read_text()).get('adopted'):
        import subprocess,sys
        subprocess.run([sys.executable,'-u','-X','utf8',str(Path(__file__).with_name('tta_dima.py')),'--reproduce','--output',str(args.output)],check=True)
        return
    data=np.load(ROOT/'data'/'dataset.npz')
    checkpoint=torch.load(record['final_checkpoint_paths'][-1],map_location='cpu',weights_only=True)
    if selected['mode']=='linear':
        # Regenerate image features for a standalone reproduction, rather than rely on cached test features.
        from upgrade_models import UpgradeFER,UpgradeImages
        source=UpgradeFER('hf_dima').cuda().eval()
        head=FeatureHead(source.initial_weight,source.initial_bias,'linear').cuda().eval();head.load_state_dict(checkpoint['model'])
        dl=torch.utils.data.DataLoader(UpgradeImages(data['test_images'],'hf_dima'),batch_size=16,num_workers=0)
        result=[]
        with torch.inference_mode():
            for x,_ in dl:
                x=x.cuda();views=[]
                for flip in range(2):
                    with torch.autocast('cuda',dtype=torch.bfloat16):_,f=source(x if not flip else x.flip(-1))
                    z,_=head(f.float().half().float());views.append(z.float().softmax(-1))
                result.append(torch.stack(views).mean(0).cpu().numpy())
        prob=np.concatenate(result)
    else:
        model=Model(selected['mode']).cuda();load_delta(model,checkpoint['model'])
        prob=predict(model,data['test_images']);prob=prob.mean(1) if selected['tta']=='flip' else prob[:,0]
    with args.output.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(data['test_ids'],prob.argmax(1)))
    saved=np.load(OUT/'final_probabilities.npz')['probabilities']
    report={'output':str(args.output),'rows':len(prob),'same_labels':bool(np.array_equal(prob.argmax(1),saved.argmax(1))),
            'maximum_probability_difference':float(np.abs(prob-saved).max())}
    (OUT/'dima_inference_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))

if __name__=='__main__':main()
