"""Serial GPU queue: one seed/holdout, four ablations, validation selection, final full-data fit."""
import argparse,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('FER_ROOT',r'D:\FER2026task2'))
OUT=HERE.parent

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def execute(variant,name,epochs,full=False,schedule_epochs=12):
    run=ROOT/'runs'/name
    if (run/'completed.json').exists(): return
    command=[sys.executable,'-u','-X','utf8',str(HERE/'fer.py'),'train','--variant',variant,
             '--name',name,'--epochs',str(epochs),'--schedule-epochs',str(schedule_epochs),'--size','160','--batch','64','--workers','0']
    if full: command.append('--full')
    log=ROOT/'runs'/(name+'.log')
    print(json.dumps({'event':'launch','name':name,'command':command,'log':str(log)}),flush=True)
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=HERE)
    if result.returncode: raise RuntimeError(f'{name} failed, see {log}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--variants',nargs='+',default=['baseline','local','scn','rul'],choices=['baseline','local','scn','rul'])
    p.add_argument('--no-refit',action='store_true');args=p.parse_args()
    variants=args.variants;assert len(variants)>=2;results=[]
    for variant in variants:
        name=variant+'_s42_w0'; execute(variant,name,args.epochs,schedule_epochs=args.epochs)
        run=ROOT/'runs'/name; score=read(run/'validation_tta.json'); done=read(run/'completed.json')
        results.append({'variant':variant,'name':name,'accuracy':score['accuracy'],'macro_f1':score['macro_f1'],
                        'best_epoch':done['best_epoch'],'elapsed_seconds':done['elapsed_seconds']})
        write(OUT/'experiment_results.json',results)
    ranked=sorted(results,key=lambda r:r['accuracy'],reverse=True); first,second=ranked[:2]
    a=np.load(ROOT/'runs'/first['name']/'validation_tta.npz')
    b=np.load(ROOT/'runs'/second['name']/'validation_tta.npz')
    assert np.array_equal(a['indices'],b['indices']) and np.array_equal(a['labels'],b['labels'])
    ensemble=(a['probabilities']+b['probabilities'])/2
    ensemble_acc=float((ensemble.argmax(1)==a['labels']).mean())
    selected=[first,second] if ensemble_acc>=first['accuracy']+.0025 else [first]
    selection={'results':results,'selection_rule':'best validation TTA accuracy; equal top-2 ensemble only if gain >=0.25 percentage points',
               'top2_ensemble_validation_accuracy':ensemble_acc,'selected_variants':[r['variant'] for r in selected],
               'single_seed':42,'cross_validation':False,'metric_note':'held-out training subset; not Kaggle score'}
    write(OUT/'model_selection.json',selection)
    probs=[]; final_runs=[]; ids=None
    for result in selected:
        epoch=result['best_epoch']
        if epoch>0 and not args.no_refit:
            name=result['variant']+'_full_s42'; execute(result['variant'],name,epoch,True,schedule_epochs=args.epochs)
        else: name=result['name']
        pred=np.load(ROOT/'runs'/name/'test_probabilities.npz')
        if ids is not None: assert np.array_equal(ids,pred['ids'])
        ids=pred['ids'];probs.append(pred['probabilities']);final_runs.append(name)
    probabilities=np.mean(probs,axis=0); labels=probabilities.argmax(1)
    with (OUT/'submission.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(ids,labels))
    assert len(ids)==7065 and len(set(ids))==7065 and set(labels)<=set(range(6))
    np.savez(OUT/'final_probabilities.npz',ids=ids,probabilities=probabilities)
    with (OUT/'experiment_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    selection.update({'final_runs':final_runs,'submission_rows':len(ids),'completed':True,'full_refit':not args.no_refit,
                      'priority_note':'Strong pretrained feature adaptation takes priority; unrun B0 variants are not completed results.' if args.no_refit else ''})
    write(OUT/'model_selection.json',selection)
    (OUT/'run_logs').mkdir(exist_ok=True)
    for name in [r['name'] for r in results]+final_runs:
        run=ROOT/'runs'/name; dst=OUT/'run_logs'/name;dst.mkdir(exist_ok=True)
        for fn in ['config.json','split.json','history.jsonl','validation_best.json','validation_tta.json','completed.json']:
            if (run/fn).exists(): shutil.copy2(run/fn,dst/fn)
    print(json.dumps({'event':'all_completed','submission':str(OUT/'submission.csv'),'selection':selection}),flush=True)

if __name__=='__main__': main()
