"""Regenerate predictions from the five saved DINO LP-FT checkpoints.

Run from the project root. Optional --size permits an explicitly separate
resolution experiment; never overwrites the original training predictions.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import plant_finetune as runner
from plant_dino_lpft import DinoNetwork
from plant_v3_features import save_submission


@torch.inference_mode()
def infer(model, batches, device, label):
    model.eval(); result=[];begin=time.time()
    for step,(x,_) in enumerate(batches):
        x=x.to(device);p=[]
        for j in range(8):
            v=x.rot90(j//2,[2,3]);v=v.flip(3) if j%2 else v
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                logits=model(v)
            p.append(logits.float().softmax(1))
        result.append(torch.stack(p).mean(0).cpu().numpy())
        if (step+1)%20==0 or step+1==len(batches):
            print(json.dumps({'stage':label,'batch':step+1,'batches':len(batches),'seconds':round(time.time()-begin,1)}),flush=True)
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='outputs/stronger_20260929/dino_lpft')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--size', type=int)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    run, out = Path(a.run_dir), Path(a.output_dir)
    if run.resolve() == out.resolve():
        raise ValueError('Use a separate output directory to preserve the training run')
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/'config.json').read_text())
    size = a.size or config['size']
    inference_config=dict(vars(a),size=size,training_config=config,views=8)
    cfg=out/'inference_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=inference_config:
        raise ValueError('Inference config differs; use a new directory')
    cfg.write_text(json.dumps(inference_config,indent=2))
    if size % 14:
        raise ValueError('DINOv2 input size must be divisible by 14')
    os.environ.setdefault('TORCH_HOME', str(Path('work/torch_cache').resolve()))
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, counts, root = runner.discover_training_images(config['train_dir'])
    classes = np.array(sorted(counts))
    sample = runner.read_submission(config['sample'])
    paths = [root/r[0] for r in rows]
    test_paths = [Path(config['test_dir'])/r['ID'] for r in sample]
    h = hashlib.sha256()
    for path in paths+test_paths:
        h.update(path.name.encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    if h.hexdigest() != config['data_hash']:
        raise ValueError('Dataset or row order differs from training')
    y = np.array([list(classes).index(r[1]) for r in rows])
    images, test_images = runner.load_images(paths), runner.load_images(test_paths)
    oof = np.zeros((len(rows), len(classes)))
    fold_ids = np.zeros(len(rows), int)
    predictions, scores = [], []
    for fold, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y,y), 1):
        cached=out/f'fold{fold}_predictions.npz'
        if cached.exists():
            z=np.load(cached);assert np.array_equal(z['val_indices'],va)
            oof[va]=z['val'];fold_ids[va]=fold;predictions.append(z['test'])
            scores.append(float(accuracy_score(y[va],z['val'].argmax(1))))
            print(json.dumps({'fold':fold,'cached':True,'accuracy':scores[-1]}),flush=True)
            continue
        checkpoint = torch.load(run/f'fold{fold}.pth', map_location='cpu', weights_only=True, mmap=True)
        assert checkpoint['config'] == config and checkpoint['fold'] == fold
        assert checkpoint['classes'] == classes.tolist()
        model = DinoNetwork()
        model.classifier[-1] = nn.Linear(6144, len(classes))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model = model.to(device).eval()
        valid = runner.loader(images, va, y, size, a.batch_size)
        test = runner.loader(test_images, np.arange(len(test_images)), np.zeros(len(test_images)), size, a.batch_size)
        vp = infer(model, valid, device,f'fold{fold}_valid')
        print(json.dumps({'fold':fold,'validation_accuracy':float(accuracy_score(y[va],vp.argmax(1)))}),flush=True)
        tp = infer(model, test, device,f'fold{fold}_test')
        np.savez_compressed(cached,val_indices=va,val=vp,test=tp)
        oof[va], fold_ids[va] = vp, fold
        predictions.append(tp)
        scores.append(float(accuracy_score(y[va], vp.argmax(1))))
        print(json.dumps({'fold': fold, 'size': size, 'accuracy': scores[-1]}), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    test = np.mean(predictions, axis=0)
    np.savez_compressed(out/'probabilities.npz', oof=oof, test=test, labels=classes[y], classes=classes, folds=fold_ids)
    save_submission(out/'submission.csv', [r['ID'] for r in sample], classes[test.argmax(1)])
    report = dict(accuracy=float(accuracy_score(y, oof.argmax(1))), fold_scores=scores, size=size,
                  checkpoint_run=str(run), data_hash=config['data_hash'], views=8,
                  note='Five held-out folds; test predictions average all five checkpoint models.')
    (out/'cv_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
