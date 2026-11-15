"""V2 experiments, preserving V1 artifacts and the original five image splits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
import pipeline as base

ROOT = base.ROOT
V2 = ROOT/'v2'
sys.path.insert(0, str(V2/'deps'))


def clahe_green(g):
    """NumPy CLAHE: 8x8 tiles, clip=2, interpolated clipped histogram CDFs.

    Implements the documented CLAHE principle; not bitwise OpenCV equivalence.
    Uses only the current image and needs no additional package/download.
    """
    u = np.rint(g*255).clip(0, 255).astype(np.uint8)
    tables = np.empty((8, 8, 256), np.float32)
    for row in range(8):
        for col in range(8):
            h = np.bincount(u[row*64:(row+1)*64, col*64:(col+1)*64].reshape(-1), minlength=256)
            clipped = np.minimum(h, 32)
            excess = int((h-clipped).sum())
            clipped += excess//256
            remainder = excess%256
            if remainder:
                clipped[np.floor(np.arange(remainder)*256/remainder).astype(int)] += 1
            tables[row, col] = np.cumsum(clipped)/4096
    pos = np.clip((np.arange(512)+.5)/64-.5, 0, 7)
    low = pos.astype(int)
    high = np.minimum(low+1, 7)
    w = (pos-low).astype(np.float32)
    yy, xx = np.indices((512, 512))
    a = tables[low[yy], low[xx], u]
    b = tables[low[yy], high[xx], u]
    c = tables[high[yy], low[xx], u]
    d = tables[high[yy], high[xx], u]
    return ((a*(1-w[xx])+b*w[xx])*(1-w[yy])+(c*(1-w[xx])+d*w[xx])*w[yy]).astype(np.float32)


def arrays(split, clahe=False):
    x, y, nearest, ids = base.load_arrays(split)
    if clahe:
        extra = np.stack([clahe_green(im[1]) for im in x])[:, None]
        x = np.concatenate([x, extra], 1)
    return np.ascontiguousarray(x), y, nearest, ids


def model_new(channels=4):
    model = base.UNet(24)
    if channels != 4:
        model.enc0.net[0] = nn.Conv2d(channels, 24, 3, padding=1, bias=False)
    return model.to(device='cuda', memory_format=torch.channels_last)


def load_model(path, channels=None):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    nc = channels or ck.get('channels', 4)
    model = model_new(nc)
    state = ck['model']
    key = 'enc0.net.0.weight'
    if state[key].shape[1] != nc:
        expanded = torch.zeros((24, nc, 3, 3), device='cuda')
        expanded[:, :state[key].shape[1]] = state[key]
        state = {**state, key: expanded}
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def predict(model, x, style='d4'):
    model.eval()
    scales = (1.,) if style in ('d4', 'd8') else (.875, 1., 1.125)
    flips = (False,) if style == 'd4' else (False, True)
    output = []
    for im in x:
        original = torch.from_numpy(im[None]).cuda()
        result = torch.zeros((1, 1, 512, 512), device='cuda')
        count = 0
        for scale in scales:
            side = int(round(512*scale/8)*8)
            inp = original if side == 512 else F.interpolate(original, (side, side), mode='bilinear', align_corners=False)
            for flip in flips:
                z = inp.flip(-1) if flip else inp
                for k in range(4):
                    with torch.autocast('cuda', dtype=torch.float16):
                        p = model(torch.rot90(z, k, (-2, -1))).float().sigmoid()
                    p = torch.rot90(p, -k, (-2, -1))
                    if flip:
                        p = p.flip(-1)
                    if side != 512:
                        p = F.interpolate(p, (512, 512), mode='bilinear', align_corners=False)
                    result += p
                    count += 1
        output.append((result/count)[0, 0].cpu().numpy())
    return np.stack(output)


def metrics(p, y):
    candidates = [dict(threshold=round(float(t), 3), dice=float(base.dice_scores(p, y, t).mean())) for t in np.arange(.3, .701, .025)]
    best = max(candidates, key=lambda r:r['dice'])
    return dict(dice05=float(base.dice_scores(p, y, .5).mean()), **best)


def diagnose():
    x, y, _, ids = arrays('train')
    saved = np.load(base.RUN/'probabilities.npz')
    p0 = saved['oof'].astype(np.float32)
    results = {'baseline_d4': metrics(p0, y)}
    # Cheap postprocessing controls operate only on held-out validation outputs.
    for sig in (.35, .65):
        pp = ndi.gaussian_filter(p0, sigma=(0, sig, sig))
        results[f'gaussian_{sig}'] = metrics(pp, y)
    oofs = {s: np.zeros_like(p0) for s in ('d8', 'ms_d8')}
    splits = base.fold_splits(ids)
    for fold, valid in enumerate(splits):
        model = load_model(base.RUN/f'fold{fold}'/'best.pt')
        for style in oofs:
            pp = predict(model, x[valid], style)
            oofs[style][valid] = pp
        del model
        torch.cuda.empty_cache()
        print('DIAGNOSTIC_FOLD '+str(fold), flush=True)
    for style, pp in oofs.items():
        results[style] = metrics(pp, y)
        np.savez_compressed(V2/f'baseline_{style}.npz', ids=np.array(ids), oof=pp.astype(np.float16))
    for threshold in (.45, .475, .5):
        pred = p0 >= threshold
        target = y[:, 0] > .5
        tp = (pred & target).sum((1, 2))
        precision = tp/np.maximum(pred.sum((1, 2)), 1)
        recall = tp/np.maximum(target.sum((1, 2)), 1)
        results[f'error_breakdown_{threshold}'] = [dict(id=n, precision=float(pr), recall=float(re)) for n, pr, re in zip(ids, precision, recall)]
    base.save_json(V2/'diagnostic.json', results)
    print(json.dumps({k:v for k,v in results.items() if not k.startswith('error')}, indent=2), flush=True)


def sample(x, y, rng, patch, batch=12):
    xx, yy = [], []
    for i in rng.integers(len(x), size=batch):
        top, left = rng.integers(0, 513-patch, size=2)
        a, b = x[i, :, top:top+patch, left:left+patch], y[i, :, top:top+patch, left:left+patch]
        k = int(rng.integers(4))
        a, b = np.rot90(a, k, (-2, -1)), np.rot90(b, k, (-2, -1))
        if rng.random()<.5:
            a, b = a[..., ::-1], b[..., ::-1]
        xx.append(a)
        yy.append(b)
    a = torch.from_numpy(np.ascontiguousarray(np.stack(xx))).cuda().contiguous(memory_format=torch.channels_last)
    b = torch.from_numpy(np.ascontiguousarray(np.stack(yy))).cuda()
    scale = torch.empty((batch, a.shape[1], 1, 1), device='cuda').uniform_(.85, 1.15)
    shift = torch.empty((batch, a.shape[1], 1, 1), device='cuda').uniform_(-.05, .05)
    a = (a*scale+shift).clamp(0, 1)
    if rng.random()<.2:
        a = (a+torch.randn_like(a)*.015).clamp(0, 1)
    return a, b


def finetune(args, fold, variant, pilot=False):
    enhanced = variant == 'context_clahe'
    x, y, _, ids = arrays('train', clahe=enhanced)
    valid = base.fold_splits(ids)[fold]
    train = [i for i in range(len(ids)) if i not in valid]
    tx, ty = x[train], y[train]
    seed = 20261103+fold*100+9000
    base.seed_all(seed)
    rng = np.random.default_rng(seed)
    steps = args.steps
    patch = 224 if enhanced else 160
    nc = x.shape[1]
    out = V2/('pilots' if pilot else 'runs')/variant/f'fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(base.RUN/f'fold{fold}'/'best.pt', nc)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    # The warm start has NOT seen these four validation images.
    initial = predict(ema, x[valid], 'd4')
    best_score = float(base.dice_scores(initial, y[valid], .5).mean())
    best_step = 0
    torch.save(dict(model=ema.state_dict(), channels=nc, fold=fold, step=0, seed=seed, variant=variant), out/'best.pt')
    history = [dict(step=0, dice=best_score)]
    start = time.time()
    for step in range(1, steps+1):
        model.train()
        a, b = sample(tx, ty, rng, patch)
        lr = .00002+.00018*.5*(1+math.cos(math.pi*(step-1)/steps))
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = base.loss_fn(model(a), b)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for ep, mp in zip(ema.parameters(), model.parameters()):
                ep.lerp_(mp, .02)
            for eb, mb in zip(ema.buffers(), model.buffers()):
                eb.copy_(mb)
        if step%100 == 0 or step == steps:
            pp = predict(ema, x[valid], 'd4')
            score = float(base.dice_scores(pp, y[valid], .5).mean())
            if score > best_score:
                best_score, best_step = score, step
                torch.save(dict(model=ema.state_dict(), channels=nc, fold=fold, step=step, seed=seed, variant=variant), out/'best.pt')
            row = dict(fold=fold, variant=variant, step=step, total=steps, dice=score,
                       best=best_score, loss=float(loss.detach()), seconds=round(time.time()-start, 2))
            history.append(row)
            base.save_json(out/'history.json', history)
            base.save_json(V2/'status.json', row)
            print(json.dumps(row), flush=True)
    chosen = load_model(out/'best.pt')
    pp = predict(chosen, x[valid], 'd4')
    np.savez_compressed(out/'validation.npz', ids=np.array([ids[i] for i in valid]), probabilities=pp.astype(np.float16))
    summary = dict(fold=fold, variant=variant, train_ids=[ids[i] for i in train], valid_ids=[ids[i] for i in valid],
                   initial=float(base.dice_scores(initial, y[valid], .5).mean()), best_step=best_step,
                   best=best_score, calibrated=metrics(pp, y[valid]), steps=steps, patch=patch, batch=12,
                   seconds=round(time.time()-start, 2))
    base.save_json(out/'summary.json', summary)
    print('FINISHED '+json.dumps(summary), flush=True)
    del chosen, model, ema, opt, scaler
    torch.cuda.empty_cache()
    return summary


def export(pred, test_ids, threshold, details):
    dest = V2/'delivery'
    white, black = dest/'vessel_white', dest/'vessel_black'
    white.mkdir(parents=True, exist_ok=True)
    black.mkdir(parents=True, exist_ok=True)
    for name, pp in zip(test_ids, pred):
        mask = (pp>=threshold).astype(np.uint8)*255
        Image.fromarray(mask).save(white/(name+'.png'))
        Image.fromarray(255-mask).save(black/(name+'.png'))
    official = V2/'official_export'
    (official/'image').mkdir(parents=True, exist_ok=True)
    for p in white.glob('*.png'):
        shutil.copy2(p, official/'image'/p.name)
    shutil.copy2(base.DATA/'segmentation_to_csv.py', official/'segmentation_to_csv.py')
    subprocess.run([sys.executable, str(official/'segmentation_to_csv.py')], cwd=official, check=True)
    with (official/'submission.csv').open(newline='', encoding='utf-8') as f:
        values = {r['Id']:r['Predicted'] for r in csv.DictReader(f)}
    with (base.DATA/'sample_submission.csv').open(newline='', encoding='utf-8') as f:
        order = [r['Id'] for r in csv.DictReader(f)]
    assert set(order)==set(values)==set(test_ids)
    with (dest/'submission_v2.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Predicted'])
        for name in order:
            mask = np.asarray(Image.open(white/(name+'.png')))>0
            assert values[name] == base.encode(mask)
            assert np.array_equal(base.decode(values[name]), mask)
            assert 0<mask.mean()<.5
            w.writerow([name, values[name]])
    base.save_json(dest/'submission_validation.json', dict(valid=True, rows=20, sha256=hashlib.sha256((dest/'submission_v2.csv').read_bytes()).hexdigest(), **details))
    print('EXPORT_VALID '+str(dest/'submission_v2.csv'), flush=True)


def infer(args):
    enhanced = args.variant == 'context_clahe'
    x, y, _, ids = arrays('train', clahe=enhanced)
    xt, _, _, test_ids = arrays('test', clahe=enhanced)
    original = np.load(base.RUN/'probabilities.npz')
    oof = np.zeros((20, 512, 512), np.float32)
    test = np.zeros_like(oof)
    for fold, valid in enumerate(base.fold_splits(ids)):
        model = load_model(V2/'runs'/args.variant/f'fold{fold}'/'best.pt')
        oof[valid] = predict(model, x[valid], args.style)
        test += predict(model, xt, args.style)/5
        print('INFER_V2_FOLD '+str(fold), flush=True)
        del model
        torch.cuda.empty_cache()
    # Mixture weights selected on OOF predictions, never on hidden test labels.
    trials = []
    pbase = original['oof'].astype(np.float32)
    for alpha in (0., .25, .5, .75, 1.):
        pp = alpha*oof+(1-alpha)*pbase
        trials.append(dict(alpha=alpha, **metrics(pp, y)))
    best = max(trials, key=lambda r:r['dice'])
    alpha = best['alpha']
    combined = alpha*oof+(1-alpha)*pbase
    combined_test = alpha*test+(1-alpha)*original['test'].astype(np.float32)
    details = dict(baseline_online_reported=.85042, baseline_oof=metrics(pbase,y),
                   variant_oof=metrics(oof,y), selected=best, trials=trials,
                   per_image=[dict(id=name, baseline=float(d0), v2=float(d1), delta=float(d1-d0)) for name,d0,d1 in zip(ids,base.dice_scores(pbase,y,.475),base.dice_scores(combined,y,best['threshold']))],
                   style=args.style, variant=args.variant, note='Checkpoints, mixture and threshold selected using OOF; model-selection validation, not a new online score.')
    base.save_json(V2/'metrics.json', details)
    np.savez_compressed(V2/'probabilities.npz', train_ids=np.array(ids), oof=combined.astype(np.float32), test_ids=np.array(test_ids), test=combined_test.astype(np.float32))
    export(combined_test, test_ids, best['threshold'], details)
    print('V2_COMPLETE '+json.dumps(best), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['diagnose', 'pilot', 'train', 'infer'])
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 2])
    ap.add_argument('--variants', nargs='+', default=['control', 'context_clahe'])
    ap.add_argument('--variant', choices=['control', 'context_clahe'], default='control')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--style', choices=['d4', 'd8', 'ms_d8'], default='d8')
    args = ap.parse_args()
    V2.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.mode=='diagnose':
        diagnose()
    elif args.mode in ('pilot', 'train'):
        # Bilinear-upsample backward has no deterministic CUDA implementation;
        # deterministic forward inference remains available as a separate mode.
        torch.use_deterministic_algorithms(False)
        base.save_json(V2/(args.mode+'_config.json'), dict(**vars(args),
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       starting_models='V1 per-fold best.pt; unchanged original image splits',
                       seed_base=20261103+9000, lr_start=.0002, lr_end=.00002, ema_decay=.98,
                       batch=12, baseline_online_reported=.85042))
        summaries = []
        for fold in args.folds:
            for variant in args.variants:
                summaries.append(finetune(args, fold, variant, pilot=args.mode=='pilot'))
        base.save_json(V2/(args.mode+'_summary.json'), summaries)
    else:
        infer(args)


if __name__=='__main__':
    main()
