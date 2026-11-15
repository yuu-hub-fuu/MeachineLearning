"""Reproducible vessel segmentation; no external training data or weights.

Masks follow the ACTUAL supplied labels: vessel=255, background=0.
The official exporter resizes labels bilinearly to 512, then tests >0.
We use exactly that target convention and also report nearest-label Dice.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'segmentation'
RUN = ROOT / 'runs'
SIZE = 512


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(split):
    images, masks, nearest, ids = [], [], [], []
    for p in sorted((DATA / split / 'image').glob('*.jpg'), key=lambda p: int(p.stem)):
        rgb = np.asarray(Image.open(p).convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        # Per-image local green contrast supplies illumination-invariant vessel detail.
        g = rgb[..., 1]
        local = np.clip((g - gaussian_filter(g, 8)) * 4 + .5, 0, 1)
        x = np.concatenate([rgb, local[..., None]], axis=-1)
        images.append(x.transpose(2, 0, 1))
        ids.append(p.stem)
        if split == 'train':
            lab = Image.open(DATA / split / 'label' / p.name).convert('L')
            masks.append((np.asarray(lab.resize((SIZE, SIZE), Image.Resampling.BILINEAR)) > 0).astype(np.float32)[None])
            nearest.append((np.asarray(lab.resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 127).astype(np.float32)[None])
    x = np.ascontiguousarray(np.stack(images))
    return x, np.stack(masks) if masks else None, np.stack(nearest) if nearest else None, ids


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        w = width
        self.enc0 = Block(4, w)
        self.enc1 = Block(w, w*2)
        self.enc2 = Block(w*2, w*4)
        self.enc3 = Block(w*4, w*8)
        self.dec2 = Block(w*12, w*4)
        self.dec1 = Block(w*6, w*2)
        self.dec0 = Block(w*3, w)
        self.head = nn.Conv2d(w, 1, 1)
    def forward(self, x):
        a = self.enc0(x)
        b = self.enc1(F.max_pool2d(a, 2))
        c = self.enc2(F.max_pool2d(b, 2))
        d = self.enc3(F.max_pool2d(c, 2))
        u = self.dec2(torch.cat([F.interpolate(d, size=c.shape[-2:], mode='bilinear', align_corners=False), c], 1))
        u = self.dec1(torch.cat([F.interpolate(u, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        u = self.dec0(torch.cat([F.interpolate(u, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return self.head(u)


def fold_splits(ids, seed=20261103, nfolds=5):
    order = np.random.default_rng(seed).permutation(len(ids))
    return [sorted(order[f::nfolds].tolist()) for f in range(nfolds)]


def sample_batch(x, y, batch, patch, rng):
    # Sampling occurs only inside the fold training subset; no test/validation crops.
    ix = rng.integers(len(x), size=batch)
    xx, yy = [], []
    for i in ix:
        top, left = rng.integers(0, SIZE-patch+1, size=2)
        a, b = x[i, :, top:top+patch, left:left+patch], y[i, :, top:top+patch, left:left+patch]
        k = int(rng.integers(4))
        a, b = np.rot90(a, k, (-2, -1)), np.rot90(b, k, (-2, -1))
        if rng.random() < .5:
            a, b = a[..., ::-1], b[..., ::-1]
        xx.append(a)
        yy.append(b)
    xx = torch.from_numpy(np.ascontiguousarray(np.stack(xx))).cuda(non_blocking=True)
    yy = torch.from_numpy(np.ascontiguousarray(np.stack(yy))).cuda(non_blocking=True)
    # Modest photometric augmentation; same geometry for image and label.
    scale = torch.empty((batch, 4, 1, 1), device='cuda').uniform_(.85, 1.15)
    shift = torch.empty((batch, 4, 1, 1), device='cuda').uniform_(-.05, .05)
    xx = (xx*scale+shift).clamp(0, 1)
    if rng.random() < .2:
        xx = (xx + torch.randn_like(xx)*.015).clamp(0, 1)
    return xx, yy


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits.float(), y)
    p = logits.float().sigmoid()
    dice = (2*(p*y).sum((1, 2, 3))+1) / (p.sum((1, 2, 3))+y.sum((1, 2, 3))+1)
    return bce + (1-dice).mean()


@torch.inference_mode()
def predict(model, x, tta=False):
    model.eval()
    preds = []
    for arr in x:
        a = torch.from_numpy(arr[None]).cuda()
        results = []
        for k in (range(4) if tta else range(1)):
            z = torch.rot90(a, k, (-2, -1))
            with torch.autocast('cuda', dtype=torch.float16):
                p = model(z).float().sigmoid()
            results.append(torch.rot90(p, -k, (-2, -1)))
        preds.append(torch.stack(results).mean(0)[0, 0].cpu().numpy())
    return np.stack(preds)


def dice_scores(p, y, threshold=.5):
    a = p >= threshold
    b = y[:, 0] > .5 if y.ndim == 4 else y > .5
    return (2*(a & b).sum((1, 2))+1e-8) / (a.sum((1, 2))+b.sum((1, 2))+1e-8)


def train_one(args, fold, arrays, splits, smoke=False):
    x, y, nearest, ids = arrays
    valid = splits[fold]
    train = [i for i in range(len(ids)) if i not in valid]
    out = RUN / ('smoke' if smoke else f'fold{fold}')
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'split.json', dict(train=[ids[i] for i in train], valid=[ids[i] for i in valid]))
    seed = args.seed+fold*100
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = UNet(args.width).to(device='cuda', memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    total = args.smoke_steps if smoke else args.steps
    interval = min(args.eval_every, total)
    start, best, best_step = time.time(), -1., 0
    tx, ty = x[train], y[train]
    history = []
    for step in range(1, total+1):
        model.train()
        xx, yy = sample_batch(tx, ty, args.batch, args.patch, rng)
        xx = xx.contiguous(memory_format=torch.channels_last)
        lr = args.lr*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/total)))
        for pg in opt.param_groups:
            pg['lr'] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = loss_fn(model(xx), yy)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at fold={fold}, step={step}')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(opt)
        scaler.update()
        if step % interval == 0 or step == total:
            p = predict(model, x[valid])
            score = float(dice_scores(p, y[valid]).mean())
            near = float(dice_scores(p, nearest[valid]).mean())
            row = dict(step=step, loss=float(loss), dice=score, nearest_dice=near, lr=lr,
                       seconds=round(time.time()-start, 2), gpu_peak_mb=round(torch.cuda.max_memory_allocated()/2**20))
            history.append(row)
            improved = score > best
            if improved:
                best, best_step = score, step
                torch.save(dict(model=model.state_dict(), width=args.width, fold=fold, seed=seed,
                                step=step, valid_ids=[ids[i] for i in valid], score=score,
                                target='PIL bilinear 512 then >0; actual supplied vessel labels'), out / 'best.pt')
            save_json(out / 'history.json', history)
            status = dict(fold=fold, step=step, total=total, best_dice=best, best_step=best_step,
                          elapsed_seconds=row['seconds'], state='training' if step < total else 'complete')
            save_json(RUN / 'status.json', status)
            print(json.dumps(dict(**row, fold=fold, best=best, improved=improved)), flush=True)
    ckpt = torch.load(out / 'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'])
    p = predict(model, x[valid], tta=True)
    np.savez_compressed(out / 'validation.npz', ids=np.array([ids[i] for i in valid]), probabilities=p.astype(np.float16))
    summary = dict(fold=fold, best_step=best_step, best_dice_no_tta=best,
                   dice_tta=float(dice_scores(p, y[valid]).mean()),
                   nearest_dice_tta=float(dice_scores(p, nearest[valid]).mean()),
                   valid_ids=[ids[i] for i in valid], seconds=round(time.time()-start, 2))
    save_json(out / 'summary.json', summary)
    print('FOLD_COMPLETE '+json.dumps(summary), flush=True)
    del model, opt, scaler
    torch.cuda.empty_cache()
    return summary


def encode(mask):
    a = np.asarray(mask, dtype=np.uint8).reshape(-1)
    padded = np.pad(a, (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])+1
    changes[1::2] -= changes[::2]
    return ' '.join(map(str, changes.tolist()))


def decode(s):
    a = np.zeros(SIZE*SIZE, dtype=np.uint8)
    v = np.array([int(t) for t in s.split()], dtype=np.int64)
    if len(v) % 2 or (len(v) and np.any(v <= 0)):
        raise ValueError('RLE must contain positive start-length pairs')
    end = 0
    for start, length in v.reshape(-1, 2):
        if start <= end or start+length-1 > len(a):
            raise ValueError('RLE overlaps, is unsorted, or exceeds image dimensions')
        a[start-1:start-1+length] = 1
        end = start+length-1
    return a.reshape(SIZE, SIZE)


def infer(args):
    x, y, nearest, ids = load_arrays('train')
    xt, _, _, test_ids = load_arrays('test')
    oof = np.full((len(ids), SIZE, SIZE), np.nan, dtype=np.float32)
    pred = np.zeros((len(test_ids), SIZE, SIZE), dtype=np.float32)
    summaries = []
    for fold in range(5):
        out = RUN / f'fold{fold}'
        val = np.load(out / 'validation.npz')
        for name, pp in zip(val['ids'], val['probabilities']):
            ix = ids.index(str(name))
            if np.isfinite(oof[ix]).any():
                raise RuntimeError('OOF image seen more than once')
            oof[ix] = pp
        ckpt = torch.load(out / 'best.pt', map_location='cuda', weights_only=False)
        model = UNet(ckpt['width']).to(device='cuda', memory_format=torch.channels_last)
        model.load_state_dict(ckpt['model'])
        pred += predict(model, xt, tta=True)/5
        summaries.append(json.loads((out / 'summary.json').read_text(encoding='utf-8')))
        del model
        torch.cuda.empty_cache()
        print(f'INFERENCE fold {fold} finished', flush=True)
    assert np.isfinite(oof).all() and np.isfinite(pred).all()
    thresholds = np.round(np.arange(.15, .851, .025), 3)
    calibration = [dict(threshold=float(t), dice=float(dice_scores(oof, y, t).mean())) for t in thresholds]
    best = max(calibration, key=lambda r: r['dice'])
    threshold = best['threshold']
    near_calibration = [dict(threshold=float(t), dice=float(dice_scores(oof, nearest, t).mean())) for t in thresholds]
    save_json(RUN / 'metrics.json', dict(folds=summaries, oof_dice_05=float(dice_scores(oof, y).mean()),
              selected_threshold=threshold, oof_dice_selected=best['dice'], calibration=calibration,
              nearest_label_dice_at_selected=float(dice_scores(oof, nearest, threshold).mean()),
              nearest_calibration=near_calibration,
              per_image=[dict(id=name, dice=float(d)) for name, d in zip(ids, dice_scores(oof, y, threshold))],
              note='Validation selected checkpoints and threshold. These are model-selection estimates, not an untouched test score. No test labels used.'))
    np.savez_compressed(RUN / 'probabilities.npz', train_ids=np.array(ids), oof=oof.astype(np.float16),
                        test_ids=np.array(test_ids), test=pred.astype(np.float16))
    deliver = ROOT / 'delivery'
    white = deliver / 'vessel_white'
    black = deliver / 'vessel_black'
    white.mkdir(parents=True, exist_ok=True)
    black.mkdir(parents=True, exist_ok=True)
    for name, pp in zip(test_ids, pred):
        m = (pp >= threshold).astype(np.uint8)*255
        Image.fromarray(m).save(white / (name+'.png'))
        Image.fromarray(255-m).save(black / (name+'.png'))
    # Execute the supplied official exporter unmodified, with its expected ./image.
    # Stage a dedicated export folder to avoid any effects on original data.
    import shutil
    export = RUN / 'official_export'
    export.mkdir(parents=True, exist_ok=True)
    (export / 'image').mkdir(exist_ok=True)
    for p in white.glob('*.png'):
        shutil.copy2(p, export / 'image' / p.name)
    shutil.copy2(DATA / 'segmentation_to_csv.py', export / 'segmentation_to_csv.py')
    subprocess.run([sys.executable, str(export / 'segmentation_to_csv.py')], cwd=export, check=True)
    # Official traversal is lexicographic; restore exact sample row order.
    with (export / 'submission.csv').open(newline='', encoding='utf-8') as f:
        official_rows = {r['Id']: r['Predicted'] for r in csv.DictReader(f)}
    with (DATA / 'sample_submission.csv').open(newline='', encoding='utf-8') as f:
        order = [r['Id'] for r in csv.DictReader(f)]
    assert set(order) == set(official_rows) == set(test_ids)
    with (deliver / 'submission.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predicted'])
        for name in order:
            writer.writerow([name, official_rows[name]])
    validate_submission()
    # Validation previews show held-out predictions only.
    sheet = Image.new('RGB', (1024, 256*5), '#111111')
    draw = ImageDraw.Draw(sheet)
    worst = np.argsort(dice_scores(oof, y, threshold))[:5]
    for row, i in enumerate(worst):
        im = Image.open(DATA / 'train' / 'image' / (ids[i]+'.jpg')).convert('RGB').resize((256, 256))
        sheet.paste(im, (0, row*256))
        for col, arr in [(1, y[i, 0] > .5), (2, oof[i] >= threshold)]:
            thumb = Image.fromarray(arr.astype(np.uint8)*255).resize((256, 256), Image.Resampling.NEAREST).convert('RGB')
            sheet.paste(thumb, (col*256, row*256))
        color = np.array(im).copy()
        tp = (oof[i] >= threshold) & (y[i, 0] > .5)
        fp = (oof[i] >= threshold) & ~(y[i, 0] > .5)
        fn = ~(oof[i] >= threshold) & (y[i, 0] > .5)
        error = np.zeros((512, 512, 3), np.uint8)
        error[tp], error[fp], error[fn] = [100, 220, 100], [255, 80, 80], [80, 120, 255]
        sheet.paste(Image.fromarray(error).resize((256, 256)), (768, row*256))
        draw.text((3, row*256+3), f'OOF id={ids[i]} Dice={dice_scores(oof[i:i+1], y[i:i+1], threshold)[0]:.4f}', fill='white')
    sheet.save(deliver / 'validation_preview.png')
    preview = Image.new('RGB', (3*256, 4*256), '#111111')
    for row, i in enumerate([0, 4, 9, 19]):
        orig = Image.open(DATA / 'test' / 'image' / (test_ids[i]+'.jpg')).convert('RGB').resize((512, 512))
        mask = pred[i] >= threshold
        overlay = np.array(orig).copy()
        overlay[mask] = (.5*overlay[mask] + np.array([0, 127, 0])).astype(np.uint8)
        preview.paste(orig.resize((256, 256)), (0, row*256))
        preview.paste(Image.fromarray((mask*255).astype(np.uint8)).convert('RGB').resize((256, 256)), (256, row*256))
        preview.paste(Image.fromarray(overlay).resize((256, 256)), (512, row*256))
    preview.save(deliver / 'test_preview.png')
    save_json(RUN / 'status.json', dict(state='complete', submission=str(deliver / 'submission.csv'), **best))
    print('COMPLETE '+json.dumps(best), flush=True)


def validate_submission():
    dest = ROOT / 'delivery'
    with (dest / 'submission.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['Id', 'Predicted']
        rows = list(reader)
    with (DATA / 'sample_submission.csv').open(newline='', encoding='utf-8') as f:
        order = [r['Id'] for r in csv.DictReader(f)]
    assert [r['Id'] for r in rows] == order
    stats = []
    for row in rows:
        decoded = decode(row['Predicted'])
        original = np.asarray(Image.open(dest / 'vessel_white' / (row['Id']+'.png'))) > 0
        black = np.asarray(Image.open(dest / 'vessel_black' / (row['Id']+'.png')))
        assert original.shape == (SIZE, SIZE)
        assert np.array_equal(decoded, original)
        assert np.array_equal(black, (~original).astype(np.uint8)*255)
        assert encode(original) == row['Predicted']
        assert 0 < original.mean() < .5
        stats.append(dict(id=row['Id'], vessel_fraction=float(original.mean()), runs=len(row['Predicted'].split())//2))
    # Boundary and orientation cases: row-major, one-based, all/empty, edge pixels.
    for arr in [np.zeros((SIZE, SIZE), np.uint8), np.ones((SIZE, SIZE), np.uint8), np.eye(SIZE, dtype=np.uint8)]:
        assert np.array_equal(decode(encode(arr)), arr)
    save_json(dest / 'submission_validation.json', dict(valid=True, rows=len(rows), shape=[SIZE, SIZE],
              foreground='vessels; actual supplied white labels', order='sample_submission.csv',
              encoding='official unmodified script; one-based row-major; nonzero pixels', images=stats,
              sha256=hashlib.sha256((dest / 'submission.csv').read_bytes()).hexdigest()))
    print('SUBMISSION_VALID: 20 rows, official-script match, exact RLE roundtrip, binary masks', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['smoke', 'train', 'infer', 'validate'])
    ap.add_argument('--steps', type=int, default=1800)
    ap.add_argument('--smoke-steps', type=int, default=150)
    ap.add_argument('--eval-every', type=int, default=150)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patch', type=int, default=160)
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--folds', type=int, nargs='*', default=list(range(5)))
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'validate':
        validate_submission()
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for training/inference')
    torch.backends.cudnn.benchmark = True
    RUN.mkdir(parents=True, exist_ok=True)
    if args.mode == 'infer':
        infer(args)
        return
    arrays = load_arrays('train')
    ids = arrays[-1]
    splits = fold_splits(ids, args.seed)
    save_json(RUN / 'folds.json', {str(f): [ids[i] for i in v] for f, v in enumerate(splits)})
    save_json(RUN / ('smoke_config.json' if args.mode == 'smoke' else 'config.json'), dict(**vars(args),
              torch=torch.__version__, python=sys.version, gpu=torch.cuda.get_device_name(0),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              official_exporter_sha256=hashlib.sha256((DATA/'segmentation_to_csv.py').read_bytes()).hexdigest()))
    for fold in ([0] if args.mode == 'smoke' else args.folds):
        train_one(args, fold, arrays, splits, smoke=args.mode == 'smoke')
    if args.mode == 'train' and all((RUN / f'fold{f}' / 'best.pt').exists() for f in range(5)):
        infer(args)


if __name__ == '__main__':
    main()
