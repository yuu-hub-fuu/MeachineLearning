import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .common import rmse, seed_all, split_rows


DARK_DIR = Path(os.environ.get('DARKPOSE_DIR', Path(__file__).resolve().parent.parent / 'third_party' / 'DarkPose'))
sys.path.insert(0, str(DARK_DIR / 'lib'))

from config import cfg as upstream_cfg
from core.inference import get_final_preds
from core.loss import JointsMSELoss
from models.pose_hrnet import get_pose_net

_spec = importlib.util.spec_from_file_location('joints_dataset', DARK_DIR / 'lib' / 'dataset' / 'JointsDataset.py')
_joints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_joints)


def load_upstream(dark_dir=DARK_DIR):
    return dict(cfg=upstream_cfg, preds=get_final_preds, loss=JointsMSELoss, net=get_pose_net,
                base=_joints.JointsDataset, dark_dir=Path(dark_dir))


def build_config(up, root, size):
    c = up['cfg'].clone()
    c.merge_from_file(str(up['dark_dir'] / 'experiments' / 'coco' / 'hrnet' / 'w32_256x192_adam_lr1e-3.yaml'))
    c.MODEL.NUM_JOINTS = 1
    c.MODEL.IMAGE_SIZE = [size, size]
    c.MODEL.HEATMAP_SIZE = [size // 4, size // 4]
    c.MODEL.SIGMA = 2
    c.MODEL.PRETRAINED = str(Path(root) / 'cache' / 'hrnet_w32-36af842e.pth')
    c.DATASET.ROT_FACTOR = 20
    c.DATASET.SCALE_FACTOR = 0.15
    c.DATASET.PROB_HALF_BODY = 0.0
    c.DATASET.COLOR_RGB = True
    c.TEST.BLUR_KERNEL = 11
    c.freeze()
    return c


class FoveaPoints(_joints.JointsDataset):
    def __init__(self, c, root, rows, train=False):
        ops = [T.ToPILImage()]
        if train:
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.015))
        ops += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        super().__init__(c, str(Path(root) / 'data' / 'prepared'), 'train' if train else 'val', train, T.Compose(ops))
        self.num_joints = 1
        self.rows = rows
        for r in rows:
            xy = np.array(r['xy']) * r['resize']
            visible = float(r['split'] == 'train' and any(r['xy']))
            self.db.append(dict(
                image=str(Path(root) / 'data' / 'prepared' / f"{r['id']:04d}.png"),
                filename=str(r['id']),
                joints_3d=np.array([[*xy, 0]], dtype=np.float32),
                joints_3d_vis=np.full((1, 3), visible, dtype=np.float32),
                center=np.array(r['center'], dtype=np.float32),
                scale=np.array(r['scale'], dtype=np.float32)))


class HRNetDark:
    def __init__(self, root, size=384):
        self.root = Path(root)
        self.size = size
        self.up = load_upstream()
        self.cfg = build_config(self.up, root, size)

    def loader(self, rows, train=False, batch=8):
        workers = 2 if train else 0
        return DataLoader(FoveaPoints(self.cfg, self.root, rows, train), batch_size=batch, shuffle=train,
                          num_workers=workers, pin_memory=True, persistent_workers=bool(workers))

    @torch.inference_mode()
    def predict(self, model, rows, batch=8):
        model.eval()
        out = []
        for x, _, _, meta in self.loader(rows, batch=batch):
            x = x.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                heat = (model(x) + model(x.flip(-1)).flip(-1)) * 0.5
            xy, _ = self.up['preds'](self.cfg, heat.float().cpu().numpy(),
                                     meta['center'].numpy(), meta['scale'].numpy())
            out.extend(xy[:, 0])
        return np.array(out) / np.array([r['resize'] for r in rows])

    def load(self, checkpoint):
        model = self.up['net'](self.cfg, is_train=False).cuda()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['state_dict'])
        return model

    def train_fold(self, rows, fold, name, epochs=40, batch=12, lr=2e-4, seed=20261027, keep_last=True):
        seed_all(seed + fold)
        train_rows, test_rows = split_rows(rows)
        fit = [r for r in train_rows if fold < 0 or r['fold'] != fold]
        val = [r for r in train_rows if r['fold'] == fold]
        run = self.root / 'runs' / name / f'fold{fold}'
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'done.json').exists():
            return run
        model = self.up['net'](self.cfg, is_train=True).cuda()
        head_params = list(model.final_layer.parameters())
        head_ids = {id(p) for p in head_params}
        optimizer = torch.optim.AdamW(
            [{'params': [p for p in model.parameters() if id(p) not in head_ids], 'lr': lr},
             {'params': head_params, 'lr': lr * 5}], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')
        criterion = self.up['loss'](True).cuda()
        loader = self.loader(fit, True, batch)
        best, started = float('inf'), time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for x, y, w, _ in loader:
                x, y, w = x.cuda(), y.cuda(), w.cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                loss = criterion(out.float(), y, w)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            scheduler.step()
            record = {'epoch': epoch, 'loss': float(np.mean(losses))}
            if val and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                pred = self.predict(model, val, batch)
                record['val_rmse'] = rmse(pred, [r['xy'] for r in val])
                if (keep_last and epoch == epochs) or (not keep_last and record['val_rmse'] < best):
                    best = record['val_rmse']
                    torch.save({'state_dict': model.state_dict(), 'size': self.size, 'epoch': epoch}, run / 'best.pt')
                    np.savez(run / 'validation.npz', ids=[r['id'] for r in val], pred=pred,
                             gt=[r['xy'] for r in val])
            with (run / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        if not val:
            torch.save({'state_dict': model.state_dict(), 'size': self.size, 'epoch': epochs}, run / 'best.pt')
        model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        pred = self.predict(model, test_rows, batch)
        np.savez(run / 'test.npz', ids=[r['id'] for r in test_rows], pred=pred)
        (run / 'done.json').write_text(json.dumps({'fold': fold, 'best_rmse': best if val else None,
                                                   'seconds': time.time() - started}))
        return run
