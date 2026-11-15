import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

LINKED_GROUPS = {1: 1, 2: 1, 6: 1, 39: 39, 77: 39}
FOLD_SEED = 20261027


def fundus_window(image):
    mask = np.asarray(image).max(2) > 25
    cols = np.flatnonzero(mask.sum(0) > image.height * 0.06)
    rows = np.flatnonzero(mask.sum(1) > image.width * 0.06)
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])
    side = max(x1 - x0 + 1, y1 - y0 + 1) * 1.04
    return [(x0 + x1) / 2, (y0 + y1) / 2], [side / 200, side / 200]


def prepare(archive, root):
    root = Path(root)
    prepared = root / 'data' / 'prepared'
    original = root / 'data' / 'original'
    prepared.mkdir(parents=True, exist_ok=True)
    original.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        text = z.read('detection/fovea_localization_train_GT.csv').decode('utf-8-sig')
        truth = {int(r['data']): [float(r['Fovea_X']), float(r['Fovea_Y'])] for r in csv.DictReader(io.StringIO(text))}
        rows = []
        for idx in range(1, 101):
            split = 'train' if idx <= 80 else 'test'
            raw = z.read(f'detection/{split}/{idx:04d}.jpg')
            (original / f'{idx:04d}.jpg').write_bytes(raw)
            image = Image.open(io.BytesIO(raw)).convert('RGB')
            width, height = image.size
            image.thumbnail((900, 900), Image.Resampling.LANCZOS)
            image.save(prepared / f'{idx:04d}.png')
            center, scale = fundus_window(image)
            rows.append(dict(id=idx, split=split, width=width, height=height,
                             resize=[image.width / width, image.height / height],
                             center=center, scale=scale, xy=truth.get(idx, [0.0, 0.0]),
                             group=LINKED_GROUPS.get(idx, idx)))
        (root / 'data' / 'sample_submission.csv').write_bytes(z.read('detection/sample_submission.csv'))
    train = rows[:80]
    strata = [f"{r['width']}_{int(r['xy'][0] / r['width'] > 0.5)}" for r in train]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, val) in enumerate(splitter.split(np.zeros((80, 1)), strata, [r['group'] for r in train])):
        for i in val:
            rows[i]['fold'] = fold
    (root / 'data' / 'manifest.json').write_text(json.dumps(rows, indent=2))
    return len(rows)
