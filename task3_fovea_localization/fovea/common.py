import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def load_manifest(root):
    return json.loads((Path(root) / 'data' / 'manifest.json').read_text())


def split_rows(rows):
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    return train, test


def write_submission(path, root, test_rows, pred):
    values = {}
    for row, p in zip(test_rows, pred):
        if not np.isfinite(p).all():
            raise ValueError('non-finite prediction')
        values[f"{row['id']}_Fovea_X"] = float(np.clip(p[0], 0, row['width'] - 1))
        values[f"{row['id']}_Fovea_Y"] = float(np.clip(p[1], 0, row['height'] - 1))
    with (Path(root) / 'data' / 'sample_submission.csv').open(newline='') as f:
        template = list(csv.DictReader(f))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ImageID', 'value'])
        writer.writeheader()
        for row in template:
            writer.writerow({'ImageID': row['ImageID'], 'value': f"{values[row['ImageID']]:.6f}"})
    return len(template)
