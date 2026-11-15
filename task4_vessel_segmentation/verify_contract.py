"""Independent comparison against supplied exporter and data manifest."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from pipeline import DATA, ROOT, SIZE, decode, encode, fold_splits, save_json


def main():
    spec = importlib.util.spec_from_file_location('official_exporter', DATA/'segmentation_to_csv.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    files, hashes, binary = [], {}, []
    for p in sorted(DATA.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        raw = p.read_bytes()
        row = dict(path=str(p.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if p.suffix == '.jpg':
            im = Image.open(p)
            arr = np.array(im)
            row.update(size=im.size, mode=im.mode, format=im.format,
                       pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest())
            hashes.setdefault(row['pixel_sha256'], []).append(row['path'])
            if p.parent.name == 'label':
                vals = np.unique(arr)
                assert vals.tolist() == [0, 255], (p, vals)
                # Compare training-target preparation with the UNMODIFIED exporter.
                expected = np.array(im.convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)) > 0
                exported = official.turn_to_str([str(p)])[0]
                assert encode(expected) == exported
                assert np.array_equal(decode(exported), expected)
                row.update(values=vals.tolist(), native_vessel_fraction=float((arr>0).mean()),
                           official512_vessel_fraction=float(expected.mean()))
                binary.append(p.stem)
        files.append(row)
    assert len(binary) == 20
    duplicates = [v for v in hashes.values() if len(v)>1]
    assert not duplicates
    splits = fold_splits([str(i) for i in range(1, 21)])
    assert sorted(i for v in splits for i in v) == list(range(20))
    assert all(len(v) == 4 for v in splits)
    for v in splits:
        train = set(range(20)) - set(v)
        assert len(train) == 16 and not train.intersection(v)
    # Asymmetric synthetic mask detects column-major/transpose/off-by-one errors.
    synthetic = np.zeros((SIZE, SIZE), dtype=np.uint8)
    synthetic[0, 0:3] = 1
    synthetic[1, 2:5] = 1
    synthetic[-1, -1] = 1
    cases = [np.zeros_like(synthetic), np.ones_like(synthetic), synthetic]
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        for i, a in enumerate(cases):
            p = Path(td)/f'{i}.png'
            Image.fromarray(a*255).save(p)
            assert encode(a) == official.turn_to_str([str(p)])[0]
            assert np.array_equal(a, decode(encode(a)))
    for invalid in ['0 1', '-1 1', '1 0', '1 3 2 1', '262144 2', '5', '10 1 1 1']:
        try:
            decode(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid RLE was accepted: {invalid}')
    save_json(ROOT/'data_manifest.json', dict(files=files, pixel_duplicates=duplicates,
              zip_sha256='6f6e36243092f4cf55c1be4ae7c5227e3c3686501e25487ada7ef03fa9b7634a',
              official_target_roundtrip_labels=20, synthetic_cases=3, invalid_cases=7,
              fold_coverage=20, valid=True))
    print('PASS: 20 label/exporter matches; no pixel duplicates; all 5 folds disjoint; 3 boundary/orientation cases; 7 invalid RLE cases.')


if __name__ == '__main__':
    main()
