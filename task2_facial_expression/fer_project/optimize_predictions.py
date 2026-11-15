"""Fixed-holdout TTA/feature-neighbor selection and auditable error analysis.

Run after run_experiments.py has completed. No label is edited. No test label
is used. All feature banks and checkpoint-dependent caches stay on D:.
"""
from fer import (ROOT, FER, Faces, CLASSES, LABELS, dump, seed_everything,
                 loader, metrics, torch, np, F, Image)
import argparse, base64, csv, hashlib, html, io, json, shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
MODES = ['single', 'flip', 'multiscale', 'multiscale_crop']
VIEWS = {'single': [(160, 0, False)],
         'flip': [(160, 0, False), (160, 0, True)],
         'multiscale': [(s, 0, f) for s in [144, 160, 176] for f in [False, True]],
         'multiscale_crop': [(s, 0, f) for s in [144, 160, 176] for f in [False, True]]
                            + [(160, 1, False), (160, 1, True)]}
MIN_GAIN = .0025  # Predeclared practical threshold, not a significance test.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path, rows, columns):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


class ViewFaces(Faces):
    def __init__(self, images, size, crop):
        super().__init__(images, None, size)
        self.crop = crop

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i]).convert('RGB')
        if self.crop:
            c = self.crop
            img = img.crop((c, c, img.width-c, img.height-c))
        return self.transform(img), i


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['label_mapping'] == LABELS
    assert checkpoint['config']['size'] == 160
    model = FER(checkpoint['config']['variant'], initialize_pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to('cuda', memory_format=torch.channels_last).eval()
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    data_digest = hashlib.sha256((ROOT/'data'/'split.json').read_bytes()).hexdigest()[:12]
    cache = ROOT/'analysis_cache'/f'{Path(path).parent.name}_{digest[:16]}_{data_digest}'
    cache.mkdir(parents=True, exist_ok=True)
    dump(cache/'provenance.json', {'checkpoint': str(path), 'sha256': digest,
                                 'epoch': checkpoint['epoch'], 'config': checkpoint['config']})
    return model, cache


@torch.inference_mode()
def views(model, images, cache, partition, modes=MODES):
    # Average probabilities, not hard votes. Each unique view is evaluated once.
    needed = list(dict.fromkeys(v for mode in modes for v in VIEWS[mode]))
    all_p = {}
    for size, crop, flip in needed:
        key = (size, crop, flip)
        file = cache/f'{partition}_s{size}_crop{crop}_flip{int(flip)}.npy'
        if file.exists():
            p = np.load(file)
        else:
            predictions = []
            for x, _ in loader(ViewFaces(images, size, crop), 128, False, 0):
                x = x.to('cuda', memory_format=torch.channels_last)
                if flip: x = x.flip(-1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                predictions.append(logits.float().softmax(1).cpu().numpy())
            p = np.concatenate(predictions)
            np.save(file, p)
        assert p.shape == (len(images), 6) and np.isfinite(p).all()
        assert np.allclose(p.sum(1), 1, atol=1e-5)
        all_p[key] = p
    return {mode: np.mean([all_p[v] for v in VIEWS[mode]], axis=0) for mode in modes}


@torch.inference_mode()
def features(model, images, cache, partition):
    file = cache/f'{partition}_features.npz'
    if file.exists():
        with np.load(file) as saved:
            return {k: saved[k].astype(np.float32) for k in saved.files}
    fused, global_only, local_only, probabilities = [], [], [], []
    captured = []; global_captured = []
    handle = model.local.register_forward_hook(lambda m, a, out: captured.append(out)) if model.local is not None else None
    global_handle = model.backbone.bn2.register_forward_hook(lambda m, a, out: global_captured.append(out.mean((2,3)))) if handle else None
    try:
        for x, _ in loader(Faces(images, None, 160), 128, False, 0):
            x = x.to('cuda', memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, f = model(x)
            fused.append(F.normalize(f.float(), dim=1).cpu().numpy())
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            if captured:
                local = captured.pop()
                global_only.append(F.normalize(global_captured.pop().float(), dim=1).cpu().numpy())
                local_only.append(F.normalize(local.float(), dim=1).cpu().numpy())
    finally:
        if handle: handle.remove()
        if global_handle: global_handle.remove()
    result = {'fused': np.concatenate(fused), 'probabilities': np.concatenate(probabilities)}
    if local_only:
        result.update({'global': np.concatenate(global_only), 'local': np.concatenate(local_only)})
    np.savez(file, **{k: (v if k == 'probabilities' else v.astype(np.float16)) for k, v in result.items()})
    # Quantize consistently on first run and reloads; normalize again in kNN.
    return {k: (v if k == 'probabilities' else v.astype(np.float16).astype(np.float32)) for k, v in result.items()}


@torch.inference_mode()
def neighbors(bank, bank_labels, query, k=15, temperature=.07):
    b = F.normalize(torch.from_numpy(bank).to('cuda'), dim=1)
    labels = torch.from_numpy(bank_labels).long().to('cuda')
    probs, indices, similarities = [], [], []
    for start in range(0, len(query), 256):
        q = F.normalize(torch.from_numpy(query[start:start+256]).to('cuda'), dim=1)
        score, idx = (q @ b.T).topk(min(k, len(bank)), dim=1)
        weight = (score/temperature).softmax(1)
        p = torch.zeros((len(q), 6), device='cuda')
        p.scatter_add_(1, labels[idx], weight)
        probs.append(p.cpu().numpy()); indices.append(idx.cpu().numpy()); similarities.append(score.cpu().numpy())
    return np.concatenate(probs), np.concatenate(indices), np.concatenate(similarities)


def thumbnail(arr):
    buffer = io.BytesIO(); Image.fromarray(arr).save(buffer, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def error_report(packed, val_p, train_data, val_data, nn_indices, nn_similarity, variant):
    folder = OUT/'error_analysis'; folder.mkdir(exist_ok=True)
    tr, va = packed['train_idx'], packed['val_idx']
    images, y, ids, groups = [packed[k] for k in ['train_images', 'labels', 'train_ids', 'groups']]
    pred = val_p.argmax(1); wrong = np.flatnonzero(pred != y[va])
    wrong = wrong[np.argsort(-val_p[wrong, pred[wrong]], kind='stable')]
    rows = []
    for j in wrong:
        nearest = tr[nn_indices[j, 0]]
        rows.append({'image': ids[va[j]], 'given_label': CLASSES[y[va[j]]], 'prediction': CLASSES[pred[j]],
                     'confidence': float(val_p[j, pred[j]]), 'given_label_probability': float(val_p[j, y[va[j]]]),
                     'nearest_train_image': ids[nearest], 'nearest_train_label': CLASSES[y[nearest]],
                     'cosine_similarity': float(nn_similarity[j, 0])})
    csv_write(folder/'validation_errors.csv', rows, list(rows[0]) if rows else ['image'])
    feats = train_data['fused']; feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    centers = np.stack([feats[y[tr] == c].mean(0) for c in range(6)])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    sim = feats@centers.T
    own = sim[np.arange(len(tr)), y[tr]].copy(); sim[np.arange(len(tr)), y[tr]] = -np.inf
    margin = own-sim.max(1)
    train_p = train_data['probabilities']; train_pred = train_p.argmax(1)
    confidence = train_p.max(1)
    suspects = np.flatnonzero((train_pred != y[tr]) & (confidence >= .9) & (margin < 0))
    suspects = suspects[np.argsort(-confidence[suspects], kind='stable')]
    suspect_rows = [{'image': ids[tr[j]], 'given_label': CLASSES[y[tr[j]]], 'prediction': CLASSES[train_pred[j]],
                     'confidence': float(confidence[j]), 'prototype_margin': float(margin[j]),
                     'status': 'review_candidate_not_confirmed_mislabel'} for j in suspects]
    csv_write(folder/'training_review_candidates.csv', suspect_rows,
              ['image', 'given_label', 'prediction', 'confidence', 'prototype_margin', 'status'])
    group_members = {}
    for i, group in enumerate(groups): group_members.setdefault(str(group), []).append(i)
    conflicts = []; train_set = set(tr.tolist())
    for group, members in group_members.items():
        if len(set(y[members].tolist())) > 1:
            conflicts.append({'pixel_sha256': group, 'partition': 'train' if members[0] in train_set else 'validation',
                              'members': [{'image': str(ids[i]), 'given_label': CLASSES[y[i]]} for i in members]})
    dump(folder/'duplicate_label_conflicts.json', conflicts)
    prototype_pred = (val_data['fused']@centers.T).argmax(1)
    report = {'validation': metrics(val_p, y[va]), 'validation_error_count': len(wrong),
              'training_review_candidates': len(suspects), 'duplicate_label_conflict_groups': len(conflicts),
              'feature_model': variant, 'feature_dimension': int(feats.shape[1]),
              'available_feature_types': [k for k in train_data if k != 'probabilities'],
              'prototype_only_validation_accuracy': float((prototype_pred == y[va]).mean()),
              'labels_changed': 0,
              'caveats': ['Model disagreement is not proof of a wrong label.',
                          'Training predictions are in-sample, used only for inspection.',
                          'Validation labels remain fixed; test labels are unavailable.',
                          'Exact-pixel grouping does not guarantee identity or near-duplicate disjointness.']}
    dump(folder/'summary.json', report)
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>表情误分类与特征邻居</title>',
               '<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;background:#f4f6f8;color:#172838}section{background:white;padding:20px;margin:16px 0;border-radius:12px}.row{display:flex;gap:20px}figure{margin:0;width:180px}img{width:144px;height:144px;image-rendering:auto}figcaption{font-size:13px;overflow-wrap:anywhere}h2{font-size:18px}p{line-height:1.7}</style>',
               '<h1>高置信误分类与相似训练样本</h1>',
               f'<p>以下按验证误分类置信度排序，展示前 {min(36,len(wrong))} 个样本及深度特征最近的三个训练样本。特征来源：{html.escape(variant)}，1280 维，余弦相似度。标签来自原始目录；模型预测冲突不等于标签错误。</p>']
    for j in wrong[:36]:
        index = va[j]
        content.append(f'<section><h2>给定：{CLASSES[y[index]]} → 预测：{CLASSES[pred[j]]}（{val_p[j,pred[j]]:.1%}）</h2><div class="row">')
        content.append(f'<figure><img src="{thumbnail(images[index])}"><figcaption>验证图<br>{html.escape(str(ids[index]))}</figcaption></figure>')
        for rank in range(3):
            near = tr[nn_indices[j, rank]]
            content.append(f'<figure><img src="{thumbnail(images[near])}"><figcaption>邻居 {rank+1}：{CLASSES[y[near]]}<br>cos={nn_similarity[j,rank]:.3f}<br>{html.escape(str(ids[near]))}</figcaption></figure>')
        content.append('</div></section>')
    content.append('</html>'); (folder/'error_gallery.html').write_text('\n'.join(content), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args(); seed_everything(42)
    selection = read(OUT/'model_selection.json'); assert selection['completed']
    packed = np.load(ROOT/'data'/'dataset.npz'); y = packed['labels']; tr = packed['train_idx']; va = packed['val_idx']
    assert not set(packed['groups'][tr]) & set(packed['groups'][va])
    val_by_model, knn_by_model = [], []
    details = None
    for variant in selection['selected_variants']:
        model, cache = load_model(ROOT/'runs'/f'{variant}_s42_w0'/'best.pt')
        val_views = views(model, packed['train_images'][va], cache, 'validation')
        previous = np.load(ROOT/'runs'/f'{variant}_s42_w0'/'validation_tta.npz')
        assert np.array_equal(previous['indices'], va) and np.array_equal(previous['labels'], y[va])
        if not np.allclose(val_views['flip'], previous['probabilities'], atol=2e-5):
            raise RuntimeError('Flip TTA differs from the original run; inspect preprocessing before selection')
        train_f = features(model, packed['train_images'][tr], cache, 'holdout_train')
        val_f = features(model, packed['train_images'][va], cache, 'validation')
        knn_p, idx, sim = neighbors(train_f['fused'], y[tr], val_f['fused'])
        if details is None: details = (train_f, val_f, idx, sim, variant)
        val_by_model.append(val_views); knn_by_model.append(knn_p)
        del model; torch.cuda.empty_cache()
        print(json.dumps({'event': 'validation_features_and_views_ready', 'variant': variant}), flush=True)
    p_by_mode = {mode: np.mean([m[mode] for m in val_by_model], axis=0) for mode in MODES}
    knn_p = np.mean(knn_by_model, axis=0)
    table = []
    for mode in MODES:
        score = metrics(p_by_mode[mode], y[va])
        table.append({'tta': mode, 'knn_weight': 0., **{k: score[k] for k in ['accuracy', 'macro_f1', 'nll']}})
    baseline = next(r for r in table if r['tta'] == 'flip')
    best_tta = max(table, key=lambda r: r['accuracy'])
    if best_tta['accuracy'] < baseline['accuracy']+MIN_GAIN: best_tta = baseline
    mode = best_tta['tta']; selected = best_tta.copy(); final_val = p_by_mode[mode]
    # A small, fixed grid. k and temperature are not tuned on validation.
    for alpha in [.1, .2]:
        p = (1-alpha)*p_by_mode[mode]+alpha*knn_p
        score = metrics(p, y[va])
        record = {'tta': mode, 'knn_weight': alpha, **{k: score[k] for k in ['accuracy', 'macro_f1', 'nll']}}
        table.append(record)
    best_blend = max(table[-2:], key=lambda r: r['accuracy'])
    if best_blend['accuracy'] >= best_tta['accuracy']+MIN_GAIN:
        selected = best_blend; a = selected['knn_weight']; final_val = (1-a)*p_by_mode[mode]+a*knn_p
    old_correct = p_by_mode['flip'].argmax(1) == y[va]; new_correct = final_val.argmax(1) == y[va]
    result = {'candidates': table, 'selected': selected, 'baseline': baseline,
              'k': 15, 'temperature': .07, 'minimum_gain_per_stage': MIN_GAIN,
              'validation_corrected': int((~old_correct & new_correct).sum()),
              'validation_new_errors': int((old_correct & ~new_correct).sum()),
              'feature_bank_validation_policy': 'Only holdout training rows; no validation or test rows in bank.',
              'selection_caveat': 'All choices use one holdout; these scores have model-selection bias and are not Kaggle scores.'}
    dump(OUT/'optimization_results.json', result)
    csv_write(OUT/'optimization_results.csv', table, list(table[0]))
    error_report(packed, final_val, *details)
    print(json.dumps({'event': 'optimization_selected', **result}), flush=True)
    if args.analysis_only: return
    test_probabilities = []
    for name in selection['final_runs']:
        model, cache = load_model(ROOT/'runs'/name/'best.pt')
        test_p = views(model, packed['test_images'], cache, 'test', [mode])[mode]
        a = selected['knn_weight']
        if a:
            # Full-data refit -> full training bank. A zero-epoch fallback still uses
            # the same frozen representation and all permitted training labels.
            train_f = features(model, packed['train_images'], cache, 'full_train')
            test_f = features(model, packed['test_images'], cache, 'test')
            test_knn, _, _ = neighbors(train_f['fused'], y, test_f['fused'])
            test_p = (1-a)*test_p+a*test_knn
        test_probabilities.append(test_p)
        del model; torch.cuda.empty_cache()
    final = np.mean(test_probabilities, axis=0)
    assert final.shape == (7065, 6) and np.isfinite(final).all() and np.allclose(final.sum(1), 1, atol=1e-5)
    for original, backup in [('submission.csv', 'submission_core.csv'), ('model_selection.json', 'model_selection_core.json')]:
        if not (OUT/backup).exists(): shutil.copy2(OUT/original, OUT/backup)
    np.savez(OUT/'final_probabilities.npz', probabilities=final, ids=packed['test_ids'])
    with (OUT/'submission.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['ID', 'Emotion']); writer.writerows(zip(packed['test_ids'], final.argmax(1)))
    selection['postprocessing'] = result
    selection['final_validation_accuracy'] = selected['accuracy']
    selection['final_validation_macro_f1'] = selected['macro_f1']
    dump(OUT/'model_selection.json', selection)
    print(json.dumps({'event': 'optimization_completed', 'submission': str(OUT/'submission.csv'), 'selected': selected}), flush=True)


if __name__ == '__main__': main()
