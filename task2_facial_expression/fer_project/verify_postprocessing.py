"""Small real-checkpoint smoke test plus exact synthetic feature-neighbor test."""
from optimize_predictions import *


def main():
    seed_everything(42)
    data = np.load(ROOT/'data'/'dataset.npz'); y = data['labels']
    indices = np.concatenate([data['val_idx'][y[data['val_idx']] == c][:2] for c in range(6)])
    model, _ = load_model(ROOT/'runs'/'baseline_s42_w0'/'best.pt')
    cache = ROOT/'analysis_cache'/'postprocessing_smoke'; cache.mkdir(parents=True, exist_ok=True)
    # Run-specific cache location: never reuse a smoke result from an older checkpoint.
    cache = cache/str(__import__('time').time_ns()); cache.mkdir()
    images = data['train_images'][indices]
    p = views(model, images, cache, 'smoke')
    f = features(model, images, cache, 'smoke')
    assert f['fused'].shape == (12, 1280)
    assert np.allclose(p['single'], f['probabilities'], atol=1e-6)
    assert all(v.shape == (12, 6) and np.isfinite(v).all() for v in p.values())
    eye = np.eye(6, dtype=np.float32)
    knn, idx, sim = neighbors(eye, np.arange(6), eye, k=1)
    assert np.array_equal(knn.argmax(1), np.arange(6))
    assert np.array_equal(idx[:, 0], np.arange(6)) and np.allclose(sim[:, 0], 1)
    # k > 1 must retain normalized distributions and class ordering.
    knn, _, _ = neighbors(np.repeat(eye, 3, axis=0), np.repeat(np.arange(6), 3), eye, k=3)
    assert np.allclose(knn, eye)
    cached_f = features(model, images, cache, 'smoke')
    assert np.array_equal(f['fused'], cached_f['fused'])
    del model; torch.cuda.empty_cache()
    local_model = FER('local').to('cuda', memory_format=torch.channels_last).eval()
    local_p = views(local_model, images, cache, 'local_smoke')
    local_f = features(local_model, images, cache, 'local_smoke')
    assert set(local_f) == {'fused', 'global', 'local', 'probabilities'}
    assert all(local_f[k].shape == (12, 1280) for k in ['fused', 'global', 'local'])
    assert np.allclose(local_p['single'], local_f['probabilities'], atol=1e-6)
    result = {'real_checkpoint_views': list(p), 'images': len(images), 'features_shape': list(f['fused'].shape),
              'native_probabilities_match_feature_forward': True, 'synthetic_knn_class_mapping_correct': True,
              'feature_cache_reload_identical': True, 'local_branch_all_scales_valid': True,
              'global_and_local_features_extracted_separately': True, 'status': 'passed'}
    dump(OUT/'postprocessing_verification.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
