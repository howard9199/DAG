import numpy as np

from dag.data.targets import build_target_mask, generate_mapping, load_shape_mask


def test_mapping_never_maps_to_self_or_background():
    rng = np.random.default_rng(0)
    for _ in range(50):
        gt = rng.choice(np.arange(1, 21), size=rng.integers(1, 10), replace=False)
        mapping, _ = generate_mapping(gt, rng=rng)
        for g in gt:
            assert mapping[g] != g, f"target equals source for class {g}"
            assert mapping[g] != 0, f"target is background for class {g}"


def test_mapping_returns_zero_for_absent_classes():
    mapping, _ = generate_mapping(np.array([3, 7]))
    untouched = set(range(21)) - {3, 7}
    for c in untouched:
        assert mapping[c] == 0


def test_shape_masks_load():
    for shape in ("circle", "square", "strip"):
        m = load_shape_mask(shape)
        assert m.shape == (500, 500)
        assert (m >= 0).all()


def test_build_target_mask_uses_resize():
    gt = np.zeros((100, 200), dtype=np.int64)
    gt[20:80, 50:150] = 5
    rng = np.random.default_rng(0)
    out = build_target_mask("square", gt, target_size=(100, 200), rng=rng)
    assert out.shape == (100, 200)
    assert out.dtype == np.int64
    assert (out >= 0).all() and (out <= 20).all()
    # Square should produce at least one non-zero target pixel.
    assert (out != 0).any()
