import numpy as np
import pytest
from experiments.learned_metric_v3 import fit_projection, transform, development_metrics


def test_train_only_projection_is_finite_and_fixed():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 30))
    labels = np.repeat(np.arange(10), 2)
    mean, projection = fit_projection(x, labels, 8, .3)
    before = projection.copy()
    vectors = transform(rng.normal(size=(3, 30)), mean, projection)
    assert vectors.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(projection, before)


def test_development_rejects_test_rows():
    with pytest.raises(ValueError, match="test"):
        development_metrics(np.eye(2), [{"split": "test"}, {"split": "train"}])
