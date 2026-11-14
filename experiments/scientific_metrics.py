"""Validated biometric metrics; similarity = negative Euclidean distance.

The service database returns squared L2: score = -sqrt(d2), not a probability.
Test-set EER/AUC describe score distributions. Operating thresholds must be
selected on independent calibration data and then frozen.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, roc_curve


@dataclass(frozen=True)
class VerificationMetrics:
    eer: float
    eer_threshold: float
    auc: float
    far: float
    frr: float
    threshold: float
    genuine_pairs: int
    impostor_pairs: int
    threshold_source: str
    eer_threshold_kind: str = "nearest_empirical_point_not_interpolated_EER"


def _as_2d_float32(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a nonempty finite [N, D] array")
    return x


def _labels(labels, size):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != size:
        raise ValueError("Labels must be one-dimensional and match embeddings")
    if any(label is None or str(label).strip() == "" or
           (isinstance(label, (float, np.floating)) and not np.isfinite(label)) for label in labels):
        raise ValueError("Identity labels cannot be empty")
    return labels


def _validated_pairs(y_true, scores, sample_weight=None):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("Pair labels and scores must be nonempty 1-D arrays of equal length")
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Scores must be finite and pair labels must be binary")
    weight = None
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != y.shape or not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError("Invalid sample weights")
    for cls in (0, 1):
        count = np.sum(y == cls) if weight is None else weight[y == cls].sum()
        if count <= 0:
            raise ValueError("Both genuine and impostor pairs are required")
    return y.astype(np.int8), s, weight


def _pairwise_scores_neg_l2(embeddings):
    x = _as_2d_float32(embeddings)
    return -cdist(x, x, metric="euclidean")


def _verification_pairs(scores_mat, labels):
    labels = _labels(labels, len(scores_mat))
    if np.shape(scores_mat) != (len(labels), len(labels)):
        raise ValueError("Score matrix and labels do not match")
    i, j = np.triu_indices(len(labels), k=1)
    return (labels[i] == labels[j]).astype(np.int8), np.asarray(scores_mat)[i, j]


def cross_session_pairs(gallery, gallery_labels, probes, probe_labels, max_pairs=2_000_000):
    """All gallery/probe pairs; caller verifies image/session disjointness."""
    g, p = _as_2d_float32(gallery), _as_2d_float32(probes)
    gl, pl = _labels(gallery_labels, len(g)), _labels(probe_labels, len(p))
    if g.shape[1] != p.shape[1]:
        raise ValueError("Gallery/probe feature dimensions differ")
    if len(g) * len(p) > max_pairs:
        raise ValueError(f"Pair budget exceeded: {len(g) * len(p)} > {max_pairs}; predefine a smaller protocol")
    scores = -cdist(p, g, metric="euclidean")
    y = (pl[:, None] == gl[None, :]).astype(np.int8)
    return y.ravel(), scores.ravel(), np.repeat(pl, len(g)), np.tile(gl, len(p)), scores


def compute_eer(y_true, scores, sample_weight=None):
    """Linear ROC-interpolated EER and nearest empirical threshold.

    With discrete/tied scores an exact EER operating threshold often does not
    exist. The returned threshold is not claimed to attain the interpolated EER.
    """
    y, scores, weight = _validated_pairs(y_true, scores, sample_weight)
    fpr, tpr, thresholds = roc_curve(y, scores, sample_weight=weight, drop_intermediate=False)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0 or right == 0:
        eer = float(fpr[right])
    else:
        left = right - 1
        fraction = -difference[left] / (difference[right] - difference[left])
        eer = float(fpr[left] + fraction * (fpr[right] - fpr[left]))
    idx = int(np.argmin(np.abs(difference)))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(np.max(scores), np.inf))
    return eer, threshold


def compute_far_frr(y_true, scores, threshold, sample_weight=None):
    """FAR and FRR for fixed rule accept iff similarity >= threshold."""
    y, scores, weight = _validated_pairs(y_true, scores, sample_weight)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    weight = np.ones(len(y)) if weight is None else weight
    accept = scores >= threshold
    far = weight[(y == 0) & accept].sum() / weight[y == 0].sum()
    frr = weight[(y == 1) & ~accept].sum() / weight[y == 1].sum()
    return float(far), float(frr)


def select_threshold_at_far(y_true, scores, target_far=0.01):
    """Most permissive empirical threshold meeting calibration FAR <= target."""
    y, scores, _ = _validated_pairs(y_true, scores)
    if not 0 <= target_far < 1:
        raise ValueError("target_far must be in [0, 1)")
    far, _, thresholds = roc_curve(y, scores, drop_intermediate=False)
    idx = int(np.flatnonzero(far <= target_far)[-1])
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(np.max(scores), np.inf))
    return threshold


def verification_from_scores(y_true, scores, threshold=None, threshold_source="calibration", sample_weight=None):
    y, scores, weight = _validated_pairs(y_true, scores, sample_weight)
    eer, eer_threshold = compute_eer(y, scores, weight)
    fpr, tpr, _ = roc_curve(y, scores, sample_weight=weight, drop_intermediate=False)
    if threshold is None:
        threshold = eer_threshold
        threshold_source = "same_set_descriptive_only"
    far, frr = compute_far_frr(y, scores, threshold, weight)
    return VerificationMetrics(
        eer, eer_threshold, float(auc(fpr, tpr)), far, frr, float(threshold),
        int(np.sum(y == 1)), int(np.sum(y == 0)), threshold_source,
    )


def compute_verification_metrics(embeddings, labels, threshold=None):
    """Compatibility all-pairs metric; diagonal/self comparisons are excluded."""
    y, scores = _verification_pairs(_pairwise_scores_neg_l2(embeddings), labels)
    return verification_from_scores(y, scores, threshold)


def cmc_from_gallery(scores, gallery_labels, probe_labels, max_rank=10):
    """Identity-level closed-set CMC, with conservative worst-case ties.

    Multiple gallery templates per identity are pooled by maximum similarity.
    Unknown probe identities are rejected instead of silently dropped.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite nonempty [probes, gallery] matrix")
    gl, pl = _labels(gallery_labels, scores.shape[1]), _labels(probe_labels, scores.shape[0])
    if not isinstance(max_rank, (int, np.integer)) or max_rank < 1:
        raise ValueError("max_rank must be a positive integer")
    identities = np.unique(gl)
    if not np.isin(pl, identities).all():
        raise ValueError("Closed-set CMC requires a gallery match for every probe")
    identity_scores = np.stack([scores[:, gl == identity].max(axis=1) for identity in identities], axis=1)
    ranks = []
    for row, identity in zip(identity_scores, pl):
        truth = identities == identity
        ranks.append(1 + int(np.sum(row[~truth] >= row[truth][0])))
    ranks = np.asarray(ranks)
    return np.asarray([np.mean(ranks <= rank) for rank in range(1, max_rank + 1)])


def compute_cmc(embeddings, labels, max_rank=10):
    """Legacy leave-one-out CMC; prefer explicit cross-session gallery/probes."""
    x = _as_2d_float32(embeddings)
    labels = _labels(labels, len(x))
    if any(np.sum(labels == label) < 2 for label in np.unique(labels)):
        raise ValueError("Leave-one-out CMC needs >=2 samples per identity")
    scores = _pairwise_scores_neg_l2(x)
    cmcs = []
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        cmcs.append(cmc_from_gallery(scores[i:i+1, mask], labels[mask], labels[i:i+1], max_rank))
    return np.mean(cmcs, axis=0)


def bootstrap_identity_intervals(y_true, scores, probe_ids, gallery_ids, threshold, n_bootstrap=1000, seed=42):
    """Subject-resampling percentile intervals conditional on fitted model/threshold.

    Genuine weights use identity multiplicity; impostor weights use the product of
    the two identity multiplicities. No duplicate identity becomes an impostor.
    These are not pair-independent intervals and exclude training/calibration
    uncertainty. At small sample sizes or zero observed errors they can degenerate;
    they do not establish a population upper bound on FAR.
    """
    y, scores, _ = _validated_pairs(y_true, scores)
    probes, gallery = _labels(probe_ids, len(y)), _labels(gallery_ids, len(y))
    if not np.array_equal(y.astype(bool), probes == gallery):
        raise ValueError("Pair identities do not match genuine/impostor labels")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    identities = np.unique(np.concatenate([probes, gallery]))
    if len(identities) < 2:
        raise ValueError("Bootstrap requires at least two identities")
    pi, gi = np.searchsorted(identities, probes), np.searchsorted(identities, gallery)
    rng = np.random.default_rng(seed)
    samples, attempts = [], 0
    while len(samples) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        multiplicities = np.bincount(rng.integers(len(identities), size=len(identities)), minlength=len(identities))
        weight = np.where(y == 1, multiplicities[pi], multiplicities[pi] * multiplicities[gi])
        if not all(weight[y == cls].sum() > 0 for cls in (0, 1)):
            continue
        metrics = verification_from_scores(y, scores, threshold, sample_weight=weight)
        samples.append([metrics.eer, metrics.auc, metrics.far, metrics.frr])
    if len(samples) < n_bootstrap:
        raise ValueError("Unable to produce enough valid identity bootstrap replicates")
    bounds = np.percentile(np.asarray(samples), [2.5, 97.5], axis=0)
    return {
        "method": "identity_resampling_percentile_95",
        "conditional_on": "fitted_model_and_frozen_calibration_threshold",
        "replicates": n_bootstrap, "seed": seed, "identity_count": len(identities),
        "intervals": {name: [float(bounds[0, i]), float(bounds[1, i])]
                      for i, name in enumerate(("eer", "auc", "far", "frr"))},
    }


def evaluate_hypothesis_1(embeddings_lgwg, embeddings_simple, labels, cmc_max_rank=5):
    """Compatibility descriptive comparison; does not establish a hypothesis."""
    result = {"status": "descriptive_same_set_comparison_not_confirmatory"}
    for name, embeddings in (("LGLG_WPCA", embeddings_lgwg), ("Simple_Gabor", embeddings_simple)):
        metric = compute_verification_metrics(embeddings, labels)
        result[name] = {"EER": metric.eer, "AUC": metric.auc, "FAR": metric.far, "FRR": metric.frr,
                        "CMC": compute_cmc(embeddings, labels, cmc_max_rank).tolist()}
    return result


def evaluate_hypothesis_3(embeddings_local, embeddings_global, labels):
    """Pair separation is not stability; robustness needs controlled conditions."""
    result = {"status": "descriptive_separation_not_robustness"}
    for name, embeddings in (("Local_Patches", embeddings_local), ("Global_Face", embeddings_global)):
        y, scores = _verification_pairs(_pairwise_scores_neg_l2(embeddings), labels)
        _validated_pairs(y, scores)
        result[name] = {"mean_genuine_minus_impostor_score": float(scores[y == 1].mean() - scores[y == 0].mean())}
    return result

