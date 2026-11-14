"""Train-only projections of fixed Gabor response statistics.

NOT reproductions of LGLG-WPCA (Learning Gabor Log-Euclidean Gaussian with
Whitening PCA) or a published multi-fold filter convolution architecture.
Historical names remain aliases for existing clients.
"""
import warnings
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning

def _responses(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected finite, nonempty Gabor responses [H, W, filters]")
    return array.reshape(-1, array.shape[-1])

def moment_statistics(array):
    values = _responses(array)
    mean, std = values.mean(axis=0), values.std(axis=0)
    centered = np.divide(values - mean, std, out=np.zeros_like(values), where=std > 1e-12)
    skew = np.mean(centered ** 3, axis=0)
    kurtosis = np.where(std > 1e-12, np.mean(centered ** 4, axis=0) - 3, 0)
    return np.stack([mean, std, skew, kurtosis], axis=1).ravel()

def summary_statistics(array):
    values = _responses(array)
    return np.stack([values.mean(axis=0), values.std(axis=0), values.min(axis=0),
                     values.max(axis=0), np.median(values, axis=0)], axis=1).ravel()

def _training_matrix(responses, summary):
    rows = [summary(item) for item in responses]
    if len(rows) < 2:
        raise ValueError("Projection fitting requires at least two training patches")
    if len({row.size for row in rows}) != 1:
        raise ValueError("All training patches must use the same filter bank")
    matrix = np.stack(rows)
    if not np.any(np.var(matrix, axis=0) > 1e-15):
        raise ValueError("Training statistics have zero variance")
    return matrix

class GaborStatisticsPCA:
    """Inverse-variance feature weighting followed by ordinary centered PCA.

    Weights and axes are learned on training patches. Outputs are zero padded
    if the training sample count limits the number of fitted axes.
    """
    def __init__(self, n_components=128, alpha=0.1):
        if int(n_components) != n_components or n_components < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("n_components must be positive; alpha must be finite and positive")
        self.n_components, self.alpha = int(n_components), float(alpha)
        self.pca, self.weights, self.is_fitted = None, None, False

    def fit(self, gabor_responses_list):
        matrix = _training_matrix(gabor_responses_list, moment_statistics)
        return self.fit_statistics(matrix)

    def fit_statistics(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all() or not np.any(matrix.var(axis=0) > 1e-15):
            raise ValueError("Finite, varying statistics from at least two training patches are required")
        weights = self._compute_weights(matrix)
        weighted = matrix * np.sqrt(weights)[None, :]
        dimensions = min(self.n_components, len(matrix) - 1, matrix.shape[1])
        pca = PCA(n_components=dimensions, svd_solver="full")
        pca.fit(weighted)
        self.weights, self.pca, self.is_fitted = weights, pca, True
        return self

    def transform(self, gabor_responses):
        if not self.is_fitted:
            raise ValueError("Fit GaborStatisticsPCA on training data before transform")
        row = moment_statistics(gabor_responses)
        if row.size != self.weights.size:
            raise ValueError("Filter bank differs from the fitted feature space")
        projected = (row * np.sqrt(self.weights) - self.pca.mean_) @ self.pca.components_.T
        return np.pad(projected, (0, self.n_components - projected.size)).astype(np.float32)

    def _compute_weights(self, matrix):
        weights = 1.0 / (np.var(matrix, axis=0) + self.alpha)
        return weights / weights.mean()

class GaborPCAICA:
    """PCA of response summaries plus ICA of retained PCA scores."""
    def __init__(self, n_pca_components=64, n_ica_components=32):
        if any(int(v) != v or v < 1 for v in (n_pca_components, n_ica_components)):
            raise ValueError("PCA and ICA component counts must be positive integers")
        self.n_pca_components, self.n_ica_components = int(n_pca_components), int(n_ica_components)
        self.pca, self.ica, self.is_fitted = None, None, False

    def fit(self, gabor_responses_list):
        matrix = _training_matrix(gabor_responses_list, summary_statistics)
        return self.fit_statistics(matrix)

    def fit_statistics(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all() or not np.any(matrix.var(axis=0) > 1e-15):
            raise ValueError("Finite, varying statistics from at least two training patches are required")
        dimensions = min(self.n_pca_components, len(matrix) - 1, matrix.shape[1])
        pca = PCA(n_components=dimensions, svd_solver="full")
        scores = pca.fit_transform(matrix)
        valid_rank = int(np.sum(pca.explained_variance_ > max(pca.explained_variance_[0] * 1e-10, 1e-15)))
        if valid_rank < 1:
            raise ValueError("Training data have no usable ICA directions")
        ica = FastICA(n_components=min(self.n_ica_components, valid_rank), random_state=42,
                      whiten="unit-variance", max_iter=2000, tol=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                ica.fit(scores[:, :valid_rank])
            except ConvergenceWarning as error:
                raise ValueError("ICA failed to converge; reduce components or disable ICA") from error
        self.ica_input_dim = valid_rank
        self.pca, self.ica, self.is_fitted = pca, ica, True
        return self

    def transform(self, gabor_responses):
        if not self.is_fitted:
            raise ValueError("Fit GaborPCAICA on training data before transform")
        row = summary_statistics(gabor_responses)
        pca_scores = ((row - self.pca.mean_) @ self.pca.components_.T)[None, :]
        ica_scores = (pca_scores[0, :self.ica_input_dim] - self.ica.mean_) @ self.ica.components_.T
        return np.concatenate([np.pad(pca_scores[0], (0, self.n_pca_components - pca_scores.shape[1])),
                               np.pad(ica_scores, (0, self.n_ica_components - ica_scores.size))]).astype(np.float32)

LGLGWPCA = GaborStatisticsPCA
MultiFoldGaborPCA = GaborPCAICA

