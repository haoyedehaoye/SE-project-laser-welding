"""
robot_monitor v0.7 - Detection methods library

Unsupervised anomaly detectors operating on the same 32-frame window
features as XGBoostQualityChecker. Each detector implements:
    fit(X)   -> train on normal windows (where applicable)
    score(x) -> anomaly probability in [0, 1]

All sklearn-based detectors normalize their decision function with a
sigmoid, so scores are comparable across methods.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

THRESHOLD = 0.5


def _sigmoid(x) -> float:
    return float(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float))))


class ZScoreDetector:
    """Statistical: max absolute z-score of the window vs. normal baseline."""

    name = "Z-Score 统计"

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self._mean = None
        self._std = None

    def fit(self, X):
        self._mean = np.asarray(X, dtype=float).mean(axis=0)
        self._std = np.asarray(X, dtype=float).std(axis=0) + 1e-6
        return self

    def score(self, x) -> float:
        z = float(np.max(np.abs((np.asarray(x, dtype=float) - self._mean) / self._std)))
        return _sigmoid(z - self.z_threshold)


class IsolationForestDetector:
    """Isolation Forest: isolates anomalies with random splits."""

    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X):
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def score(self, x) -> float:
        d = float(self.model.decision_function(np.asarray([x], dtype=float))[0])
        return _sigmoid(-d)


class LocalOutlierFactorDetector:
    """LOF: local density outlier factor (novelty mode)."""

    name = "Local Outlier Factor"

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )

    def fit(self, X):
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def score(self, x) -> float:
        d = float(self.model.decision_function(np.asarray([x], dtype=float))[0])
        return _sigmoid(-d)


class OneClassSVMDetector:
    """One-Class SVM: learns a boundary around normal windows."""

    name = "One-Class SVM"

    def __init__(self, nu: float = 0.1):
        self.model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")

    def fit(self, X):
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def score(self, x) -> float:
        d = float(self.model.decision_function(np.asarray([x], dtype=float))[0])
        return _sigmoid(-d)


class PCAMahalanobisDetector:
    """PCA whitening + Mahalanobis distance in the reduced space."""

    name = "PCA + 马氏距离"

    def __init__(self, n_components: int = 5, random_state: int = 42):
        self.pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
        self._mean = None
        self._inv = None
        self._dist_mean = None
        self._dist_std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        Z = self.pca.fit_transform(X)
        self._mean = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        self._inv = np.linalg.inv(cov)
        dists = np.asarray([self._mahalanobis(z) for z in Z])
        self._dist_mean = float(dists.mean())
        self._dist_std = float(dists.std()) + 1e-6
        return self

    def _mahalanobis(self, z) -> float:
        diff = z - self._mean
        return float(np.sqrt(diff @ self._inv @ diff))

    def score(self, x) -> float:
        z = self.pca.transform(np.asarray([x], dtype=float))[0]
        d = self._mahalanobis(z)
        return _sigmoid((d - self._dist_mean) / (self._dist_std * 2.0))


class GMMDetector:
    """Gaussian Mixture density estimation (low likelihood -> anomaly)."""

    name = "Gaussian Mixture"

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=random_state
        )

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.gmm.fit(X)
        scores = self.gmm.score_samples(X)
        self._mean = float(scores.mean())
        self._std = float(scores.std()) + 1e-6
        return self

    def score(self, x) -> float:
        s = float(self.gmm.score_samples(np.asarray([x], dtype=float))[0])
        return _sigmoid((self._mean - s) / (self._std * 2.0))


class AutoencoderDetector:
    """MLP autoencoder: reconstruction error -> anomaly score."""

    name = "MLP Autoencoder"

    def __init__(self, hidden=(16, 8, 16), max_iter: int = 800, random_state: int = 42):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=20,
        )

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.model.fit(X, X)
        err = np.mean((self.model.predict(X) - X) ** 2, axis=1)
        self._mean = float(err.mean())
        self._std = float(err.std()) + 1e-6
        return self

    def score(self, x) -> float:
        x = np.asarray([x], dtype=float)
        e = float(np.mean((self.model.predict(x) - x) ** 2))
        return _sigmoid((e - self._mean) / (self._std * 2.0))


DETECTOR_CLASSES = [
    ZScoreDetector,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    PCAMahalanobisDetector,
    GMMDetector,
    AutoencoderDetector,
]
