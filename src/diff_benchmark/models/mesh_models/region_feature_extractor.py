from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


class RegionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert mesh dicts into fixed-width region-level feature vectors.

    Supported region representations:
    - ``flatten``
    - ``mean_std``
    - ``pca``
    - ``summary_stats``
    - ``percentiles``
    """

    VALID_REPRESENTATIONS = {
        "flatten",
        "mean_std",
        "pca",
        "summary_stats",
        "percentiles",
    }

    def __init__(
        self,
        region_representation: str = "flatten",
        pca_n_components: int = 3,
    ):
        self.region_representation = region_representation
        self.pca_n_components = pca_n_components

    @staticmethod
    def _to_numpy(mesh):
        pl = mesh["parcel_labels"]
        nf = mesh["node_features"]

        if hasattr(pl, "numpy"):
            pl = pl.numpy()
        if hasattr(nf, "numpy"):
            nf = nf.numpy()

        if nf.ndim == 1:
            nf = nf[:, np.newaxis]

        return nf, pl

    @staticmethod
    def _summary_stats(region_nodes: np.ndarray) -> np.ndarray:
        mean = region_nodes.mean(axis=0)
        std = region_nodes.std(axis=0)
        rmin = region_nodes.min(axis=0)
        rmax = region_nodes.max(axis=0)

        centered = region_nodes - mean
        m2 = np.mean(centered**2, axis=0)
        m3 = np.mean(centered**3, axis=0)
        m4 = np.mean(centered**4, axis=0)
        eps = 1e-12
        skew = m3 / np.power(m2 + eps, 1.5)
        kurtosis = m4 / np.power(m2 + eps, 2.0) - 3.0

        return np.concatenate([mean, std, rmin, rmax, skew, kurtosis])

    @staticmethod
    def _percentile_stats(region_nodes: np.ndarray) -> np.ndarray:
        percentiles = np.percentile(region_nodes, [10, 25, 50, 75, 90], axis=0)
        return percentiles.reshape(-1)

    def fit(self, X, y=None):
        if self.region_representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                "region_representation must be one of "
                "{'flatten', 'mean_std', 'pca', 'summary_stats', 'percentiles'}"
            )

        mesh = X[0]
        nf, pl = self._to_numpy(mesh)

        self.n_node_features_ = nf.shape[1]
        self.region_order_ = sorted(np.unique(pl))
        self.region_order_ = [r for r in self.region_order_ if r != 0]

        self.region_sizes_ = {}
        self.region_feature_widths_ = {}

        if self.region_representation == "pca":
            if self.pca_n_components < 1:
                raise ValueError("pca_n_components must be >= 1")

            parcel_min_nodes = {r: np.inf for r in self.region_order_}
            self.scaler_per_region_ = {}
            self.pca_per_region_ = {}
            self.n_components_per_region_ = {}

            for mesh in X:
                nf_subj, pl_subj = self._to_numpy(mesh)
                for r in self.region_order_:
                    n_nodes = int((pl_subj == r).sum())
                    if n_nodes > 0:
                        parcel_min_nodes[r] = min(parcel_min_nodes[r], n_nodes)

            for r in self.region_order_:
                min_nodes = int(parcel_min_nodes[r])
                self.region_sizes_[r] = min_nodes
                k = min(self.pca_n_components, min_nodes, self.n_node_features_)
                self.n_components_per_region_[r] = k
                self.scaler_per_region_[r] = StandardScaler(copy=False)
                self.pca_per_region_[r] = IncrementalPCA(n_components=k)
                self.region_feature_widths_[r] = k

            for mesh in X:
                nf_subj, pl_subj = self._to_numpy(mesh)
                for r in self.region_order_:
                    mask = pl_subj == r
                    if mask.sum() == 0:
                        continue
                    self.scaler_per_region_[r].partial_fit(nf_subj[mask])

            for mesh in X:
                nf_subj, pl_subj = self._to_numpy(mesh)
                for r in self.region_order_:
                    pca = self.pca_per_region_[r]
                    mask = pl_subj == r
                    if mask.sum() < pca.n_components:
                        continue
                    region_nodes = self.scaler_per_region_[r].transform(nf_subj[mask])
                    pca.partial_fit(region_nodes)
        else:
            for r in self.region_order_:
                mask = pl == r
                self.region_sizes_[r] = mask.sum()
                if self.region_representation == "flatten":
                    self.region_feature_widths_[r] = (
                        self.region_sizes_[r] * self.n_node_features_
                    )
                elif self.region_representation == "mean_std":
                    self.region_feature_widths_[r] = 2 * self.n_node_features_
                elif self.region_representation == "summary_stats":
                    self.region_feature_widths_[r] = 6 * self.n_node_features_
                else:
                    self.region_feature_widths_[r] = 5 * self.n_node_features_

        return self

    def transform(self, X):
        features = []

        for mesh in X:
            nf, pl = self._to_numpy(mesh)
            subj_feat = []

            for r in self.region_order_:
                mask = pl == r
                region_nodes = nf[mask]
                if region_nodes.shape[0] == 0:
                    subj_feat.append(
                        np.zeros(self.region_feature_widths_[r], dtype=np.float32)
                    )
                    continue

                if self.region_representation == "flatten":
                    subj_feat.append(region_nodes.reshape(-1))
                elif self.region_representation == "mean_std":
                    region_mean = region_nodes.mean(axis=0)
                    region_std = region_nodes.std(axis=0)
                    subj_feat.append(np.concatenate([region_mean, region_std]))
                elif self.region_representation == "summary_stats":
                    subj_feat.append(self._summary_stats(region_nodes))
                elif self.region_representation == "percentiles":
                    subj_feat.append(self._percentile_stats(region_nodes))
                else:
                    region_nodes = self.scaler_per_region_[r].transform(region_nodes)
                    projected = self.pca_per_region_[r].transform(region_nodes)
                    ev = (
                        projected.var(axis=0).astype(np.float32, copy=False)
                        if projected.shape[0] > 1
                        else np.abs(projected[0]).astype(np.float32, copy=False)
                    )
                    subj_feat.append(ev)

            features.append(np.concatenate(subj_feat))

        return np.vstack(features)


# Backward-compatible alias for previous class name usage.
RegionFeatureTransformer = RegionFeatureExtractor
