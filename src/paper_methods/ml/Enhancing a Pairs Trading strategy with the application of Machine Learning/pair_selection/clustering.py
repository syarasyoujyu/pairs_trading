"""Density-based clustering for PCA asset representations."""
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS


def optics_labels(
    features: pd.DataFrame,
    min_samples: int = 2,
    xi: float = 0.05,
    min_cluster_size: int | float | None = None,
) -> pd.Series:
    """Cluster PCA features with OPTICS, as proposed in the paper."""
    model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )
    labels = model.fit_predict(features.to_numpy(dtype=float))
    return pd.Series(labels, index=features.index, name="cluster")


def adaptive_density_labels(
    features: pd.DataFrame,
    min_samples: int = 2,
    radius_scale: float = 1.0,
) -> pd.Series:
    """Cluster features using each point's k-distance radius."""
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")

    names = list(features.index)
    x = features.to_numpy(dtype=float)
    n_obs = len(names)
    labels = np.full(n_obs, -1, dtype=int)

    if n_obs < min_samples:
        return pd.Series(labels, index=names, name="cluster")

    diff = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    kth = min(min_samples - 1, n_obs - 1)
    core_distances = np.sort(distances, axis=1)[:, kth]
    radii = np.maximum(core_distances[:, None], core_distances[None, :])
    adjacency = distances <= (radii * radius_scale)

    cluster_id = 0
    visited = np.zeros(n_obs, dtype=bool)
    for start in range(n_obs):
        if visited[start]:
            continue
        queue = deque([start])
        component: list[int] = []
        visited[start] = True

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in np.flatnonzero(adjacency[current]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        if len(component) >= min_samples:
            labels[component] = cluster_id
            cluster_id += 1

    return pd.Series(labels, index=names, name="cluster")


def density_cluster_labels(
    features: pd.DataFrame,
    method: str = "auto",
    min_samples: int = 2,
    xi: float = 0.05,
    min_cluster_size: int | float | None = None,
) -> pd.Series:
    """Return density-based cluster labels."""
    if method == "optics":
        return optics_labels(features, min_samples, xi, min_cluster_size)
    if method == "adaptive":
        return adaptive_density_labels(features, min_samples)
    if method == "auto":
        return optics_labels(features, min_samples, xi, min_cluster_size)
    raise ValueError("method must be 'auto', 'optics', or 'adaptive'.")


def cluster_members(labels: pd.Series) -> dict[int, list[str]]:
    """Group asset names by non-noise cluster label."""
    members: dict[int, list[str]] = {}
    for asset, label in labels.items():
        label_int = int(label)
        if label_int < 0:
            continue
        members.setdefault(label_int, []).append(str(asset))
    return members


def pair_candidates_from_clusters(
    labels: pd.Series,
    features: pd.DataFrame | None = None,
    max_pairs_per_cluster: int | None = None,
) -> list[tuple[str, str]]:
    """Create pair candidates from assets assigned to the same cluster."""
    candidates: list[tuple[str, str]] = []
    for assets in cluster_members(labels).values():
        cluster_pairs = [(left, right) for left, right in combinations(sorted(assets), 2)]
        if features is not None:
            cluster_pairs.sort(key=lambda pair: _feature_distance(features, pair[0], pair[1]))
        if max_pairs_per_cluster is not None:
            cluster_pairs = cluster_pairs[:max_pairs_per_cluster]
        candidates.extend(cluster_pairs)
    return candidates


def _feature_distance(features: pd.DataFrame, left: str, right: str) -> float:
    left_values = features.loc[left].to_numpy(dtype=float)
    right_values = features.loc[right].to_numpy(dtype=float)
    return float(np.sqrt(np.sum((left_values - right_values) ** 2)))
