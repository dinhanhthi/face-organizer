"""DBSCAN clustering and label remapping for face embeddings."""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_embeddings(
    embeddings: list[np.ndarray],
    eps: float,
    min_samples: int = 2,
    metric: str = "euclidean",
) -> list[int]:
    """Cluster face embeddings using DBSCAN.

    Args:
        embeddings: List of face embedding arrays.
        eps: Maximum distance between two samples to be considered neighbors.
        min_samples: Minimum number of samples to form a core point.
        metric: Distance metric — "euclidean" for dlib (128-D), "cosine" for
            ArcFace/insightface (512-D unit-normalized vectors).

    Returns:
        List of integer cluster labels. Label -1 means outlier/unknown.
    """
    if not embeddings:
        return []

    matrix = np.array(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    db.fit(matrix)
    return db.labels_.tolist()


def embedding_distance_stats(
    embeddings: list[np.ndarray], metric: str = "euclidean"
) -> dict[str, float]:
    """Compute pairwise distance percentiles across all embeddings.

    Useful for choosing a good --eps value: the ideal eps sits just below the
    natural gap between same-person distances (typically p25) and
    different-person distances (typically p75).

    Args:
        embeddings: List of embedding arrays.
        metric: "euclidean" or "cosine".
    """
    if len(embeddings) < 2:
        return {}
    matrix = np.array(embeddings)
    distances = []
    for i in range(len(matrix)):
        others = matrix[i + 1 :]
        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            norm_i = matrix[i] / (np.linalg.norm(matrix[i]) + 1e-10)
            norm_others = others / (np.linalg.norm(others, axis=1, keepdims=True) + 1e-10)
            dists = 1.0 - (norm_others @ norm_i)
        else:
            diffs = others - matrix[i]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
        distances.append(dists)
    all_distances = np.concatenate(distances)
    return {
        "min": float(np.min(all_distances)),
        "p25": float(np.percentile(all_distances, 25)),
        "median": float(np.median(all_distances)),
        "p75": float(np.percentile(all_distances, 75)),
        "max": float(np.max(all_distances)),
    }


def relabel_by_frequency(labels: list[int]) -> dict[int, int]:
    """Remap cluster labels so person with most images becomes person_1.

    Counts occurrences per label (excluding -1/outliers), sorts descending,
    and returns a mapping from original label to 1-based person number.

    Args:
        labels: List of DBSCAN cluster labels.

    Returns:
        Mapping of {original_label: person_number} where person_number starts at 1.
        Outlier label -1 is not included in the map.
    """
    from collections import Counter  # noqa: PLC0415

    counts = Counter(label for label in labels if label != -1)
    # Sort by count descending, then by label ascending for deterministic order
    sorted_labels = sorted(counts.keys(), key=lambda lbl: (-counts[lbl], lbl))
    return {original: idx + 1 for idx, original in enumerate(sorted_labels)}
