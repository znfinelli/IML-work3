"""
Far Efficient K-Means (FEKM) Implementation.

This module provides an improved initialization strategy for K-Means as required
by Task 1.5 of the assignment. It implements the "Far Efficient K-Means"
algorithm proposed by Mishra et al. (2019), which deterministically selects
initial centroids to avoid local minima.

References
----------
[1] Mishra, B.K., Rath, A.K., Nanda, S.K., Baidyanath, R.R., "Efficient
    Intelligent Framework for Selection of Initial Cluster Centers", 2019,
    I.J. Intelligent Systems and Applications, 8, 44-55.
    (Specifically Section IV.B, Pseudo-code Steps 1-9).
"""

import numpy as np
from scipy.spatial.distance import pdist

# Handle both relative and absolute imports for flexibility
try:
    from .kmeans import KMeans
except ImportError:
    from kmeans import KMeans


class KMeansFEKM(KMeans):
    """
    K-Means with Far Efficient K-Means (FEKM) initialization.

    This subclass overrides the `_initialize_centroids` method of the standard
    KMeans class. It implements the three-stage initialization process described
    in [1]:
    1. Identify the two farthest points in the dataset.
    2. Refine these two centers by assigning nearby points up to a threshold.
    3. Select the remaining K-2 centers using a MaxMin distance strategy.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form.
    max_iters : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    metric : str, default='euclidean'
        Distance metric.
    random_state : int, optional
        Used only if fallback randomization is needed (rare).
    """

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the FEKM strategy.

        References
        ----------
        [1] Mishra et al. (2019), Pseudo-code FEKM(data_set, k), p. 48.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Calculated initial centroids of shape (n_clusters, n_features).
        """
        X = X.astype(np.float32)
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # ---------------------------------------------------------
        # Step 1: Find the farthest pair of points
        # Ref [1]: Steps 1-2 "Determine two data points with max distance apart"
        # ---------------------------------------------------------
        # We use pdist to calculate the condensed distance matrix (1D array).
        # This is memory intensive O(N^2) but required by the algorithm.
        distances = pdist(X, metric='euclidean')

        # Find the index of the maximum distance in the condensed array
        max_condensed_idx = np.argmax(distances)

        # Convert the 1D condensed index back to 2D (i, j) indices
        # Formula derives row (i) and col (j) from the triangular index
        b = 1 - 2 * n_samples
        i = int(np.floor((-b - np.sqrt(b ** 2 - 8 * max_condensed_idx)) / 2))
        j = int(max_condensed_idx + i + 1 - n_samples * (n_samples - 1) / 2 + \
                (n_samples - i) * ((n_samples - i) - 1) / 2)

        # Set the first two centroids
        centroids[0] = X[i]
        centroids[1] = X[j]

        # ---------------------------------------------------------
        # Step 2: Threshold Assignment & Update
        # Ref [1]: Steps 3-6 "Grouping data nearest to center[1] and [2] till threshold"
        # ---------------------------------------------------------

        # Calculate distance from ALL points to the two chosen centroids
        d1 = np.linalg.norm(X - centroids[0], axis=1)
        d2 = np.linalg.norm(X - centroids[1], axis=1)

        # Determine threshold (Ref [1] Step 4: 0.5 * (N/k))
        threshold_count = int(0.5 * (n_samples / self.n_clusters))

        # Track assigned points (to exclude them from Step 3)
        assigned_mask = np.zeros(n_samples, dtype=bool)

        # Identify candidates for C1 (closer to C1 than C2)
        # Sort by distance to prioritize the "closest" ones first
        c1_candidates_idx = np.where(d1 <= d2)[0]
        c1_sorted_idx = c1_candidates_idx[np.argsort(d1[c1_candidates_idx])]

        # Assign up to threshold
        count_c1 = min(len(c1_sorted_idx), threshold_count)
        final_c1_idx = c1_sorted_idx[:count_c1]
        assigned_mask[final_c1_idx] = True

        # Identify candidates for C2 (closer to C2 than C1)
        c2_candidates_idx = np.where(d1 > d2)[0]
        c2_sorted_idx = c2_candidates_idx[np.argsort(d2[c2_candidates_idx])]

        # Assign up to threshold
        count_c2 = min(len(c2_sorted_idx), threshold_count)
        final_c2_idx = c2_sorted_idx[:count_c2]
        assigned_mask[final_c2_idx] = True

        # Update Centers 1 & 2 to be the Mean of their assigned points (Ref [1] Steps 5-6)
        if count_c1 > 0:
            centroids[0] = np.mean(X[final_c1_idx], axis=0)
        if count_c2 > 0:
            centroids[1] = np.mean(X[final_c2_idx], axis=0)

        # ---------------------------------------------------------
        # Step 3: Select Remaining (K-2) Centers
        # Ref [1]: Step 8 "max(min(distance))" strategy on remaining points
        # ---------------------------------------------------------

        # We only look at points that were NOT assigned in Step 2
        remaining_indices = np.where(~assigned_mask)[0]

        # Safety check: if threshold consumed everything (unlikely), use all points
        if len(remaining_indices) == 0:
            remaining_indices = np.arange(n_samples)

        X_remaining = X[remaining_indices]

        for k in range(2, self.n_clusters):
            # Calculate distance from remaining points to ALL currently chosen centroids
            # Shape: (n_remaining, k_current)
            current_centroids = centroids[:k]

            # Broadcasting: (N, 1, F) - (1, K, F) -> (N, K, F) -> norm -> (N, K)
            dists_to_current = np.linalg.norm(
                X_remaining[:, None, :] - current_centroids[None, :, :],
                axis=2
            )

            # For each point, find distance to its NEAREST centroid (min_list)
            min_dists_to_centers = np.min(dists_to_current, axis=1)

            # Pick the point that is FARTHEST from its nearest centroid (MaxMin)
            next_center_local_idx = np.argmax(min_dists_to_centers)
            centroids[k] = X_remaining[next_center_local_idx]

        return centroids