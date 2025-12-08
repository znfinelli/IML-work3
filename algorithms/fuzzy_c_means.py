"""
Fuzzy C-Means (FCM) and Suppressed FCM (s-FCM) Implementation.

This module provides a unified implementation of Fuzzy C-Means algorithms
as required by Task 1.6 of the assignment. It supports both the standard
FCM algorithm (Bezdek) and the Suppressed FCM variant (Fan et al.) via
the `alpha` parameter.

References
----------
[1] Bezdek, J.C., Ehrlich, R., Full, W., "FCM: The fuzzy c-means clustering
    algorithm", 1984, Computers & Geosciences, 10(2-3), pp. 191-203.
[2] Fan, J.L., Zhen, W.Z., Xie, W.X., "Suppressed fuzzy c-means clustering
    algorithm", 2003, Pattern Recognition Letters, 24, pp. 1607-1612.
"""

import numpy as np
from typing import Optional

class FuzzyCMeans:
    """
    Unified Fuzzy C-Means implementation.

    This class implements both Standard FCM and Suppressed FCM (s-FCM).
    The algorithm iterates between calculating cluster centers and updating
    membership degrees until convergence.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form.
    m : float, default=2.0
        The fuzziness exponent (weighting exponent). Must be > 1.
        Controls the "fuzziness" of the resulting partition.
    alpha : float, default=1.0
        The suppression parameter (0 < alpha <= 1).
        - If alpha = 1.0: Runs Standard FCM (Bezdek, 1984).
        - If alpha < 1.0: Runs Suppressed FCM (Fan et al., 2003).
          Recommended value: alpha=0.5 for unknown data structures.
    max_iters : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-5
        Convergence tolerance based on the change in the membership matrix U.
    random_state : int, optional
        Seed for random initialization of the membership matrix.
    """

    def __init__(
            self,
            n_clusters: int,
            m: float = 2.0,
            alpha: float = 1.0,
            max_iters: int = 300,
            tol: float = 1e-5,
            random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.u = None  # Membership matrix (N x C)
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X: np.ndarray):
        """
        Execute the clustering algorithm.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # 1. Initialize U randomly
        # Constraint: sum of memberships for each point must equal 1 (Eq 2b in [1])
        self.u = np.random.rand(n_samples, self.n_clusters)
        self.u = self.u / self.u.sum(axis=1, keepdims=True)

        for iteration in range(self.max_iters):
            self.n_iter_ = iteration + 1
            u_prev = self.u.copy()

            # 2. Calculate Centroids (V)
            # Ref [1]: Equation 11a, p. 193
            # V_j = sum(u_ij^m * x_i) / sum(u_ij^m)
            u_pow_m = self.u ** self.m
            denominator = u_pow_m.sum(axis=0).reshape(-1, 1)
            numerator = np.dot(u_pow_m.T, X)

            # Safe division (handle empty clusters)
            self.centroids = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0
            )

            # 3. Calculate Standard Membership (Bezdek Step)
            # Ref [1]: Equation 11b, p. 193
            u_bezdek = self._calculate_bezdek_membership(X)

            # 4. Apply Suppression (Fan et al. Step)
            # Ref [2]: Section 4 "S-FCM algorithm", p. 1609
            # If alpha=1.0, this returns u_bezdek unchanged (Standard FCM)
            self.u = self._suppress_membership(u_bezdek)

            # 5. Check Convergence (Eq 14 in [1])
            diff = np.max(np.abs(self.u - u_prev))
            if diff < self.tol:
                break

        # Assign hard labels based on maximum membership
        self.labels_ = np.argmax(self.u, axis=1)
        return self

    def _calculate_bezdek_membership(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates standard U matrix based on Euclidean distances.

        Ref [1]: Eq 11b.
        """
        n_samples = X.shape[0]
        exponent = 2.0 / (self.m - 1)

        # 1. Calculate Distance Matrix (N x C)
        dist_matrix = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            dist_matrix[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)

        # 2. Avoid division by zero
        dist_matrix_safe = np.fmax(dist_matrix, 1e-10)

        # 3. Calculate terms: (1 / d_ik) ^ (2 / m-1)
        inv_dist_pow = 1.0 / (dist_matrix_safe ** exponent)

        # Sum of inverted distances for each point (Denominator)
        sum_inv_dist_pow = np.sum(inv_dist_pow, axis=1, keepdims=True)

        # 4. Compute Membership
        u_new = inv_dist_pow / sum_inv_dist_pow

        # 5. Fix Singularities: If a point is exactly at a centroid, membership is 1
        zero_indices = np.where(dist_matrix == 0)
        if len(zero_indices[0]) > 0:
            u_new[zero_indices[0], :] = 0
            u_new[zero_indices[0], zero_indices[1]] = 1.0

        return u_new

    def _suppress_membership(self, u_old: np.ndarray) -> np.ndarray:
        """
        Applies suppression logic from Fan et al. (2003).

        "The algorithm prizes the biggest membership and suppresses the others."
        Ref [2]: Equation on p. 1609.

        u_winner = 1 - alpha + alpha * u_winner
        u_loser  = alpha * u_loser
        """
        if self.alpha >= 1.0:
            return u_old

        # 1. Suppress everyone by alpha first (u_loser logic)
        u_new = u_old * self.alpha

        # 2. Adjust the winner's membership
        rows = np.arange(u_old.shape[0])
        winners = np.argmax(u_old, axis=1)

        # Apply the winner formula: 1 - alpha + alpha * u_old
        u_new[rows, winners] = 1.0 - self.alpha + u_new[rows, winners]

        return u_new

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model and returns hard cluster labels.
        """
        self.fit(X)
        return self.labels_