import numpy as np

class FuzzyCMeans:
    """
    Unified Fuzzy C-Means implementation.

    Can behave as either Standard FCM or Suppressed FCM based on the 'alpha' parameter.

    References:
    1. Standard FCM: Bezdek, J. C. (1981). Pattern recognition with fuzzy objective function algorithms.
    2. Suppressed FCM: Fan, J. L., et al. (2003). Suppressed fuzzy c-means clustering algorithm.
    """

    def __init__(
            self,
            n_clusters: int,
            m: float = 2.0,
            alpha: float = 1.0,
            max_iters: int = 300,
            tol: float = 1e-5,
            random_state: int = None
    ):
        """
        Args:
            n_clusters (c): Number of clusters.
            m: Fuzziness exponent (default 2.0). m > 1.
            alpha: Suppression parameter (0 < alpha <= 1).
                   - If alpha = 1.0: Runs Standard FCM (Bezdek).
                   - If alpha < 1.0: Runs Suppressed FCM (Fan et al.).
                     (Paper recommends alpha=0.5 for unknown data structures).
            max_iters: Maximum iterations.
            tol: Tolerance for convergence.
            random_state: Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.u = None  # Membership matrix
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X):
        """
        Execute the clustering algorithm.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # --- FIX APPLIED HERE ---
        # We need the integer number of rows, not the shape tuple
        n_samples = X.shape[0]

        # 1. Initialize U randomly
        # Constraint: sum of memberships for each point must equal 1
        self.u = np.random.rand(n_samples, self.n_clusters)
        self.u = self.u / self.u.sum(axis=1, keepdims=True)

        for iteration in range(self.max_iters):
            self.n_iter_ = iteration + 1
            u_prev = self.u.copy()

            # 2. Calculate Centroids (V)
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
            u_bezdek = self._calculate_bezdek_membership(X)

            # 4. Apply Suppression (Fan et al. Step)
            # If alpha=1.0, this returns u_bezdek unchanged.
            self.u = self._suppress_membership(u_bezdek)

            # 5. Check Convergence
            diff = np.max(np.abs(self.u - u_prev))
            if diff < self.tol:
                break

        self.labels_ = np.argmax(self.u, axis=1)
        return self

    def _calculate_bezdek_membership(self, X):
        """
        Calculates standard U matrix based on Euclidean distances (Bezdek 1981).
        Vectorized for performance.
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

        # 5. Fix Singularities
        zero_indices = np.where(dist_matrix == 0)
        if len(zero_indices[0]) > 0:
            u_new[zero_indices[0], :] = 0
            u_new[zero_indices[0], zero_indices[1]] = 1.0

        return u_new

    def _suppress_membership(self, u_old):
        """
        Applies suppression logic from Fan et al. (2003).
        Vectorized for performance.
        """
        if self.alpha >= 1.0:
            return u_old

        # 1. Suppress everyone by alpha first
        u_new = u_old * self.alpha

        # 2. Adjust the winner's membership
        rows = np.arange(u_old.shape[0])
        winners = np.argmax(u_old, axis=1)

        # Apply the winner formula: 1 - alpha + alpha * u_old
        u_new[rows, winners] = 1.0 - self.alpha + u_new[rows, winners]

        return u_new

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------
# Verification Block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate sample data
    X_test, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    print("--- Test 1: Standard FCM (Bezdek) alpha=1.0 ---")
    fcm_std = FuzzyCMeans(n_clusters=3, m=2.0, alpha=1.0, random_state=42)
    fcm_std.fit(X_test)
    print(f"Converged in {fcm_std.n_iter_} iterations.")

    print("\n--- Test 2: Suppressed FCM (Fan et al.) alpha=0.75 ---")
    fcm_sup = FuzzyCMeans(n_clusters=3, m=2.0, alpha=0.75, random_state=42)
    fcm_sup.fit(X_test)
    print(f"Converged in {fcm_sup.n_iter_} iterations.")

    # Recommended alpha=0.5 from Fan et al. paper
    print("\n--- Test 3: Heavy Suppression (Fan et al.) alpha=0.5 ---")
    fcm_heavy = FuzzyCMeans(n_clusters=3, m=2.0, alpha=0.5, random_state=42)
    fcm_heavy.fit(X_test)
    print(f"Converged in {fcm_heavy.n_iter_} iterations.")

    print("\n(Note: Lower alpha usually leads to fewer iterations and crisper clusters)")