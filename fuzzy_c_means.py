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
            m: Fuzziness exponent (default 2.0).
            alpha: Suppression parameter (0 < alpha <= 1). 
                   - If alpha = 1.0: Runs Standard FCM (Bezdek).
                   - If alpha < 1.0: Runs Suppressed FCM (Fan et al.).
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
        self.n_iter_ = 0
        self.labels_ = None

    def fit(self, X):
        """
        Execute the clustering algorithm.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # 1. Initialize U randomly
        # Constraint: sum of memberships for each point must equal 1
        self.u = np.random.rand(n_samples, self.n_clusters)
        self.u = self.u / self.u.sum(axis=1, keepdims=True)
        
        for iteration in range(self.max_iters):
            self.n_iter_ = iteration
            u_prev = self.u.copy()
            
            # 2. Calculate Centroids (V)
            # This formula is identical for both Bezdek and Fan et al.
            # V_j = sum(u_ij^m * x_i) / sum(u_ij^m)
            u_pow_m = self.u ** self.m
            denominator = u_pow_m.sum(axis=0).reshape(-1, 1)
            numerator = np.dot(u_pow_m.T, X)
            
            # Safe division (handle empty clusters)
            self.centroids = np.divide(
                numerator, 
                denominator, 
                out=np.zeros_like(numerator), 
                where=denominator!=0
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
        """
        n_samples = X.shape[0]
        u_new = np.zeros((n_samples, self.n_clusters))
        exponent = 2.0 / (self.m - 1)
        
        # Calculate distances from points to centroids
        dist_matrix = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            dist_matrix[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
            
        for i in range(n_samples):
            dists_i = dist_matrix[i]
            
            # Handle singularity (point exactly on centroid)
            if np.any(dists_i == 0):
                u_new[i, :] = 0
                u_new[i, np.argmin(dists_i)] = 1.0
            else:
                # Eq 8 from Bezdek (1984)
                sum_terms = 0.0
                for j in range(self.n_clusters):
                    ratio = dists_i / dists_i[j]
                    sum_terms += ratio ** exponent
                u_new[i] = 1.0 / sum_terms
        return u_new

    def _suppress_membership(self, u_old):
        """
        Applies suppression logic from Fan et al. (2003).
        If alpha=1.0, acts as identity function (Standard FCM).
        """
        if self.alpha >= 1.0:
            return u_old
            
        n_samples = u_old.shape[0]
        u_suppressed = u_old.copy()
        
        for i in range(n_samples):
            # Identify the winning cluster
            winner_idx = np.argmax(u_old[i])
            
            # Suppress non-winners: u_new = alpha * u_old
            u_suppressed[i] = self.alpha * u_old[i]
            
            # Boost the winner: u_winner = 1 - sum(suppressed_others)
            # Note: calculate sum of OTHERS, not including the winner yet
            current_sum_others = np.sum(u_suppressed[i]) - u_suppressed[i, winner_idx]
            u_suppressed[i, winner_idx] = 1.0 - current_sum_others
            
        return u_suppressed

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# ---------------------------------------------------------------------
# Verification Block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    
    print("--- Test 1: Standard FCM (Bezdek) alpha=1.0 ---")
    fcm_std = FuzzyCMeans(n_clusters=3, m=2.0, alpha=1.0, random_state=42)
    fcm_std.fit(X)
    print(f"Converged in {fcm_std.n_iter_} iterations.")
    
    print("\n--- Test 2: Suppressed FCM (Fan et al.) alpha=0.7 ---")
    fcm_sup = FuzzyCMeans(n_clusters=3, m=2.0, alpha=0.7, random_state=42)
    fcm_sup.fit(X)
    print(f"Converged in {fcm_sup.n_iter_} iterations.")
    print("(Note: Suppressed FCM usually converges faster or creates crisper partitions)")