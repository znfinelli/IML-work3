import numpy as np

# Handle both relative and absolute imports for flexibility
try:
    from .kmeans import KMeans
except ImportError:
    from kmeans import KMeans


class KernelKMeans(KMeans):
    """
    Intelligent Kernel K-Means Clustering.

    This algorithm implements K-Means in a high-dimensional feature space using the Kernel Trick.
    It includes the "Intelligent" initialization strategy proposed by Handhayani & Hiryanto,
    which selects initial centroids based on their distance from the global center of mass.

    References:
    -----------
    [1] Handhayani, T., & Hiryanto, L. (2015). "Intelligent Kernel K-Means for Clustering
        Gene Expression". Procedia Computer Science, 59, 171-177.
    [2] Dhillon, I. S., et al. (2004). "Kernel k-means: spectral clustering and normalized cuts".
    """

    def __init__(self, n_clusters: int, max_iters: int = 300, tol: float = 1e-4,
                 kernel: str = 'rbf', gamma: float = None, random_state: int = None):
        """
        Initialize Kernel K-Means.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters.
        kernel : str
            Kernel type. Options: 'linear', 'rbf' (Gaussian).
            - 'linear': K(x,y) = <x, y>. Equivalent to standard K-Means.
            - 'rbf': K(x,y) = exp(-gamma * ||x - y||^2). Captures non-linear shapes.
        gamma : float
            Kernel coefficient for 'rbf'. If None, defaults to 1 / n_features.
        """
        # We ignore 'metric' since Kernel K-Means uses its own distance logic derived from the Kernel Matrix
        super().__init__(n_clusters, max_iters, tol, metric='kernel', random_state=random_state)
        self.kernel_name = kernel
        self.gamma = gamma
        self.K_matrix = None  # The Kernel Matrix (N x N)
        self.center_indices = None  # Indices of points acting as centers

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the Kernel Matrix K where K[i, j] = kernel(x_i, x_j).
        This matrix represents the similarity between all pairs of points in the feature space.
        """
        X = X.astype(np.float32)

        if self.kernel_name == 'linear':
            return np.dot(X, X.T)

        elif self.kernel_name == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]

            # Computations in float32
            sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)

            # The result of dot will be float32 (approx 8.9 GB)
            dot_product = np.dot(X, X.T)

            # This calculation stays within safe RAM limits
            dist_sq = sq_norms + sq_norms.T - 2 * dot_product

            # Clean up immediately
            del dot_product

            dist_sq = np.maximum(dist_sq, 0)
            return np.exp(-self.gamma * dist_sq)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def _get_distance_to_mean(self, indices=None) -> np.ndarray:
        """
        Helper to calculate the squared distance of every point to the mean (centroid)
        of a specific set of points (specified by 'indices').

        If indices is None, calculates distance to the global Center of Mass (CoM).

        Based on Eq (3) in Handhayani paper:
        ||phi(x) - Mean||^2 = K(x,x) - 2/L * sum(K(x, xi)) + 1/L^2 * sum(K(xi, xj))
        """
        N = self.K_matrix.shape[0]
        diag_K = np.diag(self.K_matrix)  # K(x, x)

        if indices is None:
            # Global Mean (all points)
            # Term 2: - 2/N * sum(K(x, all_xi))
            term2 = -2.0 * np.mean(self.K_matrix, axis=1)
            # Term 3: Constant (can be ignored for argmax comparisons)
            return diag_K + term2
        else:
            # Mean of a specific cluster/subset
            L = len(indices)
            if L == 0: return np.full(N, np.inf)

            # Extract relevant columns for the sum
            subset_K = self.K_matrix[:, indices]
            term2 = -2.0 / L * np.sum(subset_K, axis=1)

            # The third term is constant for a fixed set of indices, so we skip computing it
            # if we are just comparing distances.
            return diag_K + term2

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initializes cluster assignments using the "Intelligent" strategy described
        in Handhayani & Hiryanto, Section 2 (Figures 2 & 3).

        Strategy:
        1. Compute Center of Mass (CoM) of the entire dataset.
        2. C1 = Object farthest from CoM (Eq 3 in paper).
        3. C2 = Object farthest from C1 (Eq 4 in paper).
        4. For K > 2, select next center as the object with maximum distance to its nearest center.
        """
        n_samples = X.shape[0]

        # 1. Find Global Center of Mass (CoM) in Feature Space and distances to it
        # We use the helper function derived from Eq (3)
        dist_to_global_mean = self._get_distance_to_mean(indices=None)

        # 2. C1 is the object farthest from CoM
        c1_idx = np.argmax(dist_to_global_mean)
        center_indices = [c1_idx]

        # 3. Select remaining centers
        # The paper describes finding C2 farthest from C1.
        # We generalize this to K centers using a MaxMin strategy in Feature Space.
        # Distance between two points x and z in feature space (Eq 4 in paper):
        # ||phi(x) - phi(z)||^2 = K(x,x) + K(z,z) - 2K(x,z)

        diag_K = np.diag(self.K_matrix)

        for _ in range(1, self.n_clusters):
            # Track min distance of every point to the SET of already chosen centers
            min_dists = np.full(n_samples, np.inf)

            for c_idx in center_indices:
                # Calculate distance from every point 'i' to center 'c_idx'
                # D^2(i, c) = K(i,i) + K(c,c) - 2K(i,c)
                dists = diag_K + self.K_matrix[c_idx, c_idx] - 2 * self.K_matrix[:, c_idx]

                # We keep the distance to the NEAREST center found so far
                min_dists = np.minimum(min_dists, dists)

            # Select the point that is FARTHEST from its nearest center (MaxMin)
            next_c = np.argmax(min_dists)
            center_indices.append(next_c)

        return np.array(center_indices)

    def fit(self, X):
        """
        Compute Kernel K-Means clustering.
        """
        # 1. Precompute Kernel Matrix (Computationally intensive step)
        # If input is DataFrame, convert to numpy array
        X_data = X.values if hasattr(X, 'values') else X
        self.K_matrix = self._compute_kernel_matrix(X_data)

        n_samples = X_data.shape[0]

        # 2. Initialize Centers (Indices of points acting as initial prototypes)
        self.center_indices = self._initialize_centroids(X_data)

        # Initial assignment based on these centers
        # We calculate distance from each point to each initial center point
        diag_K = np.diag(self.K_matrix)
        dists = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            c_idx = self.center_indices[k]
            # Eq (4) in paper: distance between two single points
            dists[:, k] = diag_K + self.K_matrix[c_idx, c_idx] - 2 * self.K_matrix[:, c_idx]

        self.labels_ = np.argmin(dists, axis=1)

        # 3. Iterative Update
        for iteration in range(self.max_iters):
            labels_prev = self.labels_.copy()
            dists = np.zeros((n_samples, self.n_clusters))

            # For each cluster, calculate distance from every point x to the cluster centroid
            # The centroid is implicit (average of points in the cluster in feature space)
            for k in range(self.n_clusters):
                # Get indices of points currently in cluster k
                mask = (self.labels_ == k)
                Nk = np.sum(mask)

                if Nk == 0:
                    # Handle empty cluster: keep distance infinite so no one is assigned
                    dists[:, k] = np.inf
                    continue

                # Calculate Squared Distance in Feature Space
                # Formula derived from Eq (3) in paper for a general centroid mu_k:
                # ||phi(x) - mu_k||^2 = K(xx) - 2/Nk * sum_{y in Ck} K(xy) + 1/Nk^2 * sum_{y,z in Ck} K(yz)

                # Term 2: Interaction between point x and cluster k
                # sum(K(x, y)) is sum of columns of K corresponding to cluster members
                term2 = -2.0 / Nk * np.sum(self.K_matrix[:, mask], axis=1)

                # Term 3: Interaction within cluster k (constant for the cluster)
                term3 = 1.0 / (Nk ** 2) * np.sum(self.K_matrix[np.ix_(mask, mask)])

                dists[:, k] = diag_K + term2 + term3

            # Assign new labels based on minimum distance
            self.labels_ = np.argmin(dists, axis=1)

            # Check for convergence (assignments stop changing)
            if np.array_equal(self.labels_, labels_prev):
                break

        # Calculate final inertia (sum of squared distances to closest centroid)
        min_dists = np.min(dists, axis=1)
        self.inertia_ = np.sum(min_dists)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------
# Verification Block (Run this file directly to test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_circles, make_blobs

    print("--- Testing Intelligent Kernel K-Means ---")

    # 1. Test on Non-Linear Data (Concentric Circles)
    # Standard K-Means fails here, but Kernel K-Means (RBF) should succeed.
    print("\nTest 1: Concentric Circles (Non-linear)")
    X_circles, y_circles = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=42)

    # Gamma=10 usually works well for these tight circles
    kkm_circles = KernelKMeans(n_clusters=2, kernel='rbf', gamma=10.0, random_state=42)
    labels_circles = kkm_circles.fit_predict(X_circles)

    print(f"Inertia: {kkm_circles.inertia_:.4f}")
    # Simple accuracy check (should be near 100% or 0% depending on label swap)
    acc = np.mean(labels_circles == y_circles)
    print(f"Accuracy (approx): {max(acc, 1 - acc):.2f}")

    # 2. Test on Linear Data (Blobs)
    # Using 'linear' kernel should behave exactly like Standard K-Means
    print("\nTest 2: Blobs (Linear Kernel)")
    X_blobs, y_blobs = make_blobs(n_samples=200, centers=3, random_state=42)

    kkm_blobs = KernelKMeans(n_clusters=3, kernel='linear', random_state=42)
    labels_blobs = kkm_blobs.fit_predict(X_blobs)
    print(f"Inertia: {kkm_blobs.inertia_:.4f}")
    print("Test Complete.")