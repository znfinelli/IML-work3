import numpy as np


class PCA:
    def __init__(self, n_components: int, verbose: bool = False):
        """
        Initializes the PCA algorithm.

        Args:
            n_components (int): The target dimension.
            verbose (bool): If True, prints covariance, eigenvalues, and vectors
                            to console (Required by Assignment Task 1.2.1).
        """
        self.n_components = n_components
        self.verbose = verbose
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Computes the principal components from the dataset X.
        """
        # Step 3: Compute d-dimensional mean vector
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 4: Compute Covariance Matrix
        covariance_matrix = np.cov(X_centered.T)

        if self.verbose:
            print("\n[PCA] Covariance Matrix:")
            print(covariance_matrix)

        # Step 5: Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        if self.verbose:
            print("\n[PCA] Raw Eigenvalues:", eigenvalues)
            print("[PCA] Raw Eigenvectors:\n", eigenvectors)

        # Step 6: Sort Eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]

        self.eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        if self.verbose:
            print("\n[PCA] Sorted Eigenvalues:", self.eigenvalues)
            print("[PCA] Sorted Eigenvectors (Top k):\n", sorted_eigenvectors[:, :self.n_components])

        # Form the projection matrix (d x k)
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Explained Variance
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        """
        Step 7: Derive the new dataset (Project X onto new subspace).
        """
        if self.components is None:
            raise Exception("PCA has not been fitted yet. Call fit() first.")

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_transformed):
        """
        Step 9: Reconstruct the data set back to the original one.
        X_original approx = (X_transformed @ components.T) + mean
        """
        return np.dot(X_transformed, self.components.T) + self.mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)