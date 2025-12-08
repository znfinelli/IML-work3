"""
Principal Component Analysis (PCA) Implementation.

This module provides a custom implementation of PCA as required by Task 1.2.1
of the assignment. It manually computes the covariance matrix and eigendecomposition
to project data into a lower-dimensional subspace.

It explicitly includes verbose logging to satisfy the requirement: "Show this
information in console" for covariance, eigenvalues, and eigenvectors.

References
----------
[1] Work 3 Description, UB, 2025, "1.2.1 Tasks", Steps 1-9, pp. 5.
[2] Support Slides Session 4, Salamó, 2025, "Principal Component Analysis", pp. 10-11.
"""

import numpy as np


class PCA:
    """
    Custom Principal Component Analysis algorithm.

    Parameters
    ----------
    n_components : int
        The target dimension (number of principal components to keep).
    verbose : bool, default=False
        If True, prints covariance matrix, eigenvalues, and eigenvectors to console
        as required by Assignment Steps 4, 5, and 6 [1].
    """

    def __init__(self, n_components: int, verbose: bool = False):
        self.n_components = n_components
        self.verbose = verbose
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray):
        """
        Computes the principal components from the dataset X.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        # Step 3: Compute d-dimensional mean vector [1]
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 4: Compute Covariance Matrix [1]
        # Note: rowvar=False because rows are samples, cols are features
        covariance_matrix = np.cov(X_centered.T)

        if self.verbose:
            print("\n[PCA Task Step 4] Covariance Matrix:")
            print(covariance_matrix)

        # Step 5: Compute Eigenvalues and Eigenvectors [1]
        # np.linalg.eigh is optimized for symmetric matrices (like covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        if self.verbose:
            print("\n[PCA Task Step 5] Raw Eigenvalues:", eigenvalues)
            print("[PCA Task Step 5] Raw Eigenvectors:\n", eigenvectors)

        # Step 6: Sort Eigenvectors by decreasing eigenvalues [1]
        # eigh returns eigenvalues in ascending order, so we reverse them [::-1]
        sorted_indices = np.argsort(eigenvalues)[::-1]

        self.eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        if self.verbose:
            print("\n[PCA Task Step 6] Sorted Eigenvalues:", self.eigenvalues)
            print("[PCA Task Step 6] Sorted Eigenvectors (Top k):\n",
                  sorted_eigenvectors[:, :self.n_components])

        # Step 6 (cont): Choose k eigenvectors to form projection matrix
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Calculate explained variance for analysis
        total_variance = np.sum(self.eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio = self.eigenvalues[:self.n_components] / total_variance
        else:
            self.explained_variance_ratio = np.zeros(self.n_components)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Step 7: Derive the new dataset.
        Use the d x k eigenvector matrix to transform samples onto the new subspace.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        if self.components is None:
            raise Exception("PCA has not been fitted yet. Call fit() first.")

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Step 9: Reconstruct the data set back to the original one.

        Parameters
        ----------
        X_transformed : np.ndarray
            Data in the PC subspace.

        Returns
        -------
        np.ndarray
            Reconstructed data in original space.
        """
        return np.dot(X_transformed, self.components.T) + self.mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.
        """
        self.fit(X)
        return self.transform(X)