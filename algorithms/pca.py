import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Initializes the PCA algorithm.
        
        Args:
            n_components (int): The number of principal components to keep.
                                This is the target dimension (e.g., reduce to 2D).
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Computes the principal components from the dataset X.
        
        Steps:
        1. Center the data (subtract mean).
        2. Compute Covariance Matrix.
        3. Compute Eigenvalues and Eigenvectors.
        4. Sort them and keep the top 'n_components'.
        """
        # Step 1: Mean Centering
        # We calculate the mean for each feature (column)
        self.mean = np.mean(X, axis=0)
        # We subtract the mean from the data so it is centered at 0
        X_centered = X - self.mean

        # Step 2: Covariance Matrix
        # np.cov expects rows to be variables, so we transpose X_centered (.T)
        covariance_matrix = np.cov(X_centered.T)

        # Step 3: Eigenvalues and Eigenvectors
        # These represent the magnitude (values) and direction (vectors) of the spread of data
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort Eigenvectors
        # eigh returns them in ascending order, so we reverse them ([::-1])
        # We want the highest eigenvalues first (most information)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Store the top 'n_components'
        # We take the first 'n_components' vectors (columns)
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Extra: Calculate explained variance ratio (useful for analysis)
        # This tells us "how much % of the original information is kept"
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance_ratio = sorted_eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        """
        Projects the original data X onto the principal components found in fit().
        """
        if self.components is None:
            raise Exception("PCA has not been fitted yet. Call fit() first.")

        # 1. Center the data again using the mean calculated in fit
        X_centered = X - self.mean
        
        # 2. Project data
        # Dot product between data and our principal components
        # (N_samples, N_features) dot (N_features, n_components) -> (N_samples, n_components)
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Helper method to fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)

# Main block to test the algorithm independently
if __name__ == "__main__":
    # 1. Create a random dataset (Simulating high-dimensional data)
    # Generate a toy dataset with 10 samples and 5 features to verify the logic.
    np.random.seed(42) # Set seed for reproducibility
    X_test = np.random.rand(10, 5)
    
    print("Original Data Shape:", X_test.shape) # Expected: (10, 5)
    print("First 2 rows of original data:\n", X_test[:2])
    print("-" * 30)

    # 2. Initialize PCA
    # We want to reduce the data from 5 dimensions down to 2.
    n_components = 2
    pca = PCA(n_components=n_components)

    # 3. Fit and Transform
    # Compute eigenvectors and project the data into the new lower-dimensional space.
    X_projected = pca.fit_transform(X_test)

    print(f"Projected Data Shape (Target: {n_components}):", X_projected.shape) # Expected: (10, 2)
    print("First 2 rows of projected data (New coordinates):\n", X_projected[:2])
    print("-" * 30)

    # 4. Check explained variance
    # This metric indicates how much information (variance) is preserved after reduction.
    print("Explained Variance Ratio:", pca.explained_variance_ratio)
    print(f"Total Variance Preserved: {np.sum(pca.explained_variance_ratio) * 100:.2f}%")