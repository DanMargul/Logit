import numpy as np
from scipy.optimize import linprog
from scipy.linalg import eigh


class DependencyGraphClustering:
    def __init__(self, lambda_excite: float = 0.5, lambda_corr: float = 0.3, lambda_semantic: float = 0.2):
        self.lambda_excite = lambda_excite
        self.lambda_corr = lambda_corr
        self.lambda_semantic = lambda_semantic

    def build_affinity_matrix(
            self,
            branching_matrix: np.ndarray,
            diffusion_correlation: np.ndarray,
            semantic_similarity: np.ndarray
    ) -> np.ndarray:
        symmetrized_excite = (branching_matrix + branching_matrix.T) / 2.0
        clamped_corr = np.maximum(0.0, diffusion_correlation)

        return (
                self.lambda_excite * symmetrized_excite
                + self.lambda_corr * clamped_corr
                + self.lambda_semantic * semantic_similarity
        )

    def compute_normalized_laplacian(self, affinity_matrix: np.ndarray) -> np.ndarray:
        degrees = np.sum(affinity_matrix, axis=1)
        inverse_sqrt_degrees = np.power(np.maximum(degrees, 1e-10), -0.5)
        degree_matrix_inv_sqrt = np.diag(inverse_sqrt_degrees)

        identity = np.eye(affinity_matrix.shape[0])
        return identity - (degree_matrix_inv_sqrt @ affinity_matrix @ degree_matrix_inv_sqrt)

    def extract_spectral_embedding(self, laplacian: np.ndarray, n_clusters: int) -> np.ndarray:
        _, eigenvectors = eigh(laplacian)
        embedding = eigenvectors[:, :n_clusters]
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(row_norms, 1e-10)


class CapitalAllocationOptimizer:
    def solve_allocation(
            self,
            expected_returns: np.ndarray,
            borrower_caps: np.ndarray,
            pool_capacity: float,
            market_exposures: np.ndarray,
            market_caps: np.ndarray,
            cluster_limits: np.ndarray,
            cluster_membership: np.ndarray,
            treasury_equity: float,
            expected_losses: np.ndarray,
            hawkes_multiplier: float
    ) -> np.ndarray:
        n_borrowers = len(expected_returns)

        objective_coefficients = -expected_returns

        inequality_constraints = []
        inequality_bounds = []

        inequality_constraints.append(np.ones(n_borrowers))
        inequality_bounds.append(pool_capacity)

        for i, cap in enumerate(market_caps):
            row = market_exposures[:, i]
            inequality_constraints.append(row)
            inequality_bounds.append(cap)

        unique_clusters = np.unique(cluster_membership)
        for cluster_id in unique_clusters:
            mask = (cluster_membership == cluster_id).astype(float)
            inequality_constraints.append(mask)
            inequality_bounds.append(cluster_limits[cluster_id])

        buffer_row = hawkes_multiplier * expected_losses
        inequality_constraints.append(buffer_row)
        inequality_bounds.append(treasury_equity)

        a_ub = np.array(inequality_constraints)
        b_ub = np.array(inequality_bounds)

        bounds = [(0.0, cap) for cap in borrower_caps]

        result = linprog(
            c=objective_coefficients,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs"
        )

        if not result.success:
            return np.zeros(n_borrowers)

        return result.x