import numpy as np


class SinkhornSolver:
    def __init__(self, epsilon: float = 0.1, max_iter: int = 200, tolerance: float = 1e-6):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tolerance = tolerance

    def compute_cost_matrix(
        self, empirical_scenarios: np.ndarray, candidate_scenarios: np.ndarray
    ) -> np.ndarray:
        diff = empirical_scenarios[:, np.newaxis, :] - candidate_scenarios[np.newaxis, :, :]
        return np.sum(diff**2, axis=2)

    def solve(self, p_emp: np.ndarray, w_target: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        k_matrix = np.exp(-cost_matrix / self.epsilon)

        a_vec = np.ones_like(p_emp)
        b_vec = np.ones_like(w_target)

        for _ in range(self.max_iter):
            a_prev = a_vec.copy()

            b_vec = w_target / (k_matrix.T @ a_vec)
            a_vec = p_emp / (k_matrix @ b_vec)

            if np.max(np.abs(a_vec - a_prev)) < self.tolerance:
                break

        return np.diag(a_vec) @ k_matrix @ np.diag(b_vec)


class WassersteinRobustness:
    def __init__(self, confidence_level: float = 0.01):
        self.confidence_level = confidence_level

    def compute_structure_aware_lipschitz(
        self, position_sizes: np.ndarray, recovery_discounts: np.ndarray
    ) -> float:
        effective_exposures = position_sizes * (1.0 - recovery_discounts)
        return 0.25 * np.sqrt(np.sum(effective_exposures**2))

    def calculate_capital_addon(
        self, lipschitz_constant: float, wasserstein_radius: float
    ) -> float:
        return (lipschitz_constant * wasserstein_radius) / self.confidence_level
