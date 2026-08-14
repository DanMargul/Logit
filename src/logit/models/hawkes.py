import numpy as np


class ExponentialHawkesKernel:
    def __init__(self, decay_rate: float):
        self.decay_rate = decay_rate

    def compute_recursive_sum(self, timestamps: np.ndarray) -> np.ndarray:
        n_events = len(timestamps)
        a_sum = np.zeros(n_events)

        for k in range(1, n_events):
            time_delta = timestamps[k] - timestamps[k - 1]
            decay_factor = np.exp(-self.decay_rate * time_delta)
            a_sum[k] = decay_factor * (a_sum[k - 1] + 1)

        return a_sum


class SpectralMonitor:
    def compute_spectral_radius(self, branching_matrix: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvals(branching_matrix)
        return np.max(np.abs(eigenvalues))

    def get_regime_classification(self, spectral_radius: float) -> str:
        if spectral_radius < 0.7:
            return "normal"
        if spectral_radius < 0.9:
            return "elevated"
        if spectral_radius < 1.0:
            return "critical"
        return "non-stationary"
