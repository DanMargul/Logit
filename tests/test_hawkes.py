import numpy as np
from logit.models.hawkes import ExponentialHawkesKernel, SpectralMonitor

def test_exponential_hawkes_recursive_sum():
    kernel = ExponentialHawkesKernel(decay_rate=0.5)
    timestamps = np.array([0.0, 1.0, 2.5, 4.0])
    sums = kernel.compute_recursive_sum(timestamps)
    assert len(sums) == len(timestamps)
    assert sums[0] == 0.0

def test_spectral_monitor_regimes():
    monitor = SpectralMonitor()
    matrix_normal = np.array([[0.1, 0.2], [0.2, 0.1]])
    rho_normal = monitor.compute_spectral_radius(matrix_normal)
    regime_normal = monitor.get_regime_classification(rho_normal)
    assert regime_normal == "normal"

    matrix_critical = np.array([[0.95, 0.0], [0.0, 0.95]])
    rho_critical = monitor.compute_spectral_radius(matrix_critical)
    regime_critical = monitor.get_regime_classification(rho_critical)
    assert regime_critical == "critical"
