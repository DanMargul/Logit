import numpy as np
from logit.models.hawkes_calibration import MultivariateHawkesCalibrator

def test_multivariate_hawkes_calibration():
    calibrator = MultivariateHawkesCalibrator(eta_g=0.01)
    event_times = [
        np.array([1.2, 3.4, 5.6]),
        np.array([1.5, 3.8, 6.0])
    ]
    t_max = 10.0
    mu, alpha = calibrator.fit(event_times, t_max)
    
    assert len(mu) == 2
    assert alpha.shape == (2, 2)
    assert np.all(mu >= 0.0)
    assert np.all(alpha >= 0.0)
