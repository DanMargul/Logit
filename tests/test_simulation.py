import numpy as np
from logit.models.simulation import FirstPassageSimulator

def test_first_passage_simulation():
    simulator = FirstPassageSimulator(n_paths=100, dt=0.1, liquidation_threshold=1.0)
    drifts = np.array([[0.01]])
    vols = np.array([[0.2]])
    corr = np.array([[1.0]])
    hf_paths, default_flags = simulator.simulate_health_factor_paths(
        initial_health_factor=1.2,
        drifts=drifts,
        volatilities=vols,
        correlation_matrix=corr,
        jump_intensities=np.array([0.1]),
        horizon_steps=20
    )
    pd = simulator.compute_default_probability(default_flags)
    assert hf_paths.shape == (100, 20)
    assert default_flags.shape == (100,)
    assert 0.0 <= pd <= 1.0
