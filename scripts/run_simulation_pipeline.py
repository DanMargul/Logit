import numpy as np
from logit.models.simulation import FirstPassageSimulator

def main():
    simulator = FirstPassageSimulator(n_paths=1000, dt=1.0 / 24.0, liquidation_threshold=1.0)
    
    drifts = np.array([0.005, -0.002])
    volatilities = np.array([0.15, 0.20])
    correlation = np.array([[1.0, 0.4], [0.4, 1.0]])
    theta = np.array([-0.1, -0.05])
    
    health_factors, default_flags, brownian_paths = simulator.simulate_paths_with_importance_sampling(
        initial_health_factor=1.25,
        drift=drifts,
        volatility=volatilities,
        correlation=correlation,
        theta=theta,
        t_max=10.0
    )
    
    weights = simulator.compute_girsanov_weights(theta, volatilities, brownian_paths, 10.0)
    is_pd = simulator.compute_importance_sampling_default_probability(default_flags, weights)
    standard_pd = simulator.compute_default_probability(default_flags)
    
    time_grid = np.array([0.0, 5.0, 10.0])
    intensity_grid = np.array([1.0, 2.0, 3.0])
    risk_surface = simulator.evaluate_risk_surface(
        time_grid=time_grid,
        intensity_grid=intensity_grid,
        base_drifts=drifts,
        base_vols=volatilities,
        corr=correlation
    )
    
    print("SIMULATION PIPELINE COMPLETE")
    print(f"Standard Default Probability: {standard_pd:.4f}")
    print(f"Importance Sampling Default Probability: {is_pd:.4f}")
    print(f"Risk Surface Shape: {risk_surface.shape}")
    print(f"Risk Surface Grid Values:\n{risk_surface}")

if __name__ == "__main__":
    main()
