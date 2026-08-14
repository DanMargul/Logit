import numpy as np

class FirstPassageSimulator:
    def __init__(self, n_paths: int = 50000, dt: float = 1.0 / 24.0, liquidation_threshold: float = 1.0):
        self.n_paths = n_paths
        self.dt = dt
        self.liquidation_threshold = liquidation_threshold

    def simulate_hawkes_intensities(self, baseline_intensities: np.ndarray, branching_matrix: np.ndarray, decay_rates: np.ndarray, horizon_steps: int) -> np.ndarray:
        n_markets = len(baseline_intensities)
        intensity_paths = np.zeros((self.n_paths, horizon_steps, n_markets))
        current_intensities = np.tile(baseline_intensities, (self.n_paths, 1))
        
        for t in range(1, horizon_steps):
            decay = np.exp(-decay_rates * self.dt)
            current_intensities = baseline_intensities + (current_intensities - baseline_intensities) * decay
            intensity_paths[:, t, :] = current_intensities
            
        return intensity_paths

    def simulate_health_factor_paths(
        self, 
        initial_health_factor: float, 
        drifts: np.ndarray, 
        volatilities: np.ndarray, 
        correlation_matrix: np.ndarray, 
        jump_intensities: np.ndarray,
        horizon_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n_dim = len(drifts)
        cholesky_factor = np.linalg.cholesky(correlation_matrix)
        
        health_factors = np.zeros((self.n_paths, horizon_steps))
        health_factors[:, 0] = initial_health_factor
        
        default_flags = np.zeros(self.n_paths, dtype=bool)
        
        for t in range(1, horizon_steps):
            standard_normals = np.random.normal(size=(self.n_paths, n_dim))
            correlated_brownian = standard_normals @ cholesky_factor.T
            
            diffusion_shocks = drifts * self.dt + volatilities * np.sqrt(self.dt) * correlated_brownian
            portfolio_delta = np.sum(diffusion_shocks, axis=1)
            
            current_hf = health_factors[:, t - 1] + portfolio_delta
            health_factors[:, t] = current_hf
            
            breached = (current_hf <= self.liquidation_threshold) & (~default_flags)
            default_flags |= breached
            
        return health_factors, default_flags

    def compute_default_probability(self, default_flags: np.ndarray, likelihood_weights: np.ndarray = None) -> float:
        if likelihood_weights is None:
            return np.mean(default_flags)
        return np.mean(default_flags * likelihood_weights)

    def evaluate_risk_surface(self, time_grid: np.ndarray, intensity_grid: np.ndarray, base_drifts: np.ndarray, base_vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
        n_times = len(time_grid)
        n_intensities = len(intensity_grid)
        surface = np.zeros((n_times, n_intensities))
        
        for i, s in enumerate(time_grid):
            for j, lam in enumerate(intensity_grid):
                scaled_vols = base_vols * np.sqrt(lam)
                _, flags = self.simulate_health_factor_paths(
                    initial_health_factor=1.25,
                    drifts=base_drifts,
                    volatilities=scaled_vols,
                    correlation_matrix=corr,
                    jump_intensities=np.array([lam]),
                    horizon_steps=50
                )
                surface[i, j] = np.mean(flags)
                
        return surface
