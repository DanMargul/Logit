import numpy as np


class FirstPassageSimulator:
    def __init__(self, n_paths: int = 50000, dt: float = 1.0 / 24.0, liquidation_threshold: float = 1.0):
        self.n_paths = n_paths
        self.dt = dt
        self.liquidation_threshold = liquidation_threshold

    def simulate_ogata_thinning(self, baseline_intensity: float, alpha: float, beta: float, t_max: float) -> list[
        float]:
        events = []
        t = 0.0
        while t < t_max:
            current_lambda = baseline_intensity
            if len(events) > 0:
                current_lambda += np.sum(alpha * np.exp(-beta * (t - np.array(events))))
            lambda_max = current_lambda
            u1 = np.random.uniform(0, 1)
            w = -np.log(u1) / lambda_max
            t += w
            if t >= t_max:
                break
            eval_lambda = baseline_intensity + np.sum(alpha * np.exp(-beta * (t - np.array(events))))
            u2 = np.random.uniform(0, 1)
            if u2 <= eval_lambda / lambda_max:
                events.append(t)
        return events

    def simulate_hawkes_intensities(self, baseline_intensities: np.ndarray, branching_matrix: np.ndarray,
                                    decay_rates: np.ndarray, horizon_steps: int) -> np.ndarray:
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

    def simulate_paths_with_importance_sampling(
            self,
            initial_health_factor: float,
            drift: np.ndarray,
            volatility: np.ndarray,
            correlation: np.ndarray,
            theta: np.ndarray,
            t_max: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_dim = len(drift)
        cholesky = np.linalg.cholesky(correlation)
        steps = int(t_max / self.dt)

        health_factors = np.zeros((self.n_paths, steps))
        health_factors[:, 0] = initial_health_factor
        default_flags = np.zeros(self.n_paths, dtype=bool)
        brownian_paths = np.zeros((self.n_paths, steps - 1, n_dim))

        tilted_drift = drift + theta * (volatility ** 2)

        for path_idx in range(self.n_paths):
            current_hf = initial_health_factor
            path_default = False
            for step in range(1, steps):
                normals = np.random.normal(size=n_dim)
                correlated_brownian = cholesky @ normals
                brownian_paths[path_idx, step - 1, :] = correlated_brownian * np.sqrt(self.dt)

                increment = tilted_drift * self.dt + volatility * brownian_paths[path_idx, step - 1, :]
                current_hf += np.sum(increment)
                health_factors[path_idx, step] = current_hf

                if current_hf <= self.liquidation_threshold and not path_default:
                    path_default = True

            default_flags[path_idx] = path_default

        return health_factors, default_flags, brownian_paths

    def compute_girsanov_weights(self, theta: np.ndarray, volatility: np.ndarray, brownian_increments: np.ndarray,
                                 t_max: float) -> np.ndarray:
        sum_theta_dW = np.sum(theta * brownian_increments, axis=(1, 2))
        integral_theta_sq = np.sum((theta * volatility) ** 2) * t_max
        weights = np.exp(-sum_theta_dW - 0.5 * integral_theta_sq)
        return weights

    def compute_importance_sampling_default_probability(self, default_flags: np.ndarray, weights: np.ndarray) -> float:
        return np.mean(default_flags * weights)

    def compute_default_probability(self, default_flags: np.ndarray, likelihood_weights: np.ndarray = None) -> float:
        if likelihood_weights is None:
            return np.mean(default_flags)
        return np.mean(default_flags * likelihood_weights)

    def evaluate_risk_surface(self, time_grid: np.ndarray, intensity_grid: np.ndarray, base_drifts: np.ndarray,
                              base_vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
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