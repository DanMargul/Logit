import numpy as np
from scipy.optimize import minimize

class MultivariateHawkesCalibrator:
    def __init__(self, eta_g: float = 0.01):
        self.eta_g = eta_g

    def negative_log_likelihood(self, params: np.ndarray, event_times: list[np.ndarray], T: float, n_markets: int) -> float:
        mu = params[:n_markets]
        alpha_flat = params[n_markets:]
        alpha = alpha_flat.reshape((n_markets, n_markets))
        
        if np.any(mu < 0) or np.any(alpha < 0):
            return 1e10
            
        total_log_likelihood = 0.0
        total_integral = 0.0
        
        for i in range(n_markets):
            ti = event_times[i]
            if len(ti) == 0:
                continue
                
            lambdas = np.zeros_like(ti)
            for idx, t in enumerate(ti):
                lambda_val = mu[i]
                for j in range(n_markets):
                    tj = event_times[j]
                    past_events = tj[tj < t]
                    if len(past_events) > 0:
                        lambda_val += np.sum(alpha[i, j] * np.exp(-1.0 * (t - past_events)))
                lambdas[idx] = max(lambda_val, 1e-6)
                
            total_log_likelihood += np.sum(np.log(lambdas))
            
            integral = mu[i] * T
            for j in range(n_markets):
                tj = event_times[j]
                if len(tj) > 0:
                    integral += np.sum(alpha[i, j] * (1.0 - np.exp(-1.0 * (T - tj))))
            total_integral += integral
            
        penalty = self.eta_g * np.sum(np.abs(alpha))
        nll = -(total_log_likelihood - total_integral) + penalty
        return nll

    def fit(self, event_times: list[np.ndarray], T: float) -> tuple[np.ndarray, np.ndarray]:
        n_markets = len(event_times)
        initial_mu = np.full(n_markets, 0.1)
        initial_alpha = np.full((n_markets, n_markets), 0.05) / n_markets
        initial_params = np.concatenate([initial_mu, initial_alpha.flatten()])
        
        bounds = [(1e-5, None)] * n_markets + [(0.0, None)] * (n_markets ** 2)
        
        result = minimize(
            self.negative_log_likelihood,
            initial_params,
            args=(event_times, T, n_markets),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        fitted_mu = result.x[:n_markets]
        fitted_alpha = result.x[n_markets:].reshape((n_markets, n_markets))
        return fitted_mu, fitted_alpha
