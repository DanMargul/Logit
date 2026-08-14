import numpy as np

class APREngine:
    def __init__(self, base_rate: float, stress_scaling: float, pricing_delta: float):
        self.base_rate = base_rate
        self.stress_scaling = stress_scaling
        self.pricing_delta = pricing_delta

    def calculate_risk_premium(self, default_prob: float, expected_lgd: float) -> float:
        expected_loss = default_prob * expected_lgd
        expected_survival = 1.0 - default_prob
        return expected_loss / (expected_survival * self.pricing_delta)

    def compute_mad_z_score(self, values: np.ndarray) -> np.ndarray:
        median_val = np.median(values)
        absolute_deviations = np.abs(values - median_val)
        mad = np.median(absolute_deviations)
        
        if mad == 0.0:
            return np.zeros_like(values)
            
        return (values - median_val) / mad

    def calculate_position_stress_scores(
        self, 
        volatilities: np.ndarray, 
        spreads: np.ndarray, 
        depths: np.ndarray, 
        volumes: np.ndarray
    ) -> np.ndarray:
        depth_stress_factor = -np.log1p(depths)
        activity_stress_factor = -np.log1p(volumes)
        
        z_volatility = self.compute_mad_z_score(volatilities)
        z_spread = self.compute_mad_z_score(spreads)
        z_depth = self.compute_mad_z_score(depth_stress_factor)
        z_activity = self.compute_mad_z_score(activity_stress_factor)
        
        return 0.25 * (z_volatility + z_spread + z_depth + z_activity)

    def calculate_stress_premium(self, position_scores: np.ndarray, weights: np.ndarray) -> float:
        portfolio_stress_index = np.average(position_scores, weights=weights)
        return self.stress_scaling * portfolio_stress_index

    def calculate_liquidity_premium(
        self, 
        avg_recovery_discount: float, 
        avg_unfill_rate: float, 
        slippage_penalty: float, 
        unfill_penalty: float
    ) -> float:
        slippage_component = slippage_penalty * avg_recovery_discount
        unfill_component = unfill_penalty * avg_unfill_rate
        return slippage_component + unfill_component

    def calculate_total_apr(
        self, 
        default_prob: float, 
        expected_lgd: float,
        position_scores: np.ndarray,
        weights: np.ndarray,
        liquidity_premium: float,
        ltv_premium: float,
        expiry_premium: float
    ) -> float:
        risk_premium = self.calculate_risk_premium(default_prob, expected_lgd)
        stress_premium = self.calculate_stress_premium(position_scores, weights)
        
        return (
            self.base_rate 
            + risk_premium 
            + stress_premium 
            + liquidity_premium 
            + ltv_premium 
            + expiry_premium
        )
