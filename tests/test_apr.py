import numpy as np
from logit.models.apr import APREngine

def test_apr_calculations():
    engine = APREngine(base_rate=0.05, stress_scaling=0.02, pricing_delta=1.0)
    risk_prem = engine.calculate_risk_premium(0.02, 0.3)
    scores = np.array([1.0, 2.0])
    weights = np.array([0.5, 0.5])
    stress_prem = engine.calculate_stress_premium(scores, weights)
    total = engine.calculate_total_apr(
        default_prob=0.02,
        expected_lgd=0.3,
        position_scores=scores,
        weights=weights,
        liquidity_premium=0.03,
        ltv_premium=0.01,
        expiry_premium=0.005
    )
    assert risk_prem > 0.0
    assert stress_prem > 0.0
    assert total > 0.0
