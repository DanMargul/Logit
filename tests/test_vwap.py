import numpy as np
from logit.execution.vwap import VWAPLiquidator

def test_vwap_realized_proceeds():
    liquidator = VWAPLiquidator(max_slippage=0.5)
    bids = np.array([0.6, 0.5, 0.4])
    depths = np.array([1000.0, 2000.0, 5000.0])
    proceeds = liquidator.calculate_realized_proceeds(1500.0, bids, depths)
    expected = (1000.0 * 0.6) + (500.0 * 0.5)
    assert np.isclose(proceeds, expected)

def test_stressed_recovery():
    liquidator = VWAPLiquidator()
    recovery = liquidator.calculate_stressed_recovery(0.1, 0.5, 0.2, 1.0)
    assert 0.0 <= recovery <= 1.0
