import numpy as np


class VWAPLiquidator:
    def __init__(self, max_slippage: float = 0.50):
        self.max_slippage = max_slippage

    def calculate_realized_proceeds(
        self, position_size: float, bid_prices: np.ndarray, bid_depths: np.ndarray
    ) -> float:
        remaining_size = position_size
        total_proceeds = 0.0

        for price, depth in zip(bid_prices, bid_depths):
            if remaining_size <= 0:
                break

            executed_volume = min(remaining_size, depth)
            total_proceeds += executed_volume * price
            remaining_size -= executed_volume

        return total_proceeds

    def compute_liquidity_discount(
        self, mid_price: float, position_size: float, realized_proceeds: float
    ) -> float:
        ideal_proceeds = position_size * mid_price
        fill_rate = min(1.0, realized_proceeds / ideal_proceeds)

        average_execution_price = (
            realized_proceeds / (position_size * fill_rate) if fill_rate > 0 else 0.0
        )

        slippage_discount = ((mid_price - average_execution_price) / mid_price) * fill_rate
        unfilled_discount = 1.0 - fill_rate

        return min(self.max_slippage, slippage_discount + unfilled_discount)

    def calculate_stressed_recovery(
        self,
        base_discount: float,
        adverse_price_move: float,
        stress_calibration: float,
        scale_factor: float,
    ) -> float:
        shock_component = 1.0 - (
            stress_calibration * (1.0 - np.exp(-adverse_price_move / scale_factor))
        )
        return 1.0 - ((1.0 - base_discount) * shock_component)
