import numpy as np
import polars as pl

from logit.core.microstructure import MicrostructurePipeline
from logit.execution.vwap import VWAPLiquidator
from logit.models.optimal_transport import WassersteinRobustness


def run_risk_evaluation():
    pipeline = MicrostructurePipeline(min_bid_depth=2500.0)
    robustness_engine = WassersteinRobustness(confidence_level=0.01)
    liquidator = VWAPLiquidator()

    simulated_snapshots = pl.DataFrame(
        {
            "timestamp": ["2026-08-14T10:00:00", "2026-08-14T11:00:00", "2026-08-14T12:00:00"],
            "market_id": ["FED_RATE_CUT", "FED_RATE_CUT", "FED_RATE_CUT"],
            "best_bid": [0.65, 0.60, 0.58],
            "best_ask": [0.67, 0.62, 0.60],
            "bid_depth": [15000.0, 12000.0, 8000.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    processed_df = pipeline.process_snapshots(simulated_snapshots)
    scenario_matrix = pipeline.build_scenario_matrix(processed_df)

    position_sizes = np.array([50000.0])
    bids = np.array([0.58, 0.55, 0.50])
    depths = np.array([5000.0, 15000.0, 50000.0])

    proceeds = liquidator.calculate_realized_proceeds(position_sizes[0], bids, depths)
    base_discount = liquidator.compute_liquidity_discount(0.59, position_sizes[0], proceeds)

    lipschitz_k = robustness_engine.compute_structure_aware_lipschitz(
        position_sizes=position_sizes, recovery_discounts=np.array([base_discount])
    )

    wasserstein_radius = 0.15
    capital_addon = robustness_engine.calculate_capital_addon(lipschitz_k, wasserstein_radius)

    print("RISK EVALUATION REPORT")
    print("======================")
    print(f"Scenario Matrix Shape: {scenario_matrix.shape}")
    print(f"Base Liquidity Discount: {base_discount:.4f}")
    print(f"Lipschitz Constant (Binary Structure): {lipschitz_k:.2f}")
    print(f"Certified Capital Add-on: {capital_addon:.2f}")


if __name__ == "__main__":
    run_risk_evaluation()
