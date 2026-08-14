import numpy as np
import polars as pl
from logit.core.microstructure import MicrostructurePipeline
from logit.models.optimal_transport import SinkhornSolver, WassersteinRobustness
from logit.models.hawkes import ExponentialHawkesKernel, SpectralMonitor
from logit.models.allocation import DependencyGraphClustering, CapitalAllocationOptimizer
from logit.models.apr import APREngine
from logit.execution.vwap import VWAPLiquidator

def main():
    pipeline = MicrostructurePipeline(min_bid_depth=2500.0, freq="1h")
    
    mock_snapshots = pl.DataFrame({
        "timestamp": [
            "2026-08-14T09:00:00", "2026-08-14T10:00:00", 
            "2026-08-14T11:00:00", "2026-08-14T12:00:00"
        ],
        "market_id": ["MARKET_A", "MARKET_A", "MARKET_A", "MARKET_A"],
        "best_bid": [0.60, 0.62, 0.58, 0.55],
        "best_ask": [0.62, 0.64, 0.60, 0.57],
        "bid_depth": [10000.0, 15000.0, 8000.0, 5000.0]
    }).with_columns(pl.col("timestamp").str.to_datetime())

    processed = pipeline.process_snapshots(mock_snapshots)
    scenario_matrix = pipeline.build_scenario_matrix(processed)

    empirical_data = scenario_matrix.select(pl.exclude("timestamp")).to_numpy()
    candidate_data = empirical_data + np.random.normal(0, 0.01, size=empirical_data.shape)

    sinkhorn = SinkhornSolver(epsilon=0.1)
    cost_matrix = sinkhorn.compute_cost_matrix(empirical_data, candidate_data)
    p_emp = np.ones(len(empirical_data)) / len(empirical_data)
    w_target = np.ones(len(candidate_data)) / len(candidate_data)
    transport_plan = sinkhorn.solve(p_emp, w_target, cost_matrix)

    robustness = WassersteinRobustness(confidence_level=0.01)
    lipschitz_k = robustness.compute_structure_aware_lipschitz(
        position_sizes=np.array([10000.0]),
        recovery_discounts=np.array([0.25])
    )
    capital_addon = robustness.calculate_capital_addon(lipschitz_k, 0.15)

    liquidator = VWAPLiquidator()
    proceeds = liquidator.calculate_realized_proceeds(
        position_size=10000.0,
        bid_prices=np.array([0.55, 0.50]),
        bid_depths=np.array([4000.0, 10000.0])
    )
    base_discount = liquidator.compute_liquidity_discount(0.58, 10000.0, proceeds)

    clustering = DependencyGraphClustering()
    mock_branching = np.array([[0.1, 0.2], [0.2, 0.1]])
    mock_corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    mock_semantic = np.array([[1.0, 0.8], [0.8, 1.0]])
    affinity = clustering.build_affinity_matrix(mock_branching, mock_corr, mock_semantic)
    laplacian = clustering.compute_normalized_laplacian(affinity)
    embedding = clustering.extract_spectral_embedding(laplacian, n_clusters=2)

    optimizer = CapitalAllocationOptimizer()
    allocations = optimizer.solve_allocation(
        expected_returns=np.array([0.15, 0.18]),
        borrower_caps=np.array([50000.0, 50000.0]),
        pool_capacity=100000.0,
        market_exposures=np.array([[1.0, 0.0], [0.0, 1.0]]),
        market_caps=np.array([60000.0, 60000.0]),
        cluster_limits=np.array([75000.0, 75000.0]),
        cluster_membership=np.array([0, 1]),
        treasury_equity=25000.0,
        expected_losses=np.array([1000.0, 1200.0]),
        hawkes_multiplier=1.5
    )

    apr_engine = APREngine(base_rate=0.05, stress_scaling=0.02, pricing_delta=1.0)
    risk_prem = apr_engine.calculate_risk_premium(0.02, 0.30)
    total_apr = apr_engine.calculate_total_apr(
        default_prob=0.02,
        expected_lgd=0.30,
        position_scores=np.array([0.5]),
        weights=np.array([1.0]),
        liquidity_premium=0.03,
        ltv_premium=0.01,
        expiry_premium=0.005
    )

    print("PIPELINE EXECUTION COMPLETE")
    print(f"Transport Plan Shape: {transport_plan.shape}")
    print(f"Certified Capital Add-on: {capital_addon:.2f}")
    print(f"Base Liquidity Discount: {base_discount:.4f}")
    print(f"Optimal Allocations: {allocations}")
    print(f"Computed Borrower APR: {total_apr:.4f}")

if __name__ == "__main__":
    main()
