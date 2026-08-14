import numpy as np
import polars as pl

from logit.core.microstructure import MicrostructurePipeline
from logit.core.transforms import to_logit, to_probability


def test_logit_transform_reversibility():
    input_prices = pl.Series("price", [0.1, 0.5, 0.9, 0.99])

    logits = to_logit(input_prices)
    recovered_prices = to_probability(logits)

    np.testing.assert_allclose(input_prices.to_numpy(), recovered_prices.to_numpy(), rtol=1e-5)


def test_logit_boundary_clamping():
    extreme_prices = pl.Series("price", [0.0, 1.0])

    logits = to_logit(extreme_prices, epsilon=1e-4)

    assert np.isfinite(logits[0])
    assert np.isfinite(logits[1])
    assert logits[0] < 0
    assert logits[1] > 0


def test_microstructure_pipeline_aggregation():
    pipeline = MicrostructurePipeline(min_bid_depth=1000.0, freq="1h")

    mock_data = pl.DataFrame(
        {
            "timestamp": ["2026-08-14T09:00:00", "2026-08-14T09:30:00", "2026-08-14T10:15:00"],
            "market_id": ["US_ELECTION", "US_ELECTION", "US_ELECTION"],
            "best_bid": [0.50, 0.52, 0.48],
            "best_ask": [0.52, 0.54, 0.50],
            "bid_depth": [5000.0, 2000.0, 500.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    processed = pipeline.process_snapshots(mock_data)

    assert len(processed) == 1
    assert processed["delta_x"][0] != 0.0
