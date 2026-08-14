import polars as pl
from logit.data.etl import PMXTArchiveLoader

def test_pmxt_loader_filtering():
    loader = PMXTArchiveLoader()
    mock_data = pl.DataFrame({
        "timestamp": ["2026-07-01T00:00:00", "2026-07-01T01:00:00"],
        "market_id": ["M1", "M2"],
        "bid_depth": [4000.0, 15000.0]
    }).with_columns(pl.col("timestamp").str.to_datetime())

    filtered = loader.filter_eligible_markets(mock_data, min_bid_depth=5000.0)
    assert len(filtered) == 1
    assert filtered["market_id"][0] == "M2"
