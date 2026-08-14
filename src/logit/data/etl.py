import urllib.request
import json
import polars as pl

class PolymarketGammaClient:
    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.base_url = base_url

    def fetch_markets(self, limit: int = 100) -> list[dict]:
        url = f"{self.base_url}/markets?limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Logit-Quant-Engine"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data


class PMXTArchiveLoader:
    def __init__(self, base_archive_url: str = "https://r2v2.pmxt.dev"):
        self.base_archive_url = base_archive_url

    def load_snapshot_parquet(self, file_path_or_url: str) -> pl.DataFrame:
        df = pl.read_parquet(file_path_or_url)
        return df

    def filter_eligible_markets(self, df_snapshots: pl.DataFrame, min_bid_depth: float = 5000.0) -> pl.DataFrame:
        return df_snapshots.filter(pl.col("bid_depth") >= min_bid_depth)
