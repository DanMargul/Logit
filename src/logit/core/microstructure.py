import polars as pl
from logit.transforms import to_logit

class MicrostructurePipeline:
    def __init__(self, min_bid_depth: float = 5000.0, freq: str = "1h"):
        self.min_bid_depth = min_bid_depth
        self.freq = freq
        
    def process_snapshots(self, df_snapshots: pl.DataFrame) -> pl.DataFrame:
        """
        Processes raw orderbook snapshots into synchronized hourly log-odds returns.
        
        Expected columns in df_snapshots:
        - timestamp: datetime
        - market_id: str
        - best_bid: float
        - best_ask: float
        - bid_depth: float
        """
        # 1. Filter out illiquid orderbook states
        filtered_df = df_snapshots.filter(
            pl.col("bid_depth") >= self.min_bid_depth
        )
        
        # 2. Calculate Mid-Price and Logit 
        processed_df = filtered_df.with_columns(
            mid_price=(pl.col("best_bid") + pl.col("best_ask")) / 2.0
        ).with_columns(
            logit_price=to_logit(pl.col("mid_price"))
        )
        
        # 3. Aggregate to specified frequency (Hourly OHLCV equivalent)
        aggregated = (
            processed_df
            .sort("timestamp")
            .group_by_dynamic("timestamp", every=self.freq, group_by="market_id")
            .agg(
                open_logit=pl.col("logit_price").first(),
                high_logit=pl.col("logit_price").max(),
                low_logit=pl.col("logit_price").min(),
                close_logit=pl.col("logit_price").last(),
            )
        )
        
        # 4. Calculate Log-odds Return (Delta X)
        final_df = aggregated.with_columns(
            delta_x=(pl.col("close_logit") - pl.col("open_logit"))
        )
        
        return final_df

    def build_scenario_matrix(self, df_processed: pl.DataFrame) -> pl.DataFrame:
        """
        Pivots the processed dataframe to create the M x n scenario matrix 
        required for Entropy Pooling and OT Stress Testing.
        """
        scenario_matrix = df_processed.pivot(
            values="delta_x",
            index="timestamp",
            columns="market_id",
            aggregate_function="first"
        ).drop_nulls()
        
        return scenario_matrix
