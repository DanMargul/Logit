import polars as pl


def to_logit(price: pl.Expr, epsilon: float = 1e-7) -> pl.Expr:
    clamped_price = price.clip(epsilon, 1.0 - epsilon)
    return (clamped_price / (1.0 - clamped_price)).log()


def to_probability(logit: pl.Expr) -> pl.Expr:
    return 1.0 / (1.0 + (-logit).exp())
