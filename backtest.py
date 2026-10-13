"""
Does the mispricing signal actually mean anything?

A valuation model is only useful if the players it calls underpriced go on to
*become* more expensive. This backtest groups every player-season by the model's
mispricing quintile and measures the player's market-value change over the
following season, relative to the market-wide move (so league-wide inflation
cannot flatter the result).

Run after train_model.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MINUTES = 900


def main() -> None:
    import sys
    era = sys.argv[1] if len(sys.argv) > 1 else "modern"
    path = {"modern": "data/scored_player_seasons_modern.csv",
            "fbref": "data/scored_player_seasons.csv"}[era]
    print("era: " + era + " (" + path + ")")
    df = pd.read_csv(path, low_memory=False)
    if "player_id" in df.columns and "tm_id" not in df.columns:
        df["tm_id"] = df["player_id"]
    if "season_label" in df.columns:
        df["Season_End_Year"] = df["season"].astype(int) + 1
    df = df[df["minutes"] >= MIN_MINUTES].copy()

    # market level per season, so growth is measured net of league-wide inflation
    level = df.groupby("Season_End_Year")["log_value"].median()
    df["log_value_rel"] = df["log_value"] - df["Season_End_Year"].map(level)

    nxt = df[["tm_id", "Season_End_Year", "log_value_rel", "market_value_eur", "age"]].copy()
    nxt["Season_End_Year"] -= 1
    nxt = nxt.rename(columns={"log_value_rel": "next_log_value_rel",
                              "market_value_eur": "next_value_eur",
                              "age": "next_age"})
    m = df.merge(nxt, on=["tm_id", "Season_End_Year"], how="inner")
    m["growth_log"] = m["next_log_value_rel"] - m["log_value_rel"]
    m["growth_pct"] = (m["next_value_eur"] / m["market_value_eur"] - 1) * 100

    m["quintile"] = pd.qcut(m["residual_log"], 5, labels=
                            ["most overvalued", "overvalued", "fair", "undervalued", "most undervalued"])

    print(f"n = {len(m)} player-seasons with a following season "
          f"({m['Season_End_Year'].min()}-{m['Season_End_Year'].max()})\n")
    g = m.groupby("quintile", observed=True).agg(
        n=("growth_log", "size"),
        median_growth_vs_market=("growth_log", lambda s: f"{(np.exp(s.median()) - 1) * 100:+.1f}%"),
        mean_growth_log=("growth_log", "mean"),
        pct_who_gained=("growth_log", lambda s: f"{(s > 0).mean() * 100:.0f}%"),
        median_age=("age", "median"),
        median_value=("market_value_eur", "median"),
    )
    g["mean_growth_log"] = g["mean_growth_log"].round(3)
    g["median_value"] = (g["median_value"] / 1e6).round(1).astype(str) + "m"
    print("Next-season market-value change, net of league-wide inflation:")
    print(g.to_string())

    rho = m[["residual_log", "growth_log"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman(mispricing signal, next-season value growth) = {rho:.3f}")

    top, bottom = m[m["quintile"] == "most undervalued"], m[m["quintile"] == "most overvalued"]
    spread = np.exp(top["growth_log"].median()) - np.exp(bottom["growth_log"].median())
    print(f"Spread between top and bottom quintile: {spread * 100:+.1f} percentage points")

    # control: does the signal survive inside age bands? (it must not just be "young players rise")
    print("\nSame test within age bands (median growth vs market):")
    m["age_band"] = pd.cut(m["age"], [0, 21, 24, 27, 30, 50],
                           labels=["<=21", "22-24", "25-27", "28-30", "31+"])
    pivot = m.pivot_table(index="age_band", columns="quintile", values="growth_log",
                          aggfunc="median", observed=True)
    print((np.exp(pivot) - 1).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
