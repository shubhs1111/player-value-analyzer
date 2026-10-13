"""Flatten the scored dataset into the two files the dashboard reads."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("dashboard_v2")
SEASONS = None   # every season in the scored file

CARD_STATS = {
    "GK": [("saves_p90", "Saves per 90"), ("goals_conceded_p90", "Goals conceded per 90"),
           ("clean_sheets", "Clean sheets"), ("passes_p90", "Accurate passes per 90"),
           ("long_balls_p90", "Accurate long balls per 90"), ("minutes_pct", "Share of team minutes")],
    "DF": [("tackles_p90", "Tackles per 90"), ("interceptions_p90", "Interceptions per 90"),
           ("clearances_p90", "Clearances per 90"), ("blocks_p90", "Blocks per 90"),
           ("recoveries_p90", "Ball recoveries per 90"), ("passes_p90", "Accurate passes per 90")],
    "MF": [("passes_p90", "Accurate passes per 90"), ("chances_p90", "Chances created per 90"),
           ("xa_p90", "Expected assists per 90"), ("interceptions_p90", "Interceptions per 90"),
           ("tackles_p90", "Tackles per 90"), ("recoveries_p90", "Ball recoveries per 90")],
    "FW": [("goals_p90", "Goals per 90"), ("assists_p90", "Assists per 90"),
           ("xg_p90", "Expected goals per 90"), ("sot_p90", "Shots on target per 90"),
           ("big_chances_p90", "Big chances created per 90"), ("dribbles_p90", "Dribbles won per 90")],
}
POS_LABEL = {"GK": "Goalkeepers", "DF": "Defenders", "MF": "Midfielders", "FW": "Forwards"}

SUFFIX = re.compile(r"\s*(S\.?p\.?A\.?|S\.?A\.?D\.?|Football Club|Futebol Clube|Calcio)\.?\s*$", re.I)


def tidy_club(x: str) -> str:
    x = str(x).replace("�", "").strip()
    for _ in range(2):
        x = SUFFIX.sub("", x).strip(" .-")
    return x


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv("data/scored_current.csv", low_memory=False)
    if SEASONS:
        df = df[df["season"].isin(SEASONS)]
    df = df.copy()
    df["minutes_pct"] = (df["minutes_share"] * 100).round(0)

    # percentile of share-of-minutes, for the goalkeeper card
    df["minutes_pct_pct"] = (df[df["minutes_share"] >= 0.25]
                             .groupby(["season", "pos_group"])["minutes_share"].rank(pct=True) * 100)
    df["clean_sheets_pct"] = (df[df["minutes_share"] >= 0.25]
                              .groupby(["season", "pos_group"])["clean_sheets"].rank(pct=True) * 100)

    out = pd.DataFrame({
        "id": (df["fotmob_id"].astype(str) + "-" + df["season_start"].astype(str)
               + "-" + df["league"].map({"Premier League": "PL", "La Liga": "LL", "Serie A": "SA",
                                         "Bundesliga": "BL", "Ligue 1": "L1"}).fillna("XX")),
        "player": df["player"],
        "team": df["team"].map(tidy_club),
        "league": df["league"],
        "pos": df["pos_group"],
        "season": df["season"],
        "age": df["age"].round(1),
        "matches": df["matches"].astype(int),
        "minutes": df["minutes"].astype(int),
        "minutes_share": df["minutes_share"].round(3),
        "regular": df["minutes_share"] >= 0.5,
        "market_m": (df["market_value_eur"] / 1e6).round(2),
        "model_m": (df["model_value_eur"] / 1e6).round(2),
        "lo_m": (df["model_value_lo_eur"] / 1e6).round(2),
        "hi_m": (df["model_value_hi_eur"] / 1e6).round(2),
        "gap_m": (df["value_gap_eur"] / 1e6).round(2),
        "ratio": df["value_ratio"].round(2),
    })

    # six stats per player, chosen by his position
    for i in range(1, 7):
        out[f"s{i}_val"] = np.nan
        out[f"s{i}_pct"] = np.nan
    for pos, stats in CARD_STATS.items():
        m = (df["pos_group"] == pos).to_numpy()
        for i, (col, label) in enumerate(stats, start=1):
            if col not in df.columns:
                continue
            dec = 0 if col in ("clean_sheets", "minutes_pct") else 2
            out.loc[m, f"s{i}_val"] = df.loc[m, col].round(dec)
            pct_col = col + "_pct"
            if pct_col in df.columns:
                out.loc[m, f"s{i}_pct"] = df.loc[m, pct_col].round(0)

    # why the model landed where it did (explain_positions.py)
    exp_path = Path("data/explanations.csv")
    if exp_path.exists():
        ex = pd.read_csv(exp_path)
        ex = ex.rename(columns={"player_key": "id"}).drop_duplicates("id")
        out = out.merge(ex, on="id", how="left", validate="one_to_one")
        print(f"  merged reasons for {out['perf1'].notna().sum()} player-seasons")

    out = out.dropna(subset=["market_m", "model_m"])
    out.to_csv(OUT / "players.csv", index=False, encoding="utf-8")
    print(f"players.csv {out.shape}  ({(OUT / 'players.csv').stat().st_size // 1024} KB)")
    print(out.groupby("season").agg(n=("id", "size"), regulars=("regular", "sum")).to_string())

    # accuracy, per position
    metrics = json.loads(Path("data/metrics_positions.json").read_text(encoding="utf-8"))["positions"]
    rows = []
    for pos, v in metrics.items():
        rows.append({
            "pos": pos,
            "pos_label": POS_LABEL[pos],
            "n": v["with_stats"]["n"],
            "err_nostats": v["baseline"]["median_abs_pct_err"],
            "err_withstats": v["with_stats"]["median_abs_pct_err"],
            "gain": v["improvement_pct_points"],
            "r2": v["with_stats"]["r2_log"],
            "within30": v["with_stats"]["within_30pct"],
            "coverage": v["range_coverage"],
            "stats_used": ", ".join(v["features"][:4]),
        })
    # one more row for the whole held-out season, so the page can lead with a single number
    hp = pd.read_csv("data/holdout_predictions.csv")
    err = (hp["model_m"] - hp["market_m"]).abs() / hp["market_m"] * 100
    rows.append({"pos": "ALL", "pos_label": "All players", "n": len(hp),
                 "err_nostats": np.nan, "err_withstats": round(float(err.median()), 1),
                 "gain": np.nan, "r2": np.nan,
                 "within30": round(float((err <= 30).mean() * 100), 1),
                 "coverage": np.nan, "stats_used": ""})

    acc = pd.DataFrame(rows).sort_values("pos")
    # pre-formatted, so what the page prints matches the data exactly
    acc["err_display"] = acc["err_withstats"].round(0).astype("Int64").astype(str) + "%"
    acc["within30_display"] = acc["within30"].round(0).astype("Int64").astype(str) + "%"
    acc["n_display"] = acc["n"].map(lambda v: f"{int(v):,}")
    acc["gain_display"] = acc["gain"].map(
        lambda v: "" if pd.isna(v) else f"{v:+.0f} points")
    acc.to_csv(OUT / "accuracy.csv", index=False, encoding="utf-8")
    print("\n" + acc[["pos_label", "n", "err_nostats", "err_withstats", "gain"]].to_string(index=False))


def write_worked_example() -> None:
    """One player who shows what the two models actually do, picked on the numbers."""
    hp = pd.read_csv("data/holdout_predictions.csv")
    hp["team"] = hp["team"].map(tidy_club)
    hp["off_full"] = ((hp["model_m"] - hp["market_m"]).abs() / hp["market_m"] * 100).round(0)
    hp["off_basic"] = ((hp["basic_m"] - hp["market_m"]).abs() / hp["market_m"] * 100).round(0)
    hp["gain"] = hp["off_basic"] - hp["off_full"]
    # a mid-market player, where knowing the football actually changes the answer
    pool = hp[(hp["market_m"].between(12, 80)) & (hp["off_full"] <= 12)]
    pick = pool.nlargest(1, "gain")
    cols = ["player", "team", "league", "pos", "age", "season",
            "market_m", "basic_m", "off_basic", "model_m", "off_full"]
    pick[cols].to_csv(OUT / "worked_example.csv", index=False, encoding="utf-8")
    r = pick.iloc[0]
    print(f"worked example: {r['player']} — market {r['market_m']}m, "
          f"basic {r['basic_m']}m ({r['off_basic']:.0f}%), with stats {r['model_m']}m ({r['off_full']:.0f}%)")


def write_examples() -> None:
    """A handful of recognisable players the models priced without ever seeing them."""
    hp = pd.read_csv("data/holdout_predictions.csv")
    hp["team"] = hp["team"].map(tidy_club)
    hp["off_pct"] = ((hp["model_m"] - hp["market_m"]).abs() / hp["market_m"] * 100).round(0)
    famous = hp.nlargest(10, "market_m")
    mid = hp[(hp["market_m"].between(8, 40))].nlargest(5, "market_m")
    ex = pd.concat([famous, mid]).drop_duplicates(subset=["player", "team"]).head(14)
    ex = ex.sort_values("market_m", ascending=False)
    ex[["player", "team", "league", "pos", "age", "season", "market_m", "model_m", "off_pct"]].to_csv(
        OUT / "examples.csv", index=False, encoding="utf-8")
    print(f"examples.csv: {len(ex)} players, median miss {ex['off_pct'].median():.0f}%")


if __name__ == "__main__":
    main()
    write_examples()
    write_worked_example()
