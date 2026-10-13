"""Export the scored dataset and validation results as flat files for the dashboard."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("dashboard_data")
OUT.mkdir(exist_ok=True)

import sys
ERA = sys.argv[1] if len(sys.argv) > 1 else "modern"
SRC = {"modern": ("data/scored_player_seasons_modern.csv", "data/metrics_modern.json"),
       "fbref": ("data/scored_player_seasons.csv", "data/metrics.json")}[ERA]
print("era:", ERA)
df = pd.read_csv(SRC[0], low_memory=False)
metrics = json.loads(Path(SRC[1]).read_text(encoding="utf-8"))
if "season_label" in df.columns:                      # modern schema -> common names
    df["season"] = df["season_label"]
    df["tm_squad"] = df["club"]
    df["slug"] = df["player_id"].astype(str)
    df["tm_id"] = df["player_id"]
if "Season_End_Year" not in df.columns:
    df["Season_End_Year"] = df["season"].str[:4].astype(int) + 1

# ---------------------------------------------------------------- players ----
p = df[df["minutes"] >= 900].copy()
p["id"] = p["slug"] + "-" + p["Season_End_Year"].astype(str)
p["player"] = p["player_name"]
SUFFIX = re.compile(r"\s*(S\.?p\.?A\.?|S\.?A\.?D\.?|A\.?S\.?D\.?|Football Club|Futebol Clube|"
                    r"Calcio|Associazione Sportiva|Società Sportiva|Sportiva)\.?\s*$", re.I)
def tidy_club(x: str) -> str:
    x = str(x).replace("�", "").strip()
    x = re.sub(r"^(Società Sportiva|Associazione Sportiva|Football Club|Futebol Clube)\s+",
               "", x, flags=re.I)
    for _ in range(3):
        x = SUFFIX.sub("", x).strip(" .-")
    return x

p["club"] = p["tm_squad"].map(tidy_club)
p = p[p["model_value_eur"].notna()]
p["pos"] = p["pos_group"]
out = pd.DataFrame({
    "id": p["id"],
    "player": p["player"],
    "season": p["season"],
    "season_end": p["Season_End_Year"],
    "club": p["club"],
    "league": p["league"],
    "pos": p["pos"],
    "age": p["age"].round(1),
    "minutes": p["minutes"].astype(int),
    "ga90": p["ga_p90"].round(2),
    "market_m": (p["market_value_eur"] / 1e6).round(2),
    "model_m": (p["model_value_eur"] / 1e6).round(2),
    "lo_m": (p["model_value_lo_eur"] / 1e6).round(2),
    "hi_m": (p["model_value_hi_eur"] / 1e6).round(2),
    "gap_m": (p["value_gap_eur"] / 1e6).round(2),
    "ratio": p["value_ratio"].round(2),
})
out = out.dropna(subset=["market_m", "model_m", "age"]).drop_duplicates("id")
out.to_csv(OUT / "players.csv", index=False, encoding="utf-8")
print("players.csv", out.shape, (OUT / "players.csv").stat().st_size // 1024, "KB")

# --------------------------------------------------------------- accuracy ----
t, b = metrics["full_model_test"], metrics["baseline_age_minutes_club"]
rows = [
    {"metric": "player_seasons", "label": "Player-seasons analysed",
     "value": int(len(df)), "display": f"{len(df):,}"},
    {"metric": "latest_season", "label": "Latest season",
     "value": 0, "display": str(sorted(df["season"].unique())[-1])},
    {"metric": "seasons", "label": "Complete seasons",
     "value": int(df["Season_End_Year"].nunique()), "display": str(df["Season_End_Year"].nunique())},
    {"metric": "test_n", "label": "Held-out player-seasons", "value": t["n"], "display": f"{t['n']:,}"},
    {"metric": "median_error", "label": "Median error, held-out seasons",
     "value": t["median_abs_pct_err"], "display": f"{t['median_abs_pct_err']}%"},
    {"metric": "r2", "label": "R2 on log value", "value": t["r2_log"], "display": f"{t['r2_log']:.2f}"},
    {"metric": "spearman", "label": "Rank correlation", "value": t["spearman"], "display": f"{t['spearman']:.2f}"},
    {"metric": "within_30", "label": "Within 30% of market", "value": t["within_30pct"], "display": f"{t['within_30pct']}%"},
    {"metric": "coverage", "label": "Players inside the model range",
     "value": metrics["p05_p95_coverage_test"], "display": f"{metrics['p05_p95_coverage_test']}%"},
    {"metric": "baseline_error", "label": "Median error, baseline model",
     "value": b["median_abs_pct_err"], "display": f"{b['median_abs_pct_err']}%"},
    {"metric": "baseline_r2", "label": "Baseline R2", "value": b["r2_log"], "display": f"{b['r2_log']:.2f}"},
]
pd.DataFrame(rows).to_csv(OUT / "accuracy.csv", index=False, encoding="utf-8")

# position breakdown
pos_rows = [{"pos": k, "r2": v["r2_log"], "median_error": v["median_abs_pct_err"], "n": v["n"]}
            for k, v in metrics["by_position"].items()]
pd.DataFrame(pos_rows).to_csv(OUT / "accuracy_by_position.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------- importance ----
imp = pd.DataFrame(metrics["top_features_permutation_r2_drop"], columns=["feature", "importance"])
PRETTY = {
    "log_squad_value": "Club quality", "age": "Age", "minutes": "Minutes played",
    "starts_share": "Share of starts", "ga_p90": "Goals + assists /90",
    "team_ppm": "Team points per match", "league": "League", "sot_p90": "Shots on target /90",
    "pos_group": "Position", "years_at_club": "Years at club", "pass_cmp_pct": "Pass completion %",
    "xg_p90": "xG /90", "clr_p90": "Clearances /90", "contract_years_left": "Contract years left",
    "height_m": "Height", "shots_p90": "Shots /90", "goals_p90": "Goals /90",
    "tkl_int_p90": "Tackles + interceptions /90",
    "sub_position": "Detailed position", "min_per_match": "Minutes per match",
    "matches": "Matches", "n_clubs": "Clubs that season", "assists_p90": "Assists /90",
    "team_gd_per_game": "Team goal difference", "yellow_p90": "Yellows /90",
    "red_p90": "Reds /90", "foot": "Foot",
}
imp["label"] = imp["feature"].map(PRETTY).fillna(imp["feature"])
imp["group"] = np.where(imp["feature"].isin(["log_squad_value", "team_ppm", "league"]), "Context",
                 np.where(imp["feature"].isin(["age", "height_m", "contract_years_left", "years_at_club"]), "Age & contract",
                 np.where(imp["feature"].isin(["minutes", "starts_share", "pos_group", "matches",
                                              "min_per_match", "sub_position", "n_clubs"]),
                               "Playing time", "Performance")))
imp.head(12).to_csv(OUT / "importance.csv", index=False, encoding="utf-8")

# --------------------------------------------------------------- backtest ----
bt = df[df["minutes"] >= 900].copy()
level = bt.groupby("Season_End_Year")["log_value"].median()
bt["rel"] = bt["log_value"] - bt["Season_End_Year"].map(level)
nxt = bt[["tm_id", "Season_End_Year", "rel"]].copy()
nxt["Season_End_Year"] -= 1
nxt = nxt.rename(columns={"rel": "next_rel"})
m = bt.merge(nxt, on=["tm_id", "Season_End_Year"], how="inner")
m["growth"] = m["next_rel"] - m["rel"]
labels = ["Most overvalued", "Overvalued", "Fair", "Undervalued", "Most undervalued"]
m["bucket"] = pd.qcut(m["residual_log"], 5, labels=labels)

g = m.groupby("bucket", observed=True).agg(
    n=("growth", "size"),
    growth_pct=("growth", lambda s: round((np.exp(s.median()) - 1) * 100, 1)),
    gained_pct=("growth", lambda s: round(float((s > 0).mean()) * 100, 0)),
).reset_index().rename(columns={"bucket": "verdict"})
g.to_csv(OUT / "backtest.csv", index=False, encoding="utf-8")

m["age_band"] = pd.cut(m["age"], [0, 21, 24, 27, 30, 99],
                       labels=["21 and under", "22-24", "25-27", "28-30", "31+"])
ab = (m.groupby(["age_band", "bucket"], observed=True)["growth"]
        .median().reset_index())
ab["growth_pct"] = ((np.exp(ab["growth"]) - 1) * 100).round(1)
ab = ab.drop(columns="growth").rename(columns={"age_band": "age_band", "bucket": "verdict"})
ab.to_csv(OUT / "backtest_by_age.csv", index=False, encoding="utf-8")

print("done:", sorted(f.name for f in OUT.iterdir()))
