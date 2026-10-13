"""
Train the market-value model on the modern (Transfermarkt-only) dataset.

Same design as train_model.py - log target, gradient-boosted trees, values
predicted relative to each season's market level - but a thinner feature set,
since appearance data has no xG, xAG or progression.

Split: train on 2012/13-2023/24, test on 2024/25 and 2025/26.

Outputs: models/value_model_modern.joblib, data/scored_player_seasons_modern.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path("data/player_seasons_modern.csv")
MODELS = Path("models")

NUMERIC = [
    "age", "minutes", "matches", "min_per_match", "n_clubs",
    "goals_p90", "assists_p90", "ga_p90", "yellow_p90", "red_p90",
    "team_ppm", "team_gd_per_game", "log_squad_value", "height_m",
]
CATEGORICAL = ["pos_group", "sub_position", "league", "foot"]
FEATURES = NUMERIC + CATEGORICAL

TRAIN_MAX_SEASON = 2023          # season start year -> trains through 2023/24
TEST_SEASONS = [2024, 2025]      # 2024/25 and 2025/26
MIN_MINUTES = 450
MAX_VALUE_AGE_DAYS = 300   # drop stale pre-season valuations (youth-era prices)
MIN_VALUE_EUR = 150_000

PRETTY = {
    "log_squad_value": "Club quality", "age": "Age", "minutes": "Minutes played",
    "matches": "Matches", "min_per_match": "Minutes per match", "n_clubs": "Clubs that season",
    "goals_p90": "Goals /90", "assists_p90": "Assists /90", "ga_p90": "Goals + assists /90",
    "yellow_p90": "Yellows /90", "red_p90": "Reds /90", "team_ppm": "Team points per match",
    "team_gd_per_game": "Team goal difference per game", "height_m": "Height",
    "pos_group": "Position", "sub_position": "Detailed position", "league": "League", "foot": "Foot",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    reg = df[df["minutes"] >= MIN_MINUTES]
    level = reg.groupby("season")["log_value"].median()
    df["season_level"] = df["season"].map(level)
    df["rel_log_value"] = df["log_value"] - df["season_level"]
    club_level = reg.groupby("season")["log_squad_value"].median()
    df["log_squad_value"] = df["log_squad_value"] - df["season"].map(club_level)
    return df


def make_model(loss: str = "squared_error", quantile: float | None = None):
    return HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, max_iter=600, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=40,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        random_state=42,
    )


def report(name: str, y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y_eur, p_eur = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(p_eur - y_eur) / y_eur
    m = {
        "n": int(len(y_log)),
        "r2_log": round(float(r2_score(y_log, pred_log)), 4),
        "mae_log": round(float(mean_absolute_error(y_log, pred_log)), 4),
        "median_abs_pct_err": round(float(np.median(ape)) * 100, 1),
        "within_30pct": round(float((ape <= 0.30).mean()) * 100, 1),
        "within_50pct": round(float((ape <= 0.50).mean()) * 100, 1),
        "spearman": round(float(spearmanr(y_eur, p_eur).statistic), 4),
    }
    print(f"[{name}] " + "  ".join(f"{k}={v}" for k, v in m.items()))
    return m


def main() -> None:
    df = load()
    fresh = (df["value_age_days"] <= MAX_VALUE_AGE_DAYS) & (df["market_value_eur"] >= MIN_VALUE_EUR)
    print(f"{int((~fresh).sum())} player-seasons dropped: stale or implausible valuation snapshot")
    fit_mask = ((df["minutes"] >= MIN_MINUTES) & df["log_value"].notna()
                & df["log_squad_value"].notna() & fresh)
    train = df[fit_mask & (df["season"] <= TRAIN_MAX_SEASON)]
    test = df[fit_mask & df["season"].isin(TEST_SEASONS)]
    print(f"train={len(train)} (through {TRAIN_MAX_SEASON}/{str(TRAIN_MAX_SEASON+1)[-2:]}), test={len(test)}\n")

    Xtr, ytr = train[FEATURES], train["rel_log_value"].to_numpy()
    Xte, yte = test[FEATURES], test["rel_log_value"].to_numpy()
    lvl_te = test["season_level"].to_numpy()

    base_feats = ["age", "minutes", "log_squad_value", "league", "pos_group"]
    base = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, random_state=42,
        categorical_features=[base_feats.index("league"), base_feats.index("pos_group")])
    base.fit(train[base_feats], ytr)
    metrics = {"baseline_age_minutes_club":
               report("baseline", yte + lvl_te, base.predict(test[base_feats]) + lvl_te)}

    model = make_model().fit(Xtr, ytr)
    test_pred = model.predict(Xte)
    metrics["full_model_test"] = report("full model (test)", yte + lvl_te, test_pred + lvl_te)

    by_season = {}
    for s in TEST_SEASONS:
        m = test["season"].to_numpy() == s
        if m.sum():
            by_season[int(s)] = report(f"season {s}/{str(s+1)[-2:]}", yte[m] + lvl_te[m], test_pred[m] + lvl_te[m])
    metrics["by_season"] = by_season

    by_pos = {}
    for p in ["GK", "DF", "MF", "FW"]:
        m = test["pos_group"].to_numpy() == p
        if m.sum():
            by_pos[p] = report(f"pos {p}", yte[m] + lvl_te[m], test_pred[m] + lvl_te[m])
    metrics["by_position"] = by_pos

    lo = make_model(loss="quantile", quantile=0.05).fit(Xtr, ytr)
    hi = make_model(loss="quantile", quantile=0.95).fit(Xtr, ytr)
    cover = ((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte))).mean()
    metrics["p05_p95_coverage_test"] = round(float(cover) * 100, 1)
    print(f"\n[interval] model range covers {cover:.1%} of held-out players")

    sample = test.sample(min(3000, len(test)), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample["rel_log_value"],
                                 n_repeats=5, random_state=0, scoring="r2")
    order = np.argsort(imp.importances_mean)[::-1]
    top = [(FEATURES[i], round(float(imp.importances_mean[i]), 4)) for i in order[:14]]
    metrics["top_features_permutation_r2_drop"] = top
    print("\n[importance] top drivers of market value:")
    for f, v in top:
        print(f"   {PRETTY.get(f, f):<30} {v:.4f}")

    # refit on everything, score every player-season
    yfit = df.loc[fit_mask, "rel_log_value"]
    final = make_model().fit(df.loc[fit_mask, FEATURES], yfit)
    final_lo = make_model(loss="quantile", quantile=0.05).fit(df.loc[fit_mask, FEATURES], yfit)
    final_hi = make_model(loss="quantile", quantile=0.95).fit(df.loc[fit_mask, FEATURES], yfit)

    full_mask = df["log_value"].notna() & df["log_squad_value"].notna() & fresh
    df["stale_value"] = ~fresh
    X_all, lvl_all = df.loc[full_mask, FEATURES], df.loc[full_mask, "season_level"]
    df.loc[full_mask, "pred_log"] = final.predict(X_all) + lvl_all
    df.loc[full_mask, "pred_lo_log"] = final_lo.predict(X_all) + lvl_all
    df.loc[full_mask, "pred_hi_log"] = final_hi.predict(X_all) + lvl_all
    df["model_value_eur"] = np.exp(df["pred_log"])
    df["model_value_lo_eur"] = np.exp(df["pred_lo_log"])
    df["model_value_hi_eur"] = np.exp(df["pred_hi_log"])
    df["residual_log"] = df["pred_log"] - df["log_value"]
    df["value_gap_eur"] = df["model_value_eur"] - df["market_value_eur"]
    df["value_ratio"] = df["model_value_eur"] / df["market_value_eur"]
    df["season"] = df["season"].astype(int)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "lo": final_lo, "hi": final_hi, "features": FEATURES,
                 "metrics": metrics}, MODELS / "value_model_modern.joblib")
    df.to_csv("data/scored_player_seasons_modern.csv", index=False, encoding="utf-8")
    Path("data/metrics_modern.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nwrote models/value_model_modern.joblib, data/scored_player_seasons_modern.csv")


if __name__ == "__main__":
    main()
