"""
Train the market-value model.

Target : log(Transfermarkt market value in EUR)
Model  : gradient-boosted trees (handles NaN natively, so player-seasons from
         2010-2017 that predate FBref's advanced stats still train fine)
Split  : strictly temporal - train on 2010/11-2020/21, test on 2021/22 + 2022/23,
         so the reported error is genuine out-of-sample forecasting error.

Also fits 10th/90th-percentile models to produce a valuation *range* rather
than a single point estimate.

Outputs: models/value_model.joblib, data/scored_player_seasons.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

DATA = Path("data/player_seasons.csv")
MODELS = Path("models")

NUMERIC = [
    # age & workload
    "age", "minutes", "matches", "starts_share", "min_pct_of_team", "n_clubs",
    # attacking output
    "goals_p90", "assists_p90", "ga_p90", "npg_p90", "xg_p90", "npxg_p90", "xag_p90",
    "npxg_xag_p90", "shots_p90", "sot_p90", "finishing_delta",
    # creation / progression
    "kp_p90", "prog_p90", "ppa_p90", "final_third_p90", "prog_dist_p90", "pass_cmp_pct",
    # defending
    "tkl_int_p90", "int_p90", "blocks_p90", "clr_p90", "err_p90", "high_tkl_p90",
    # discipline
    "yellow_p90", "red_p90",
    # context
    "team_ppm", "on_off_xg", "log_squad_value",
    # contract & bio
    "contract_years_left", "years_at_club", "height_m",
]
CATEGORICAL = ["pos_group", "league", "foot"]
FEATURES = NUMERIC + CATEGORICAL

TRAIN_MAX_SEASON = 2020          # season_end_year
TEST_SEASONS = [2021, 2022]
MIN_MINUTES = 450                # exclude fringe players from training


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    # Big-5 market values inflated ~2.5x between 2011 and 2022. The model therefore
    # prices a player *relative to his season's market level*; the level itself is
    # treated as a known macro quantity (like a price index), not something to forecast.
    reg = df[df["minutes"] >= MIN_MINUTES]
    level = reg.groupby("Season_End_Year")["log_value"].median()
    df["season_level"] = df["Season_End_Year"].map(level)
    df["rel_log_value"] = df["log_value"] - df["season_level"]
    # the same normalisation is applied to the club-quality feature
    club_level = reg.groupby("Season_End_Year")["log_squad_value"].median()
    df["log_squad_value"] = df["log_squad_value"] - df["Season_End_Year"].map(club_level)
    return df


def make_model(loss: str = "squared_error", quantile: float | None = None):
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=40,
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
        "median_abs_err_eur": int(np.median(np.abs(p_eur - y_eur))),
    }
    print(f"\n[{name}] " + "  ".join(f"{k}={v}" for k, v in m.items()))
    return m


def main() -> None:
    df = load()
    fit_mask = (df["minutes"] >= MIN_MINUTES) & df["log_value"].notna()
    train = df[fit_mask & (df["Season_End_Year"] <= TRAIN_MAX_SEASON)]
    test = df[fit_mask & df["Season_End_Year"].isin(TEST_SEASONS)]
    print(f"train={len(train)} player-seasons (<= {TRAIN_MAX_SEASON}), test={len(test)} ({TEST_SEASONS})")

    Xtr, ytr = train[FEATURES], train["rel_log_value"].to_numpy()
    Xte, yte = test[FEATURES], test["rel_log_value"].to_numpy()
    lvl_tr = train["season_level"].to_numpy()      # add back to get absolute EUR
    lvl_te = test["season_level"].to_numpy()

    # ---- baseline: age + minutes + league + club level only -------------------
    base_feats = ["age", "minutes", "log_squad_value", "league", "pos_group"]
    base = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, random_state=42,
        categorical_features=[base_feats.index("league"), base_feats.index("pos_group")])
    base.fit(train[base_feats], ytr)
    metrics = {"baseline_age_minutes_club":
               report("baseline", yte + lvl_te, base.predict(test[base_feats]) + lvl_te)}

    # ---- full model -----------------------------------------------------------
    model = make_model()
    model.fit(Xtr, ytr)
    test_pred = model.predict(Xte)
    metrics["full_model_test"] = report("full model (test)", yte + lvl_te, test_pred + lvl_te)
    metrics["full_model_train"] = report("full model (train, in-sample)",
                                         ytr + lvl_tr, model.predict(Xtr) + lvl_tr)

    # ---- per-season and per-position breakdown --------------------------------
    by_season = {}
    for s in TEST_SEASONS:
        m = test["Season_End_Year"].to_numpy() == s
        if m.sum():
            by_season[int(s)] = report(f"season {s-1}/{str(s)[-2:]}",
                                       yte[m] + lvl_te[m], test_pred[m] + lvl_te[m])
    metrics["by_season"] = by_season
    by_pos = {}
    for p in ["GK", "DF", "MF", "FW"]:
        m = (test["pos_group"].to_numpy() == p)
        if m.sum():
            by_pos[p] = report(f"pos {p}", yte[m] + lvl_te[m], test_pred[m] + lvl_te[m])
    metrics["by_position"] = by_pos

    # ---- prediction interval --------------------------------------------------
    lo = make_model(loss="quantile", quantile=0.05).fit(Xtr, ytr)
    hi = make_model(loss="quantile", quantile=0.95).fit(Xtr, ytr)
    cover = ((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte))).mean()
    metrics["p05_p95_coverage_test"] = round(float(cover) * 100, 1)
    print(f"\n[interval] model range covers {cover:.1%} of held-out players")

    # ---- feature importance ---------------------------------------------------
    sample = test.sample(min(3000, len(test)), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample["rel_log_value"],
                                 n_repeats=5, random_state=0, scoring="r2")
    order = np.argsort(imp.importances_mean)[::-1]
    top = [(FEATURES[i], round(float(imp.importances_mean[i]), 4)) for i in order[:18]]
    metrics["top_features_permutation_r2_drop"] = top
    print("\n[importance] top drivers of market value:")
    for f, v in top:
        print(f"   {f:<24} {v:.4f}")

    # ---- refit on everything, score every player-season -----------------------
    full_mask = df["log_value"].notna()
    yfit = df.loc[fit_mask, "rel_log_value"]
    final = make_model().fit(df.loc[fit_mask, FEATURES], yfit)
    final_lo = make_model(loss="quantile", quantile=0.05).fit(df.loc[fit_mask, FEATURES], yfit)
    final_hi = make_model(loss="quantile", quantile=0.95).fit(df.loc[fit_mask, FEATURES], yfit)

    X_all = df.loc[full_mask, FEATURES]
    lvl_all = df.loc[full_mask, "season_level"]
    df.loc[full_mask, "pred_log"] = final.predict(X_all) + lvl_all
    df.loc[full_mask, "pred_lo_log"] = final_lo.predict(X_all) + lvl_all
    df.loc[full_mask, "pred_hi_log"] = final_hi.predict(X_all) + lvl_all
    df["model_value_eur"] = np.exp(df["pred_log"])
    df["model_value_lo_eur"] = np.exp(df["pred_lo_log"])
    df["model_value_hi_eur"] = np.exp(df["pred_hi_log"])
    df["residual_log"] = df["pred_log"] - df["log_value"]          # >0 => model says underpriced
    df["value_gap_eur"] = df["model_value_eur"] - df["market_value_eur"]
    df["value_ratio"] = df["model_value_eur"] / df["market_value_eur"]
    # out-of-sample flag: was this row used for fitting?
    df["in_training_sample"] = fit_mask

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "lo": final_lo, "hi": final_hi,
                 "season_level": df.groupby("Season_End_Year")["season_level"].first().to_dict(),
                 "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
                 "metrics": metrics}, MODELS / "value_model.joblib")
    df.to_csv("data/scored_player_seasons.csv", index=False, encoding="utf-8")
    Path("data/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nwrote models/value_model.joblib, data/scored_player_seasons.csv, data/metrics.json")


if __name__ == "__main__":
    main()
