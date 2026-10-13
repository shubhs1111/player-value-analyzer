"""
One model per position, each using the statistics that matter for that role.

A centre-back is not priced on goals and a striker is not priced on clearances,
so a single model across all positions has to learn those distinctions from
scratch and dilutes every feature. Four smaller models, each given the right
stats, are easier to train and far easier to explain - which is the point: the
dashboard shows the same stats the model was given.

  Goalkeepers  saves, goals conceded, clean sheets, distribution
  Defenders    tackles, interceptions, clearances, blocks, recoveries
  Midfielders  passes, chances created, xA, tackles, interceptions, recoveries
  Forwards     goals, assists, xG, shots on target, big chances, dribbles

Split: train on 2021/22-2024/25, test on 2025/26 (complete and unseen).
2026/27 is then scored for current values - it cannot be validated, because
the season has not finished.

Outputs: models/position_models.joblib, data/scored_current.csv, data/metrics_positions.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path("data/player_seasons_current.csv")
MODELS = Path("models")

SHARED = ["age", "minutes_share", "matches_share", "log_squad_value", "height_m",
          "fotmob_rating", "fouls_p90", "yellows_p90"]

POSITION_FEATURES = {
    "GK": ["saves_p90", "goals_conceded_p90", "clean_sheets", "passes_p90", "long_balls_p90"],
    "DF": ["tackles_p90", "interceptions_p90", "clearances_p90", "blocks_p90",
           "recoveries_p90", "passes_p90", "long_balls_p90", "goals_p90", "assists_p90",
           "dribbles_p90"],
    "MF": ["passes_p90", "chances_p90", "xa_p90", "big_chances_p90", "dribbles_p90",
           "tackles_p90", "interceptions_p90", "recoveries_p90", "high_turnovers_p90",
           "goals_p90", "assists_p90", "xg_p90"],
    "FW": ["goals_p90", "assists_p90", "xg_p90", "xa_p90", "shots_p90", "sot_p90",
           "big_chances_p90", "big_chances_missed", "dribbles_p90", "chances_p90"],
}
CATEGORICAL = ["league", "foot", "sub_position"]

# what the dashboard shows on a player card, per position: (column, label, higher_is_better)
CARD_STATS = {
    "GK": [("saves_p90", "Saves per 90", True), ("goals_conceded_p90", "Goals conceded per 90", False),
           ("clean_sheets", "Clean sheets", True), ("passes_p90", "Accurate passes per 90", True),
           ("long_balls_p90", "Accurate long balls per 90", True), ("minutes_share", "Share of minutes", True)],
    "DF": [("tackles_p90", "Tackles per 90", True), ("interceptions_p90", "Interceptions per 90", True),
           ("clearances_p90", "Clearances per 90", True), ("blocks_p90", "Blocks per 90", True),
           ("recoveries_p90", "Ball recoveries per 90", True), ("passes_p90", "Accurate passes per 90", True)],
    "MF": [("passes_p90", "Accurate passes per 90", True), ("chances_p90", "Chances created per 90", True),
           ("xa_p90", "Expected assists per 90", True), ("interceptions_p90", "Interceptions per 90", True),
           ("tackles_p90", "Tackles per 90", True), ("recoveries_p90", "Ball recoveries per 90", True)],
    "FW": [("goals_p90", "Goals per 90", True), ("assists_p90", "Assists per 90", True),
           ("xg_p90", "Expected goals per 90", True), ("sot_p90", "Shots on target per 90", True),
           ("big_chances_p90", "Big chances created per 90", True), ("dribbles_p90", "Dribbles won per 90", True)],
}

TRAIN_MAX = 2024          # season start year -> trains through 2024/25
TEST_SEASON = 2025        # 2025/26
CURRENT_SEASON = 2026     # 2026/27, scored but not validated
MIN_MINUTES_SHARE = 0.25  # a quarter of the team's minutes
MAX_VALUE_AGE_DAYS = 300
MIN_VALUE_EUR = 150_000


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    reg = df[df["minutes_share"] >= MIN_MINUTES_SHARE]
    level = reg.groupby("season_start")["log_value"].median()
    df["season_level"] = df["season_start"].map(level)
    df["rel_log_value"] = df["log_value"] - df["season_level"]
    club_level = reg.groupby("season_start")["log_squad_value"].median()
    df["log_squad_value"] = df["log_squad_value"] - df["season_start"].map(club_level)
    return df


def make_model(feats: list[str], loss: str = "squared_error", quantile: float | None = None):
    return HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, max_iter=500, learning_rate=0.06,
        max_leaf_nodes=24, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=35,
        categorical_features=[feats.index(c) for c in CATEGORICAL if c in feats],
        random_state=42,
    )


def report(name: str, y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y_eur, p_eur = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(p_eur - y_eur) / y_eur
    m = {"n": int(len(y_log)),
         "r2_log": round(float(r2_score(y_log, pred_log)), 4),
         "mae_log": round(float(mean_absolute_error(y_log, pred_log)), 4),
         "median_abs_pct_err": round(float(np.median(ape)) * 100, 1),
         "within_30pct": round(float((ape <= 0.30).mean()) * 100, 1),
         "within_50pct": round(float((ape <= 0.50).mean()) * 100, 1),
         "spearman": round(float(spearmanr(y_eur, p_eur).statistic), 4)}
    print(f"[{name:<22}] " + "  ".join(f"{k}={v}" for k, v in m.items()))
    return m


def main() -> None:
    df = load()
    fresh = (df["value_age_days"] <= MAX_VALUE_AGE_DAYS) & (df["market_value_eur"] >= MIN_VALUE_EUR)
    usable = (df["minutes_share"] >= MIN_MINUTES_SHARE) & fresh & df["log_squad_value"].notna()
    print(f"{int((~fresh).sum())} rows dropped on stale/implausible valuations\n")

    metrics: dict = {"positions": {}}
    bundles: dict = {}
    scored, holdout = [], []

    for pos, extra in POSITION_FEATURES.items():
        feats = SHARED + extra + CATEGORICAL
        sub = df[(df["pos_group"] == pos) & usable]
        train = sub[sub["season_start"] <= TRAIN_MAX]
        test = sub[sub["season_start"] == TEST_SEASON]
        if len(train) < 200 or len(test) < 50:
            print(f"skipping {pos}: too few rows")
            continue

        Xtr, ytr = train[feats], train["rel_log_value"].to_numpy()
        Xte, yte = test[feats], test["rel_log_value"].to_numpy()
        lvl = test["season_level"].to_numpy()

        # baseline: age, playing time and club only - no performance stats at all
        base_feats = ["age", "minutes_share", "log_squad_value", "league"]
        base = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=42,
            categorical_features=[base_feats.index("league")]).fit(train[base_feats], ytr)

        model = make_model(feats).fit(Xtr, ytr)
        test_pred_pos = model.predict(Xte)
        print(f"\n--- {pos} (train {len(train)}, test {len(test)}) ---")
        b = report(f"{pos} no stats", yte + lvl, base.predict(test[base_feats]) + lvl)
        f = report(f"{pos} with stats", yte + lvl, model.predict(Xte) + lvl)

        # keep the held-out predictions themselves: the dashboard shows real examples
        # from a season the model never saw, which is worth more than a summary number
        holdout.append(pd.DataFrame({
            "player": test["player"].to_numpy(),
            "team": test["team"].to_numpy(),
            "league": test["league"].to_numpy(),
            "pos": pos,
            "age": test["age"].round(1).to_numpy(),
            "season": test["season"].to_numpy(),
            "market_m": (np.exp(yte + lvl) / 1e6).round(1),
            "model_m": (np.exp(test_pred_pos + lvl) / 1e6).round(1),
            # the same players priced without any of their statistics, for comparison
            "basic_m": (np.exp(base.predict(test[base_feats]) + lvl) / 1e6).round(1),
        }))

        lo = make_model(feats, loss="quantile", quantile=0.05).fit(Xtr, ytr)
        hi = make_model(feats, loss="quantile", quantile=0.95).fit(Xtr, ytr)
        cover = float(((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte))).mean())

        metrics["positions"][pos] = {
            "baseline": b, "with_stats": f,
            "improvement_pct_points": round(b["median_abs_pct_err"] - f["median_abs_pct_err"], 1),
            "range_coverage": round(cover * 100, 1),
            "features": extra,
        }

        # refit on everything usable, then score every row of this position
        fit = sub
        final = make_model(feats).fit(fit[feats], fit["rel_log_value"])
        final_lo = make_model(feats, loss="quantile", quantile=0.05).fit(fit[feats], fit["rel_log_value"])
        final_hi = make_model(feats, loss="quantile", quantile=0.95).fit(fit[feats], fit["rel_log_value"])
        bundles[pos] = {"model": final, "lo": final_lo, "hi": final_hi, "features": feats}

        allpos = df[(df["pos_group"] == pos) & df["log_squad_value"].notna() & fresh].copy()
        lvl_all = allpos["season_level"].to_numpy()
        allpos["model_value_eur"] = np.exp(final.predict(allpos[feats]) + lvl_all)
        allpos["model_value_lo_eur"] = np.exp(final_lo.predict(allpos[feats]) + lvl_all)
        allpos["model_value_hi_eur"] = np.exp(final_hi.predict(allpos[feats]) + lvl_all)
        scored.append(allpos)

    out = pd.concat(scored, ignore_index=True)
    out["value_gap_eur"] = out["model_value_eur"] - out["market_value_eur"]
    out["value_ratio"] = out["model_value_eur"] / out["market_value_eur"]
    out["residual_log"] = np.log(out["model_value_eur"]) - out["log_value"]
    out["enough_minutes"] = out["minutes_share"] >= MIN_MINUTES_SHARE

    # percentile of each card stat within position + season (regulars only)
    for pos, stats in CARD_STATS.items():
        m = out["pos_group"] == pos
        for col, _, higher in stats:
            if col not in out.columns:
                continue
            ranks = (out[m & out["enough_minutes"]]
                     .groupby("season_start")[col].rank(pct=True) * 100)
            out.loc[ranks.index, col + "_pct"] = ranks if higher else 100 - ranks

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"bundles": bundles, "shared": SHARED, "card_stats": CARD_STATS,
                 "metrics": metrics}, MODELS / "position_models.joblib")
    out.to_csv("data/scored_current.csv", index=False, encoding="utf-8")
    pd.concat(holdout, ignore_index=True).to_csv("data/holdout_predictions.csv",
                                                 index=False, encoding="utf-8")
    Path("data/metrics_positions.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nwrote data/scored_current.csv  shape={out.shape}")
    print(out.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
