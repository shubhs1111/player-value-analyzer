"""
Work out WHY each player's predicted price is what it is.

Method: sampled Shapley values in log-price space. Starting from a wholly typical
player in that position, the features are switched to this player's own values in a
random order, and each switch is credited with the move it causes. Averaging over
several random orders removes the order dependence.

The point of doing it this way rather than one-feature-at-a-time: every permutation
telescopes to the same total, so the contributions **add up exactly** to the distance
between the typical player and this one. An earlier version measured each feature in
isolation, which produced cards where every listed reason was negative while the
prediction sat above the baseline - true of each feature alone, nonsense as a whole.

Because the model works in log space, contributions are reported as percentage
effects that multiply out:

    typical defender EUR8.1m  x  (1 + .34)  x  (1 - .12)  ...  =  his EUR11.7m

Outputs: data/explanations.csv
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS = Path("models/position_models.joblib")
SCORED = Path("data/scored_current.csv")
OUT = Path("data/explanations.csv")

N_PERMUTATIONS = 10
N_REASONS = 3
MIN_EFFECT = 0.08          # ignore anything that moves the price less than 8%

CODE = {
    "age": "age", "minutes_share": "mins", "matches_share": "apps",
    "log_squad_value": "club", "height_m": "hgt", "fotmob_rating": "rtg",
    "fouls_p90": "foul", "yellows_p90": "card", "league": "lg", "foot": "foot",
    "sub_position": "spos",
    "goals_p90": "gls", "assists_p90": "ast", "xg_p90": "xg", "xa_p90": "xa",
    "sot_p90": "sot", "shots_p90": "sh", "big_chances_p90": "bigc",
    "big_chances_missed": "bcm", "chances_p90": "chc", "dribbles_p90": "drb",
    "passes_p90": "pass", "long_balls_p90": "lb",
    "tackles_p90": "tkl", "interceptions_p90": "int", "clearances_p90": "clr",
    "blocks_p90": "blk", "recoveries_p90": "rec", "high_turnovers_p90": "hturn",
    "defensive_actions_p90": "def",
    "saves_p90": "sav", "goals_conceded_p90": "gc", "clean_sheets": "cs",
}
# club quality and age tower over everything else, so they are reported separately
# rather than being allowed to crowd out what the player actually does
CONTEXT_CODES = {"club", "age", "lg", "foot", "hgt", "spos"}

LEAGUE_CODE = {"Premier League": "PL", "La Liga": "LL", "Serie A": "SA",
               "Bundesliga": "BL", "Ligue 1": "L1"}


def player_key(df: pd.DataFrame) -> pd.Series:
    """Unique per player-season-league: a January move between leagues gives two rows."""
    return (df["fotmob_id"].astype(str) + "-" + df["season_start"].astype(str)
            + "-" + df["league"].astype(str).map(LEAGUE_CODE).fillna("XX"))


def shapley_log(model, X: pd.DataFrame, reference: dict, feats: list[str],
                rng: np.random.Generator) -> np.ndarray:
    """Sampled Shapley contributions in log space; columns follow `feats`."""
    phi = np.zeros((len(X), len(feats)))
    for _ in range(N_PERMUTATIONS):
        order = rng.permutation(len(feats))
        cur = X.copy()
        for c in feats:                      # start from the typical player
            cur[c] = reference[c]
        prev = model.predict(cur)
        for j in order:
            cur[feats[j]] = X[feats[j]].to_numpy()
            now = model.predict(cur)
            phi[:, j] += now - prev
            prev = now
    return phi / N_PERMUTATIONS


def main() -> None:
    bundle = joblib.load(MODELS)["bundles"]
    df = pd.read_csv(SCORED, low_memory=False)
    rng = np.random.default_rng(7)

    frames = []
    for pos, b in bundle.items():
        model, feats = b["model"], b["features"]
        sub = df[df["pos_group"] == pos].copy()
        if sub.empty:
            continue
        for c in feats:
            if not pd.api.types.is_numeric_dtype(sub[c]):
                sub[c] = sub[c].astype("category")

        X = sub[feats]
        level = sub["season_level"].to_numpy()

        fit = sub[sub["minutes_share"] >= 0.25]
        reference = {}
        for c in feats:
            col = fit[c] if len(fit) else sub[c]
            if not pd.api.types.is_numeric_dtype(col):
                reference[c] = col.mode().iloc[0] if not col.mode().empty else col.iloc[0]
            else:
                reference[c] = col.median()

        ref_X = X.copy()
        for c in feats:
            ref_X[c] = reference[c]
        base_log = model.predict(ref_X)
        base_eur = np.exp(base_log + level)

        phi = shapley_log(model, X, reference, feats, rng)

        # the check that matters: contributions must close the gap exactly
        closed = np.abs((base_log + phi.sum(1)) - model.predict(X)).max()
        print(f"  {pos}: {len(sub)} player-seasons, {len(feats)} inputs, "
              f"largest failure to add up: {closed:.2e}")

        effect = np.exp(phi) - 1.0            # a multiplier, shown as a percentage
        codes = np.array([CODE.get(c, c) for c in feats])
        is_ctx = np.array([c in CONTEXT_CODES for c in codes])

        out = pd.DataFrame(index=sub.index)
        out["player_key"] = player_key(sub)
        out["base_m"] = (base_eur / 1e6).round(2)
        for group, mask in (("perf", ~is_ctx), ("ctx", is_ctx)):
            gv = np.where(mask[None, :], effect, np.nan)
            order_g = np.argsort(-np.abs(np.nan_to_num(gv)), axis=1)
            for k in range(N_REASONS):
                idx = order_g[:, k]
                take = gv[np.arange(len(gv)), idx]
                keep = np.isfinite(take) & (np.abs(take) >= MIN_EFFECT)
                out[f"{group}{k+1}"] = np.where(keep, codes[idx], "")
                out[f"{group}{k+1}_pct"] = np.where(keep, (take * 100).round(0), np.nan)
        frames.append(out)

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}  shape={res.shape}")

    scored = df.assign(player_key=player_key(df))
    ex = scored[scored["season"] == "2026/27"].merge(res, on="player_key")
    for name in ("Adjei", "Haaland"):
        hit = ex[ex["player"].str.contains(name, na=False)]
        if not len(hit):
            continue
        r = hit.iloc[0]
        print(f"\n{r['player']} ({r['team']}): typical {r['pos_group']} EUR{r['base_m']}m "
              f"-> model EUR{r['model_value_eur']/1e6:.1f}m, market EUR{r['market_value_eur']/1e6:.1f}m")
        for g, lab in (("perf", "on the pitch"), ("ctx", "his situation")):
            bits = [f"{r[f'{g}{k}']} {r[f'{g}{k}_pct']:+.0f}%" for k in (1, 2, 3) if r[f"{g}{k}"]]
            print(f"   {lab:<14}" + ", ".join(bits))


if __name__ == "__main__":
    main()
