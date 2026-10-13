"""
Join FotMob per-player statistics to Transfermarkt market values.

Covers 2021/22 - 2026/27, including the season in progress.

Two design choices make a part-played season comparable with a finished one:
  * playing time is a SHARE of the team's available minutes, never an absolute
  * every performance stat is a rate per 90 minutes

Output: data/player_seasons_current.csv
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

FM = Path("rawdata_fm/fotmob_player_seasons.csv")
TM = Path("rawdata_tm")
OUT = Path("data")

# absent from a stat list means the player recorded none of it, not unknown
COUNT_STATS = ["goals", "assists", "shots", "shots_on_target", "chances_created",
               "big_chances_created", "big_chances_missed", "dribbles_won", "tackles",
               "interceptions", "clearances", "blocks", "recoveries", "high_turnovers",
               "defensive_actions", "accurate_passes", "accurate_long_balls", "saves",
               "clean_sheets", "goals_conceded", "fouls", "yellows", "reds", "xg", "xa"]
# genuinely missing where absent (FotMob publishes these for some leagues only)
OPTIONAL_STATS = ["fotmob_rating", "distance_m", "sprints", "top_speed_kmh"]

# FotMob publishes some stats as a rate already ("Tackles per 90") and others as a
# season total ("Assists"). Dividing an already-normalised rate by 90-minute units
# again is silently wrong - it quietly deflated every passing and defending number -
# so the two groups are handled separately.
ALREADY_P90 = {
    "shots_on_target": "sot_p90", "shots": "shots_p90", "accurate_passes": "passes_p90",
    "accurate_long_balls": "long_balls_p90", "dribbles_won": "dribbles_p90",
    "defensive_actions": "defensive_actions_p90", "tackles": "tackles_p90",
    "interceptions": "interceptions_p90", "clearances": "clearances_p90",
    "blocks": "blocks_p90", "recoveries": "recoveries_p90",
    "high_turnovers": "high_turnovers_p90", "saves": "saves_p90",
    "goals_conceded": "goals_conceded_p90", "fouls": "fouls_p90",
}
SEASON_TOTALS = {
    "goals": "goals_p90", "assists": "assists_p90", "xg": "xg_p90", "xa": "xa_p90",
    "big_chances_created": "big_chances_p90", "chances_created": "chances_p90",
    "yellows": "yellows_p90", "reds": "reds_p90",
}

POS_MAP = {"Goalkeeper": "GK", "Defender": "DF", "Midfield": "MF", "Attack": "FW"}


def norm_name(x: str) -> str:
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def name_keys(x: str) -> tuple[str, str]:
    """Full normalised name, plus a first-initial+surname key for 'J. Bellingham' forms."""
    full = norm_name(x)
    parts = full.split()
    short = f"{parts[0][0]} {parts[-1]}" if len(parts) > 1 else full
    return full, short


def load_fotmob() -> pd.DataFrame:
    df = pd.read_csv(FM)
    for c in COUNT_STATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in OPTIONAL_STATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["matches"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)
    df = df[df["minutes"] > 0].copy()

    for src, new in ALREADY_P90.items():
        df[new] = pd.to_numeric(df[src], errors="coerce")
    for src, new in SEASON_TOTALS.items():
        df[new] = np.where(df["minutes"] > 0, df[src] / (df["minutes"] / 90.0), np.nan)

    df["ga_p90"] = df["goals_p90"] + df["assists_p90"]
    df["xgxa_p90"] = df["xg_p90"] + df["xa_p90"]
    df["tackles_int_p90"] = df["tackles_p90"] + df["interceptions_p90"]
    df["km_per_90"] = np.where(df["minutes"] > 0, df["distance_m"] / 1000 / (df["minutes"] / 90.0), np.nan)
    df["sprints_p90"] = np.where(df["minutes"] > 0, df["sprints"] / (df["minutes"] / 90.0), np.nan)

    # playing time as a share of what the squad actually played, so a part-season works
    team_max = df.groupby(["season", "league", "team"])["minutes"].transform("max")
    df["minutes_share"] = (df["minutes"] / team_max).clip(0, 1)
    team_games = df.groupby(["season", "league", "team"])["matches"].transform("max")
    df["matches_share"] = (df["matches"] / team_games).clip(0, 1)
    df["team_matches_played"] = team_games
    df["season_complete"] = team_games >= 30

    df["n_full"], df["n_short"] = zip(*df["player"].map(name_keys))
    return df


def tm_season_values(seasons: list[str]) -> pd.DataFrame:
    """Transfermarkt value for each season, plus player bio.

    Two marks per season:
      value_start  - the price before a ball was kicked (what the market thought going in)
      value_end    - the price once the season finished (what his season earned him)

    The model predicts value_end from that season's statistics, which is the honest
    direction of causation: play well, get revalued. For the season in progress there
    is no end mark yet, so the model is forecasting what this form should be worth,
    and value_start is the price to compare it against.
    """
    v = pd.read_csv(TM / "player_valuations.csv",
                    usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v.dropna(subset=["date"]).sort_values("date")

    frames = []
    for s in seasons:
        start_year = int(s.split("/")[0])
        end_year = start_year + 1

        cutoff = pd.Timestamp(f"{start_year}-09-15")
        w = v[(v["date"] <= cutoff) & (v["date"] >= cutoff - pd.Timedelta(days=365))]
        start = w.drop_duplicates("player_id", keep="last").copy()
        start["value_age_days"] = (cutoff - start["date"]).dt.days
        start = start.rename(columns={"market_value_in_eur": "value_start_eur"})
        start = start[["player_id", "value_start_eur", "value_age_days"]]

        # end-of-season mark: the last valuation from the second half of the season
        end_cut = pd.Timestamp(f"{end_year}-07-31")
        w2 = v[(v["date"] <= end_cut) & (v["date"] >= pd.Timestamp(f"{end_year}-01-01"))]
        end = w2.drop_duplicates("player_id", keep="last").copy()
        end = end.rename(columns={"market_value_in_eur": "value_end_eur"})
        end = end[["player_id", "value_end_eur"]]

        snap = start.merge(end, on="player_id", how="outer")
        snap["season"] = s
        frames.append(snap)
    vals = pd.concat(frames, ignore_index=True)
    vals["market_value_eur"] = vals["value_end_eur"].fillna(vals["value_start_eur"])
    vals["value_is_forecast"] = vals["value_end_eur"].isna()
    vals = vals.dropna(subset=["market_value_eur"])

    players = pd.read_csv(TM / "players.csv", usecols=[
        "player_id", "name", "date_of_birth", "position", "sub_position", "foot",
        "height_in_cm", "country_of_citizenship"])
    players["n_full"], players["n_short"] = zip(*players["name"].map(name_keys))
    return vals.merge(players, on="player_id", how="left")


def build() -> pd.DataFrame:
    fm = load_fotmob()
    seasons = sorted(fm["season"].unique())
    tm = tm_season_values(seasons)
    print(f"FotMob {len(fm)} player-seasons, Transfermarkt {len(tm)} valuation snapshots")

    # one valuation per (season, name); drop names that are ambiguous on the TM side
    tm_u = tm.dropna(subset=["n_full"]).sort_values("market_value_eur", ascending=False)
    dup = tm_u.duplicated(["season", "n_full"], keep=False)
    print(f"  {int(dup.sum())} ambiguous Transfermarkt name-seasons set aside")
    tm_full = tm_u[~dup]

    merged = fm.merge(tm_full, on=["season", "n_full"], how="left",
                      suffixes=("", "_tm"), validate="many_to_one")

    # second pass for the unmatched, on first-initial + surname
    miss = merged["player_id"].isna()
    if miss.any():
        tm_s = tm_u.drop_duplicates(["season", "n_short"], keep="first")
        fill = (merged.loc[miss, ["season", "n_short"]]
                .merge(tm_s, on=["season", "n_short"], how="left", suffixes=("", "_s")))
        for col in ["player_id", "market_value_eur", "value_start_eur", "value_end_eur",
                    "value_is_forecast", "value_age_days", "date_of_birth",
                    "position", "sub_position", "foot", "height_in_cm",
                    "country_of_citizenship"]:
            merged.loc[miss, col] = fill[col].to_numpy()

    # Two different players can normalise to the same name - Vitinha of Genoa and
    # Vitinha of PSG - which handed a Genoa squad player a EUR140m valuation. Where
    # several FotMob players in one season land on the same Transfermarkt player,
    # none of them can be trusted, so all are dropped.
    claimed = merged.dropna(subset=["player_id"]).groupby(
        ["season", "player_id"])["fotmob_id"].transform("nunique")
    clash = merged.index.isin(merged.dropna(subset=["player_id"]).index[claimed > 1])
    if clash.any():
        print(f"  {int(clash.sum())} rows dropped: several players share one Transfermarkt record")
        merged.loc[clash, ["market_value_eur", "player_id"]] = np.nan

    matched = merged["market_value_eur"].notna()
    print(f"  matched {int(matched.sum())}/{len(merged)} ({matched.mean():.1%})")
    df = merged[matched].copy()

    start_year = df["season"].str.slice(0, 4).astype(int)
    ref = pd.to_datetime(start_year.astype(str) + "-09-15")
    df["age"] = (ref - pd.to_datetime(df["date_of_birth"], errors="coerce")).dt.days / 365.25
    df["pos_group"] = df["position"].map(POS_MAP).fillna("MF")
    df["height_m"] = df["height_in_cm"] / 100
    df["season_start"] = start_year

    # Early in a season a per-90 rate is built on a few hundred minutes and swings wildly.
    # Shrink every rate toward the average for that position and season, weighted by minutes
    # played: a full season barely moves, eight matches moves a long way. K is in minutes.
    K = 450.0
    rate_cols = [c for c in df.columns if c.endswith("_p90")]
    for c in rate_cols:
        prior = (df[df["minutes"] >= 900]
                 .groupby(["season", "pos_group"])[c].median()
                 .rename("prior").reset_index())
        df = df.merge(prior, on=["season", "pos_group"], how="left")
        df[c + "_raw"] = df[c]
        w = df["minutes"] / (df["minutes"] + K)
        df[c] = w * df[c] + (1 - w) * df["prior"].fillna(df[c])
        df = df.drop(columns="prior")
    print(f"  shrunk {len(rate_cols)} per-90 rates toward position averages (K={K:.0f} minutes)")

    grp = df.groupby(["season", "team"])["market_value_eur"]
    s_sum, s_cnt = grp.transform("sum"), grp.transform("count")
    df["squad_value_mean_loo"] = np.where(s_cnt > 1, (s_sum - df["market_value_eur"]) / (s_cnt - 1), np.nan)
    df["log_squad_value"] = np.log1p(df["squad_value_mean_loo"])

    df["market_value_eur"] = df["market_value_eur"].clip(lower=25_000)
    df["log_value"] = np.log(df["market_value_eur"])
    df = df.dropna(subset=["age"])
    df = df[(df["age"] >= 15) & (df["age"] <= 45)]

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "player_seasons_current.csv", index=False, encoding="utf-8")
    print(f"\nwrote {OUT / 'player_seasons_current.csv'}  shape={df.shape}")
    print(df.groupby("season").agg(players=("player", "size"),
                                   median_value=("market_value_eur", "median"),
                                   median_minutes=("minutes", "median")).to_string())
    return df


if __name__ == "__main__":
    build()
