"""
Build the modern (2012/13 - 2025/26) player-valuation dataset.

The FBref mirror used by build_dataset.py was archived in September 2025, so it
stops at 2021/22. This pipeline uses the actively-maintained Transfermarkt
dataset instead (refreshed monthly), which carries every Big-5 appearance and
every dated market valuation through June 2026.

Trade-off: appearance data gives minutes, goals, assists and cards, but no xG,
xAG or progression. Those advanced features carried little weight in the FBref
model (goals+assists per 90 was the only performance feature in the top five),
so the loss is real but small - train_model_modern.py measures it.

Output: data/player_seasons_modern.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("rawdata_tm")
OUT = Path("data")

BIG5 = {"GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
        "L1": "Bundesliga", "FR1": "Ligue 1"}
POS_MAP = {"Goalkeeper": "GK", "Defender": "DF", "Midfield": "MF", "Attack": "FW"}
MIN_SEASON = 2012


def load_games() -> pd.DataFrame:
    g = pd.read_csv(RAW / "games.csv", usecols=[
        "game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
        "home_club_goals", "away_club_goals"])
    return g[g["competition_id"].isin(BIG5) & (g["season"] >= MIN_SEASON)]


def team_points(games: pd.DataFrame) -> pd.DataFrame:
    """Points per match for each club-season, as a club-quality signal."""
    home = games.rename(columns={"home_club_id": "club_id", "home_club_goals": "gf",
                                 "away_club_goals": "ga"})[["club_id", "season", "gf", "ga"]]
    away = games.rename(columns={"away_club_id": "club_id", "away_club_goals": "gf",
                                 "home_club_goals": "ga"})[["club_id", "season", "gf", "ga"]]
    both = pd.concat([home, away], ignore_index=True)
    both["pts"] = np.where(both["gf"] > both["ga"], 3, np.where(both["gf"] == both["ga"], 1, 0))
    out = both.groupby(["club_id", "season"]).agg(
        team_ppm=("pts", "mean"), team_gd90=("gf", "mean")).reset_index()
    conceded = both.groupby(["club_id", "season"])["ga"].mean().reset_index(name="ga_per_game")
    out = out.merge(conceded, on=["club_id", "season"])
    out["team_gd_per_game"] = out["team_gd90"] - out["ga_per_game"]
    return out[["club_id", "season", "team_ppm", "team_gd_per_game"]]


def load_appearances(games: pd.DataFrame) -> pd.DataFrame:
    app = pd.read_csv(RAW / "appearances.csv", usecols=[
        "game_id", "player_id", "player_club_id", "competition_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played"])
    app = app[app["competition_id"].isin(BIG5)]
    app = app.merge(games[["game_id", "season"]], on="game_id", how="inner")

    agg = app.groupby(["player_id", "season"]).agg(
        minutes=("minutes_played", "sum"),
        matches=("game_id", "count"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellows=("yellow_cards", "sum"),
        reds=("red_cards", "sum"),
    ).reset_index()

    # primary club and league of the season = wherever he played the most minutes
    by_club = (app.groupby(["player_id", "season", "player_club_id", "competition_id"])
               ["minutes_played"].sum().reset_index()
               .sort_values("minutes_played", ascending=False))
    primary = by_club.drop_duplicates(["player_id", "season"])
    clubs_per_season = by_club.groupby(["player_id", "season"]).size().rename("n_clubs")

    agg = agg.merge(primary[["player_id", "season", "player_club_id", "competition_id"]],
                    on=["player_id", "season"], how="left")
    return agg.merge(clubs_per_season, on=["player_id", "season"], how="left")


def season_values(seasons: list[int]) -> pd.DataFrame:
    """Each player's market value as of the start of each season (TM's pre-season mark)."""
    v = pd.read_csv(RAW / "player_valuations.csv",
                    usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v.dropna(subset=["date", "market_value_in_eur"]).sort_values("date")

    out = []
    for s in seasons:
        cutoff = pd.Timestamp(f"{s}-09-15")          # just after the summer window
        window = v[(v["date"] <= cutoff) & (v["date"] >= cutoff - pd.Timedelta(days=365))]
        snap = window.drop_duplicates("player_id", keep="last")[
            ["player_id", "market_value_in_eur", "date"]].copy()
        snap["season"] = s
        # how stale the valuation is: a youth-era price on a player who has since
        # broken into the first team makes him look absurdly underpriced
        snap["value_age_days"] = (cutoff - snap["date"]).dt.days
        out.append(snap.drop(columns="date"))
    return pd.concat(out, ignore_index=True).rename(columns={"market_value_in_eur": "market_value_eur"})


def build() -> pd.DataFrame:
    games = load_games()
    seasons = sorted(games["season"].unique())
    print(f"seasons {seasons[0]}-{seasons[-1]}, {len(games)} Big-5 matches")

    app = load_appearances(games)
    vals = season_values(seasons)
    pts = team_points(games)

    players = pd.read_csv(RAW / "players.csv", usecols=[
        "player_id", "name", "date_of_birth", "position", "sub_position", "foot",
        "height_in_cm", "country_of_citizenship"])

    clubs = pd.read_csv(RAW / "clubs.csv", usecols=["club_id", "name"]).rename(
        columns={"name": "club"})

    df = app.merge(vals, on=["player_id", "season"], how="inner")
    df = df.merge(players, on="player_id", how="left")
    df = df.merge(clubs, left_on="player_club_id", right_on="club_id", how="left")
    df = df.merge(pts, left_on=["player_club_id", "season"], right_on=["club_id", "season"],
                  how="left", suffixes=("", "_y"))

    df["league"] = df["competition_id"].map(BIG5)
    df["pos_group"] = df["position"].map(POS_MAP).fillna("MF")
    df["Season_End_Year"] = df["season"] + 1
    df["season_label"] = df["season"].astype(str) + "/" + (df["season"] + 1).astype(str).str[-2:]

    season_start = pd.to_datetime(df["season"].astype(str) + "-09-15")
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = (season_start - dob).dt.days / 365.25
    df["height_m"] = df["height_in_cm"] / 100.0

    mins = df["minutes"].clip(lower=0)
    for new, src in [("goals_p90", "goals"), ("assists_p90", "assists"),
                     ("yellow_p90", "yellows"), ("red_p90", "reds")]:
        df[new] = np.where(mins > 0, df[src] / (mins / 90.0), np.nan)
    df["ga_p90"] = df["goals_p90"] + df["assists_p90"]
    df["min_per_match"] = np.where(df["matches"] > 0, df["minutes"] / df["matches"], np.nan)

    # club quality: leave-one-out mean squad value, as in the FBref pipeline
    grp = df.groupby(["season", "player_club_id"])["market_value_eur"]
    s_sum, s_cnt = grp.transform("sum"), grp.transform("count")
    df["squad_value_mean_loo"] = np.where(s_cnt > 1, (s_sum - df["market_value_eur"]) / (s_cnt - 1), np.nan)
    df["log_squad_value"] = np.log1p(df["squad_value_mean_loo"])

    df["market_value_eur"] = df["market_value_eur"].clip(lower=25_000)
    df["log_value"] = np.log(df["market_value_eur"])
    df["player_name"] = df["name"]
    df["nationality"] = df["country_of_citizenship"]

    df = df.dropna(subset=["age", "league"])
    df = df[(df["age"] >= 15) & (df["age"] <= 45)]

    keep = ["player_id", "player_name", "season", "season_label", "Season_End_Year", "club",
            "value_age_days",
            "league", "pos_group", "sub_position", "nationality", "age", "height_m", "foot",
            "minutes", "matches", "min_per_match", "n_clubs", "goals", "assists",
            "goals_p90", "assists_p90", "ga_p90", "yellow_p90", "red_p90",
            "team_ppm", "team_gd_per_game", "squad_value_mean_loo", "log_squad_value",
            "market_value_eur", "log_value"]
    df = df[keep].copy()

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "player_seasons_modern.csv", index=False, encoding="utf-8")
    print(f"wrote {OUT / 'player_seasons_modern.csv'}  shape={df.shape}")
    print(df.groupby("season_label").agg(n=("player_id", "size"),
                                         median_value=("market_value_eur", "median")).to_string())
    return df


if __name__ == "__main__":
    build()
