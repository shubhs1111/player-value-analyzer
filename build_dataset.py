"""
Build the player-valuation dataset.

Sources (both mirrored in the public `worldfootballR_data` repo, so no scraping of
Cloudflare-protected FBref/Transfermarkt pages is needed):
  * FBref Big-5 player season stats   -> performance features
  * Transfermarkt Big-5 player values -> market value target + age/contract/bio

Output: data/player_seasons.csv  (one row per player-season)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

RAW = Path("rawdata")
OUT = Path("data")
LEAGUES = {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
TM_COMP_FIX = {"LaLiga": "La Liga"}


def read_rds(name: str) -> pd.DataFrame:
    return list(pyreadr.read_r(str(RAW / name)).values())[0]


def slugify(name: str) -> str:
    """Normalised ASCII key from a FBref/Transfermarkt URL slug."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def fbref_keys(df: pd.DataFrame) -> pd.DataFrame:
    url = df["Url"].astype(str)
    df = df.copy()
    df["fbref_id"] = url.str.extract(r"/players/([0-9a-f]+)/")[0]
    df["slug"] = url.str.rsplit("/", n=1).str[-1].map(slugify)
    return df


def per90(num: pd.Series, mins: pd.Series) -> np.ndarray:
    return np.where(mins > 0, num / (mins / 90.0), np.nan)


# --------------------------------------------------------------------------- #
# 1. FBref performance data
# --------------------------------------------------------------------------- #
def load_fbref() -> pd.DataFrame:
    std = fbref_keys(read_rds("fb_big5_advanced_season_stats_big5_player_standard.rds"))
    sht = fbref_keys(read_rds("fb_big5_advanced_season_stats_big5_player_shooting.rds"))
    pas = fbref_keys(read_rds("fb_big5_advanced_season_stats_big5_player_passing.rds"))
    dfn = fbref_keys(read_rds("fb_big5_advanced_season_stats_big5_player_defense.rds"))
    pt = fbref_keys(read_rds("fb_big5_advanced_season_stats_big5_player_playing_time.rds"))

    key = ["Season_End_Year", "Squad", "fbref_id"]

    base = std[key + ["slug", "Comp", "Pos", "Age", "Born", "MP_Playing", "Starts_Playing",
                      "Min_Playing", "Gls", "Ast", "G_minus_PK", "PK", "CrdY", "CrdR",
                      "xG_Expected", "npxG_Expected", "xAG_Expected"]].copy()

    base = base.merge(
        sht[key + ["Sh_Standard", "SoT_Standard", "G_per_SoT_Standard", "Dist_Standard",
                   "np:G_minus_xG_Expected"]].drop_duplicates(key), on=key, how="left")
    base = base.merge(
        pas[key + ["Cmp_percent_Total", "PrgDist_Total", "KP", "Final_Third", "PPA",
                   "Prog", "xA"]].drop_duplicates(key), on=key, how="left")
    base = base.merge(
        dfn[key + ["Tkl_Tackles", "Int", "Tkl+Int", "Blocks_Blocks", "Clr", "Err",
                   "Att 3rd_Tackles"]].drop_duplicates(key), on=key, how="left")
    base = base.merge(
        pt[key + ["Min_percent_Playing.Time", "Compl_Starts", "PPM_Team.Success",
                  "On_minus_Off_Team.Success..xG"]].drop_duplicates(key), on=key, how="left")
    return base


SUM_COLS = ["MP_Playing", "Starts_Playing", "Min_Playing", "Gls", "Ast", "G_minus_PK", "PK",
            "CrdY", "CrdR", "xG_Expected", "npxG_Expected", "xAG_Expected", "Sh_Standard",
            "SoT_Standard", "KP", "Final_Third", "PPA", "Prog", "xA", "Tkl_Tackles", "Int",
            "Tkl+Int", "Blocks_Blocks", "Clr", "Err", "Compl_Starts", "Att 3rd_Tackles"]

WMEAN_COLS = ["Cmp_percent_Total", "Dist_Standard", "G_per_SoT_Standard", "PrgDist_Total",
              "Min_percent_Playing.Time", "PPM_Team.Success",
              "On_minus_Off_Team.Success..xG", "np:G_minus_xG_Expected"]


def aggregate_player_season(df: pd.DataFrame) -> pd.DataFrame:
    """A mid-season transfer gives one FBref row per club; fold them into one row."""
    df = df[df["Comp"].isin(LEAGUES)].copy()
    df["Min_Playing"] = pd.to_numeric(df["Min_Playing"], errors="coerce").fillna(0)
    df = df.sort_values("Min_Playing", ascending=False)

    keys = ["Season_End_Year", "fbref_id"]
    g = df.groupby(keys, sort=False)
    out = g[SUM_COLS].sum(min_count=1)
    out["n_clubs"] = g.size()
    first = g[["slug", "Comp", "Squad", "Pos", "Age", "Born"]].first()
    out = out.join(first)

    for c in WMEAN_COLS:
        w = df[keys + ["Min_Playing", c]].dropna(subset=[c]).copy()
        w["_p"] = w[c] * w["Min_Playing"]
        agg = w.groupby(keys)[["_p", "Min_Playing"]].sum()
        vals = np.where(agg["Min_Playing"] > 0, agg["_p"] / agg["Min_Playing"], np.nan)
        out[c] = pd.Series(vals, index=agg.index)

    return out.reset_index()


# --------------------------------------------------------------------------- #
# 2. Transfermarkt values
# --------------------------------------------------------------------------- #
def load_tm() -> pd.DataFrame:
    v = read_rds("tm_player_vals_big5_player_vals.rds").copy()
    v["comp_name"] = v["comp_name"].replace(TM_COMP_FIX)
    v["Season_End_Year"] = v["season_start_year"].astype(int) + 1
    v["tm_id"] = v["player_url"].astype(str).str.extract(r"/spieler/(\d+)")[0]
    v["slug"] = (v["player_url"].astype(str)
                 .str.extract(r"transfermarkt\.com/([^/]+)/profil")[0].map(slugify))
    v = v.dropna(subset=["player_market_value_euro", "slug"])

    season_end = pd.to_datetime(v["Season_End_Year"].astype(str) + "-06-30")
    expiry = pd.to_datetime(v["contract_expiry"], errors="coerce")
    v["contract_years_left"] = ((expiry - season_end).dt.days / 365.25).clip(-0.5, 6)

    joined = pd.to_datetime(v["date_joined"], errors="coerce")
    v["years_at_club"] = ((season_end - joined).dt.days / 365.25).clip(0, 20)

    dob = pd.to_datetime(v["player_dob"], errors="coerce")
    v["age_exact"] = (season_end - dob).dt.days / 365.25

    v = v.rename(columns={"player_market_value_euro": "market_value_eur",
                          "player_height_mtrs": "height_m",
                          "player_foot": "foot",
                          "player_position": "tm_position",
                          "player_nationality": "nationality",
                          "squad": "tm_squad",
                          "comp_name": "tm_comp"})

    cols = ["Season_End_Year", "slug", "tm_id", "tm_comp", "tm_squad", "tm_position",
            "nationality", "height_m", "foot", "market_value_eur", "contract_years_left",
            "years_at_club", "age_exact", "player_url"]
    v = v[cols].sort_values("market_value_eur", ascending=False)
    # a player who moved mid-season can appear twice; keep the higher valuation
    return v.drop_duplicates(subset=["Season_End_Year", "slug"], keep="first")


# --------------------------------------------------------------------------- #
# 3. Join + feature engineering
# --------------------------------------------------------------------------- #
def pos_group(pos: str) -> str:
    p = str(pos).split(",")[0].strip().upper()
    return p if p in {"GK", "DF", "MF", "FW"} else "MF"


RATE_SRC = {
    "goals_p90": "Gls", "assists_p90": "Ast", "npg_p90": "G_minus_PK",
    "xg_p90": "xG_Expected", "npxg_p90": "npxG_Expected", "xag_p90": "xAG_Expected",
    "shots_p90": "Sh_Standard", "sot_p90": "SoT_Standard", "kp_p90": "KP",
    "prog_p90": "Prog", "ppa_p90": "PPA", "final_third_p90": "Final_Third",
    "tkl_int_p90": "Tkl+Int", "int_p90": "Int", "blocks_p90": "Blocks_Blocks",
    "clr_p90": "Clr", "err_p90": "Err", "high_tkl_p90": "Att 3rd_Tackles",
    "yellow_p90": "CrdY", "red_p90": "CrdR",
}


def build() -> pd.DataFrame:
    fb = aggregate_player_season(load_fbref())
    tm = load_tm()

    dup_fb = fb.duplicated(subset=["Season_End_Year", "slug"], keep=False)
    print(f"dropping {int(dup_fb.sum())} ambiguous FBref name-seasons (shared name slugs)")
    fb = fb[~dup_fb]

    df = fb.merge(tm, on=["Season_End_Year", "slug"], how="inner", validate="one_to_one")
    print(f"FBref player-seasons {len(fb)}, TM rows {len(tm)}, matched {len(df)} "
          f"({len(df) / len(fb):.1%})")

    mins = df["Min_Playing"]
    df["minutes"] = mins
    df["matches"] = df["MP_Playing"]
    df["starts_share"] = np.where(df["MP_Playing"] > 0,
                                  df["Starts_Playing"] / df["MP_Playing"], np.nan)
    df["min_pct_of_team"] = df["Min_percent_Playing.Time"]
    df["age"] = df["age_exact"].fillna(pd.to_numeric(df["Age"], errors="coerce"))
    df["pos_group"] = df["Pos"].map(pos_group)
    df["league"] = df["Comp"]

    for new, src in RATE_SRC.items():
        df[new] = per90(pd.to_numeric(df[src], errors="coerce"), mins)

    df["ga_p90"] = df["goals_p90"] + df["assists_p90"]
    df["npxg_xag_p90"] = df["npxg_p90"] + df["xag_p90"]
    df["finishing_delta"] = df["np:G_minus_xG_Expected"]
    df["pass_cmp_pct"] = df["Cmp_percent_Total"]
    df["prog_dist_p90"] = per90(df["PrgDist_Total"], mins)
    df["team_ppm"] = df["PPM_Team.Success"]
    df["on_off_xg"] = df["On_minus_Off_Team.Success..xG"]

    # Club quality proxy: leave-one-out median market value of the squad that season.
    # Leave-one-out so a player's own value never feeds his own feature.
    grp = df.groupby(["Season_End_Year", "tm_squad"])["market_value_eur"]
    s_sum, s_cnt = grp.transform("sum"), grp.transform("count")
    df["squad_value_mean_loo"] = np.where(
        s_cnt > 1, (s_sum - df["market_value_eur"]) / (s_cnt - 1), np.nan)
    df["log_squad_value"] = np.log1p(df["squad_value_mean_loo"])

    # Drop incomplete seasons: the public mirror can be frozen mid-season, which
    # deflates every minutes/output feature and silently corrupts the model.
    max_mp = df.groupby("Season_End_Year")["MP_Playing"].transform("max")
    partial = max_mp < 30
    if partial.any():
        bad = sorted(df.loc[partial, "Season_End_Year"].unique())
        print(f"dropping incomplete season(s) {bad} "
              f"(max matches played = {df.loc[partial, 'MP_Playing'].max():.0f})")
        df = df[~partial].copy()

    df["market_value_eur"] = df["market_value_eur"].clip(lower=25_000)
    df["log_value"] = np.log(df["market_value_eur"])
    df["player_name"] = df["slug"].str.replace("-", " ").str.title()
    df["season"] = (df["Season_End_Year"] - 1).astype(str) + "/" + \
                   df["Season_End_Year"].astype(str).str[-2:]

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "player_seasons.csv", index=False, encoding="utf-8")
    print(f"wrote {OUT / 'player_seasons.csv'}  shape={df.shape}")
    print(df.groupby("Season_End_Year").size().to_string())
    return df


if __name__ == "__main__":
    build()
