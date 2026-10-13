"""
Player Value Analyzer - command line interface.

  python analyze.py player "Kevin De Bruyne"
  python analyze.py player "Erling Haaland" --season 2022
  python analyze.py rank undervalued --season 2022 --max-age 23 --top 15
  python analyze.py rank overvalued --league "Premier League" --season 2022
  python analyze.py squad "Real Madrid" --season 2022
  python analyze.py compare "Jude Bellingham" "Pedri" "Jamal Musiala" --season 2022

Verdict: the model prices a player from age, minutes, per-90 output, contract
length and club context, then compares that to his actual Transfermarkt value.
A positive gap means the market is paying LESS than the player's profile implies.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ERAS = {
    # current: FotMob per-position stats + Transfermarkt values, 2016/17 - 2026/27 (live season)
    "current": Path("data/scored_current.csv"),
    # modern: Transfermarkt appearances, 2012/13-2025/26, no xG (actively updated)
    "modern": Path("data/scored_player_seasons_modern.csv"),
    # fbref: FBref advanced stats, 2010/11-2021/22, richer features (mirror archived 2025)
    "fbref": Path("data/scored_player_seasons.csv"),
}
SCORED = ERAS["current"]
LOW_MINUTES = 450   # training threshold; below this a prediction is an extrapolation

PROFILE_STATS = [
    ("minutes", "Minutes"), ("min_per_match", "Minutes per match"),
    ("goals_p90", "Goals /90"), ("assists_p90", "Assists /90"), ("ga_p90", "Goals+assists /90"),
    ("npxg_xag_p90", "npxG+xAG /90"), ("sot_p90", "Shots on target /90"),
    ("kp_p90", "Key passes /90"), ("prog_p90", "Progressive actions /90"),
    ("tkl_int_p90", "Tackles+Int /90"), ("pass_cmp_pct", "Pass completion %"),
    ("yellow_p90", "Yellows /90"),
    # position-model stats (current era)
    ("goals_p90", "Goals /90"), ("assists_p90", "Assists /90"), ("xg_p90", "xG /90"),
    ("xa_p90", "xA /90"), ("sot_p90", "Shots on target /90"), ("chances_p90", "Chances created /90"),
    ("passes_p90", "Accurate passes /90"), ("dribbles_p90", "Dribbles won /90"),
    ("tackles_p90", "Tackles /90"), ("interceptions_p90", "Interceptions /90"),
    ("clearances_p90", "Clearances /90"), ("blocks_p90", "Blocks /90"),
    ("recoveries_p90", "Recoveries /90"), ("saves_p90", "Saves /90"),
    ("goals_conceded_p90", "Goals conceded /90"), ("km_per_90", "Kilometres /90"),
]

# the current-era models are per position, so show the stats that position was judged on
POSITION_PROFILE = {
    "GK": [("saves_p90", "Saves /90"), ("goals_conceded_p90", "Goals conceded /90"),
           ("clean_sheets", "Clean sheets"), ("passes_p90", "Accurate passes /90"),
           ("long_balls_p90", "Accurate long balls /90")],
    "DF": [("tackles_p90", "Tackles /90"), ("interceptions_p90", "Interceptions /90"),
           ("clearances_p90", "Clearances /90"), ("blocks_p90", "Blocks /90"),
           ("recoveries_p90", "Recoveries /90"), ("passes_p90", "Accurate passes /90")],
    "MF": [("passes_p90", "Accurate passes /90"), ("chances_p90", "Chances created /90"),
           ("xa_p90", "xA /90"), ("interceptions_p90", "Interceptions /90"),
           ("tackles_p90", "Tackles /90"), ("recoveries_p90", "Recoveries /90"),
           ("km_per_90", "Kilometres /90")],
    "FW": [("goals_p90", "Goals /90"), ("assists_p90", "Assists /90"), ("xg_p90", "xG /90"),
           ("sot_p90", "Shots on target /90"), ("big_chances_p90", "Big chances created /90"),
           ("dribbles_p90", "Dribbles won /90")],
}


try:                                    # names carry accents; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def eur(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if abs(x) >= 1_000_000:
        return f"EUR {x / 1_000_000:,.1f}m"
    return f"EUR {x / 1_000:,.0f}k"


def load(era: str = "modern") -> pd.DataFrame:
    path = ERAS[era]
    if not path.exists():
        raise SystemExit(f"{path} missing - build and train the {era} pipeline first")
    df = pd.read_csv(path, low_memory=False)
    # the two eras keep different column names; normalise to one schema
    renames = {"club": "tm_squad", "team": "tm_squad", "sub_position": "tm_position",
               "player": "player_name"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    if "season_label" in df.columns:
        df["season"] = df["season_label"]        # 2025/26 reads better than 2025
    if "Season_End_Year" not in df.columns and "season_start" in df.columns:
        df["Season_End_Year"] = df["season_start"].astype(int) + 1
    for col in ["contract_years_left", "years_at_club", "nationality", "height_m", "foot",
                "tm_position", "ga_p90"]:
        if col not in df.columns:
            df[col] = np.nan
    df["era"] = era
    df["key"] = df["player_name"].astype(str).str.lower()
    return df


def find_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    q = name.lower().strip()
    hit = df[df["key"] == q]
    if hit.empty:
        hit = df[df["key"].str.contains(q, regex=False, na=False)]
    if hit.empty:
        close = difflib.get_close_matches(q, df["key"].unique().tolist(), n=5, cutoff=0.6)
        if not close:
            raise SystemExit(f"No player matching {name!r}")
        print(f"No exact match for {name!r}. Did you mean: " +
              ", ".join(sorted({c.title() for c in close})))
        hit = df[df["key"] == close[0]]
    names = hit["player_name"].unique()
    if len(names) > 1:
        print(f"{len(names)} players match: {', '.join(names[:8])}  -> using {names[0]!r}")
        hit = hit[hit["player_name"] == names[0]]
    return hit.sort_values("Season_End_Year")


def verdict(ratio: float) -> str:
    if ratio >= 1.5:
        return "UNDERVALUED (market well below profile)"
    if ratio >= 1.15:
        return "slightly undervalued"
    if ratio > 0.87:
        return "fairly priced"
    if ratio > 0.67:
        return "slightly overvalued"
    return "OVERVALUED (market well above profile)"


def percentiles(df: pd.DataFrame, row: pd.Series) -> list[tuple[str, float, float]]:
    """Percentile rank of the player within his position + season + 450-minute peers."""
    peers = df[(df["Season_End_Year"] == row["Season_End_Year"]) &
               (df["pos_group"] == row["pos_group"]) & (df["minutes"] >= 450)]
    era = str(row.get("era", ""))
    stats = ([("minutes", "Minutes")] + POSITION_PROFILE.get(str(row["pos_group"]), [])
             if era == "current" else PROFILE_STATS)
    seen, ordered = set(), []
    for col, label in stats:
        if col not in seen:
            seen.add(col)
            ordered.append((col, label))

    out = []
    for col, label in ordered:
        if col not in df.columns:
            continue
        v = row[col]
        if pd.isna(v) or peers[col].notna().sum() < 20:
            continue
        out.append((label, float(v), float((peers[col] < v).mean() * 100)))
    return out


# --------------------------------------------------------------------------- #
def cmd_player(df: pd.DataFrame, args) -> None:
    hit = find_player(df, args.name)
    p = hit.iloc[-1]
    print(f"\n=== {p['player_name']} ===")
    print(f"{p['tm_position']} | {p['nationality']} | born approx {int(p['Season_End_Year'] - p['age']) if pd.notna(p['age']) else '?'}"
          f" | height {p['height_m']}m | {p['foot']} footed")

    print(f"\n{'Season':<9}{'Club':<22}{'Lg':<16}{'Age':>5}{'Min':>7}{'G+A':>6}"
          f"{'Market':>12}{'Model':>12}{'Model range':>24}{'Gap':>12}")
    for _, r in hit.iterrows():
        rng = f"{eur(r['model_value_lo_eur'])} - {eur(r['model_value_hi_eur'])}"
        print(f"{r['season']:<9}{str(r['tm_squad'])[:21]:<22}{str(r['league'])[:15]:<16}"
              f"{r['age']:>5.1f}{r['minutes']:>7.0f}{r['ga_p90']:>6.2f}"
              f"{eur(r['market_value_eur']):>12}{eur(r['model_value_eur']):>12}{rng:>24}"
              f"{eur(r['value_gap_eur']):>12}")

    sel = hit[hit["Season_End_Year"] == args.season] if args.season else hit.tail(1)
    if sel.empty:
        sel = hit.tail(1)
    r = sel.iloc[-1]
    print(f"\n--- {r['season']} verdict ---")
    print(f"  Market value : {eur(r['market_value_eur'])}")
    print(f"  Model value  : {eur(r['model_value_eur'])}  "
          f"(range {eur(r['model_value_lo_eur'])} - {eur(r['model_value_hi_eur'])})")
    print(f"  Gap          : {eur(r['value_gap_eur'])}  ({r['value_ratio']:.2f}x)")
    print(f"  Verdict      : {verdict(r['value_ratio'])}")
    if r["age"] <= 22.5 and r["value_ratio"] < 0.8:
        print("                 (note: under-22 - the market prices future potential that a "
              "single season of output cannot show)")
    if r["minutes"] < LOW_MINUTES:
        print(f"                 (note: only {r['minutes']:.0f} minutes - low confidence)")
    print(f"  Contract     : {r['contract_years_left']:.1f} yrs left, {r['years_at_club']:.1f} yrs at club"
          if pd.notna(r["contract_years_left"]) else "  Contract     : unknown")

    print(f"\n  Profile vs {r['pos_group']} peers in {r['season']} (percentile):")
    for label, val, pct in percentiles(df, r):
        bar = "#" * int(round(pct / 5))
        print(f"    {label:<22}{val:>9.2f}  {pct:>5.1f}  {bar}")


def filtered(df: pd.DataFrame, args) -> pd.DataFrame:
    d = df[df["minutes"] >= args.min_minutes].copy()
    if args.season:
        d = d[d["Season_End_Year"] == args.season]
    if args.league:
        d = d[d["league"].str.lower() == args.league.lower()]
    if args.pos:
        d = d[d["pos_group"].str.upper() == args.pos.upper()]
    if args.max_age:
        d = d[d["age"] <= args.max_age]
    if args.min_value:
        d = d[d["market_value_eur"] >= args.min_value]
    return d


def print_table(d: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Player':<26}{'Season':<9}{'Club':<20}{'Pos':<5}{'Age':>5}{'Min':>6}"
          f"{'Market':>11}{'Model':>11}{'Ratio':>7}")
    for _, r in d.iterrows():
        print(f"{str(r['player_name'])[:25]:<26}{r['season']:<9}{str(r['tm_squad'])[:19]:<20}"
              f"{r['pos_group']:<5}{r['age']:>5.1f}{r['minutes']:>6.0f}"
              f"{eur(r['market_value_eur']):>11}{eur(r['model_value_eur']):>11}{r['value_ratio']:>7.2f}")


def cmd_rank(df: pd.DataFrame, args) -> None:
    d = filtered(df, args)
    if d.empty:
        raise SystemExit("No players match those filters")
    asc = args.direction == "overvalued"
    d = d.sort_values("residual_log", ascending=asc).head(args.top)
    label = "Most undervalued" if args.direction == "undervalued" else "Most overvalued"
    bits = [f"season {args.season}" if args.season else "all seasons"]
    if args.league:
        bits.append(args.league)
    if args.pos:
        bits.append(args.pos)
    if args.max_age:
        bits.append(f"age<={args.max_age}")
    print_table(d, f"{label} ({', '.join(bits)}, min {args.min_minutes} mins)")


def cmd_squad(df: pd.DataFrame, args) -> None:
    d = df[df["tm_squad"].str.lower().str.contains(args.name.lower(), na=False)]
    d = d[d["minutes"] >= args.min_minutes]
    if args.season:
        d = d[d["Season_End_Year"] == args.season]
    else:
        d = d[d["Season_End_Year"] == d["Season_End_Year"].max()]
    if d.empty:
        raise SystemExit(f"No squad matching {args.name!r}")
    d = d.sort_values("market_value_eur", ascending=False)
    print_table(d, f"{d['tm_squad'].iloc[0]} - {d['season'].iloc[0]}")
    print(f"\n  Squad market value : {eur(d['market_value_eur'].sum())}")
    print(f"  Squad model value  : {eur(d['model_value_eur'].sum())}")
    print(f"  Aggregate gap      : {eur(d['value_gap_eur'].sum())}")


def cmd_compare(df: pd.DataFrame, args) -> None:
    rows = []
    for n in args.names:
        h = find_player(df, n)
        if args.season:
            h = h[h["Season_End_Year"] == args.season]
        if h.empty:
            print(f"  (no season data for {n})")
            continue
        rows.append(h.iloc[-1])
    if not rows:
        raise SystemExit("nothing to compare")
    print_table(pd.DataFrame(rows), "Comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description="Football player market-value analyzer (Big 5 leagues)")
    ap.add_argument("--era", choices=sorted(ERAS), default="current",
                    help="current = 2016/17-2026/27 position models (default); "
                         "modern = 2012/13-2025/26; fbref = 2010/11-2021/22")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--season", type=int, help="season end year, e.g. 2022 for 2021/22")
        p.add_argument("--league", help="Premier League | La Liga | Serie A | Bundesliga | Ligue 1")
        p.add_argument("--pos", help="GK | DF | MF | FW")
        p.add_argument("--max-age", type=float)
        p.add_argument("--min-minutes", type=int, default=900)
        p.add_argument("--min-value", type=float, default=0)
        p.add_argument("--top", type=int, default=20)

    p1 = sub.add_parser("player", help="value one player")
    p1.add_argument("name")
    p1.add_argument("--season", type=int)

    p2 = sub.add_parser("rank", help="rank mispriced players")
    p2.add_argument("direction", choices=["undervalued", "overvalued"])
    common(p2)

    p3 = sub.add_parser("squad", help="value a whole squad")
    p3.add_argument("name")
    p3.add_argument("--season", type=int)
    p3.add_argument("--min-minutes", type=int, default=450)

    p4 = sub.add_parser("compare", help="compare several players")
    p4.add_argument("names", nargs="+")
    p4.add_argument("--season", type=int)

    args = ap.parse_args()
    df = load(args.era)
    {"player": cmd_player, "rank": cmd_rank, "squad": cmd_squad, "compare": cmd_compare}[args.cmd](df, args)


if __name__ == "__main__":
    main()
