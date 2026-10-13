"""
Pull per-player season statistics for the Big 5 leagues from FotMob's public
stats host, including the season in progress.

This is the only free source found that carries the stats the model needs by
position - tackles, interceptions, clearances, recoveries, saves and distance
covered - and keeps the current season up to date. FBref's advanced stats are
behind a bot challenge, and the archived worldfootballR mirror stops at 2022/23.

Each (league, season, stat) is one small JSON file keyed by FotMob player id;
they are pivoted into one row per player-season.

Output: rawdata_fm/fotmob_player_seasons.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("rawdata_fm")
CACHE = OUT / "cache"
BASE = "https://data.fotmob.com/stats/{league}/season/{season}/{stat}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}

LEAGUES = {47: "Premier League", 87: "La Liga", 54: "Bundesliga", 55: "Serie A", 53: "Ligue 1"}

# season label -> FotMob tournament id, per league (generated, see rawdata_fm/seasons.json)
SEASONS = {int(k): v for k, v in
           json.loads(Path('rawdata_fm/seasons.json').read_text(encoding='utf-8')).items()}

# FotMob stat key -> our column name
STATS = {
    "mins_played": "minutes",
    "rating": "fotmob_rating",
    "goals": "goals",
    "goal_assist": "assists",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "ontarget_scoring_att": "shots_on_target",
    "total_scoring_att": "shots",
    "big_chance_created": "big_chances_created",
    "big_chance_missed": "big_chances_missed",
    "total_att_assist": "chances_created",
    "accurate_pass": "accurate_passes",
    "accurate_long_balls": "accurate_long_balls",
    "won_contest": "dribbles_won",
    "total_tackle": "tackles",
    "interception": "interceptions",
    "effective_clearance": "clearances",
    "outfielder_block": "blocks",
    "ball_recovery": "recoveries",
    "poss_won_att_3rd": "high_turnovers",
    "defensive_contributions": "defensive_actions",
    "phys_tdc": "distance_m",
    "phys_sprints": "sprints",
    "phys_ts": "top_speed_kmh",
    "saves": "saves",
    "clean_sheet": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "fouls": "fouls",
    "yellow_card": "yellows",
    "red_card": "reds",
}


def fetch(league: int, season_id: int, stat: str, pause: float = 0.25) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{league}_{season_id}_{stat}.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached.unlink()

    url = BASE.format(league=league, season=season_id, stat=stat)
    r = requests.get(url, headers=HEADERS, timeout=40)
    time.sleep(pause)                      # be polite to a free public host
    if r.status_code != 200:
        cached.write_text("[]", encoding="utf-8")
        return []
    payload = r.json()
    rows: list[dict] = []
    for block in payload.get("TopLists", []):
        rows.extend(block.get("StatList", []))
    cached.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frames = []
    for league_id, league_name in LEAGUES.items():
        for season, season_id in SEASONS[league_id].items():
            per_player: dict[int, dict] = {}
            for stat_key, col in STATS.items():
                for row in fetch(league_id, season_id, stat_key):
                    pid = row.get("ParticiantId")
                    if pid is None:
                        continue
                    rec = per_player.setdefault(pid, {
                        "fotmob_id": pid,
                        "player": row.get("ParticipantName"),
                        "team": row.get("TeamName"),
                        "country": row.get("ParticipantCountryCode"),
                        "league": league_name,
                        "season": season,
                    })
                    rec[col] = row.get("StatValue")
                    # minutes and matches come attached to every stat row
                    rec.setdefault("minutes_played", row.get("MinutesPlayed"))
                    rec.setdefault("matches", row.get("MatchesPlayed"))
            if not per_player:
                print(f"  !! no data for {league_name} {season}")
                continue
            df = pd.DataFrame(per_player.values())
            frames.append(df)
            print(f"  {league_name} {season}: {len(df)} players")

    out = pd.concat(frames, ignore_index=True)
    out["minutes"] = out["minutes"].fillna(out["minutes_played"])
    out = out.drop(columns=["minutes_played"])
    out.to_csv(OUT / "fotmob_player_seasons.csv", index=False, encoding="utf-8")
    print(f"\nwrote {OUT / 'fotmob_player_seasons.csv'}  shape={out.shape}")
    print(out.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
