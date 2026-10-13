# Player Value Analyzer — Europe's Big 5 leagues

Sets a model-implied market value for every player in the Premier League, La Liga,
Serie A, the Bundesliga and Ligue 1, from **performance stats, age, playing time,
contract length and club context**, then compares that to the player's actual
Transfermarkt valuation to flag who is mispriced.

Three pipelines, because no single public source covers the whole period. The
**current** one is the default and the one to use:

| Era | Seasons | Source | Model | Accuracy |
|---|---|---|---|---|
| **`current`** (default) | 2016/17 – **2026/27 (live)** | FotMob stats + Transfermarkt values | one model per position | **28–35%** median error |
| `modern` | 2012/13 – 2025/26 | Transfermarkt appearances only | single model | 39.8% |
| `fbref` | 2010/11 – 2021/22 | FBref advanced stats (archived) | single model | 31.6% |

```bash
python analyze.py player "Florian Wirtz"                        # current season
python analyze.py rank undervalued --season 2027 --top 15       # 2026/27
python analyze.py --era fbref player "Kevin De Bruyne"          # older seasons, xG features
```


---

## Where the data comes from

The screenshot suggested `soccerdata` (Python) / `worldfootballR` (R). Those libraries
scrape FBref and Transfermarkt live — and **both sites block this environment at the
edge (HTTP 403, Cloudflare)**, so a live scrape is not possible from here.

Instead the pipeline pulls from the `worldfootballR_data` mirror on GitHub, which is
the same data those libraries fetch, pre-harvested and committed as `.rds` files:

| Source file | What it gives |
|---|---|
| `big5_player_standard / shooting / passing / defense / playing_time.rds` | FBref per-season player stats (goals, xG, xAG, progression, tackles, minutes, team-success-on-pitch) |
| `big5_player_vals.rds` | Transfermarkt market value, DOB, height, foot, contract expiry, date joined |

Joining FBref to Transfermarkt is normally painful because player names disagree on
accents — and in these files the accented characters are already corrupted. The fix:
**both sources carry a clean ASCII name slug inside their own URLs**
(`fbref.com/en/players/<id>/Kevin-De-Bruyne`, `transfermarkt.com/kevin-de-bruyne/profil/...`),
so the join runs on slugs, not display names. Match rate: **85.4%** of FBref
player-seasons. Ambiguous slugs (two players sharing a name in one season) are dropped.

### The 2022/23 – 2025/26 extension

The FBref mirror (`JaseZiv/worldfootballR_data`) was **archived in September 2025**, which is
why it stops at 2021/22. `build_dataset_modern.py` closes the gap from a different source —
the `dcaribou/transfermarkt-datasets` project, which publishes the full Transfermarkt database
to a public Cloudflare R2 bucket and refreshes it monthly (last build 5 Sep 2026):

```
https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{players,player_valuations,appearances,games,clubs}.csv.gz
```

That gives **four more complete seasons: 2022/23, 2023/24, 2024/25 and 2025/26**, with market
valuations dated through June 2026. It does not give xG — see FINDINGS.md for what that costs.

Two data-quality guards in that pipeline are worth keeping:

- **Valuation freshness.** A player's "pre-season value" is the last valuation on or before
  15 September. For a youth player who broke into the first team, that can be a youth-era
  price, which makes him look 170x underpriced. Snapshots older than 300 days, and values
  under €150k, are dropped (641 player-seasons).
- **No contract feature.** `players.csv` carries only the player's *current* contract expiry,
  so applying it to a 2015 season would be anachronistic. The modern model omits it.

### If you want live/current data
`pip install soccerdata` and run it from a network that FBref allows; `build_dataset.py`
reads from `rawdata/` and nothing downstream cares where the files came from. One caveat
worth knowing: the GitHub mirror's last season, 2022/23, is **frozen mid-season**
(max 13 matches played), which silently deflates every minutes-based feature — the
builder detects and drops any season with `max matches < 30`. Keep that guard.

---

## Install & run

```bash
pip install pandas numpy scikit-learn scipy pyreadr joblib
```

```bash
python build_dataset.py     # rawdata/*.rds  ->  data/player_seasons.csv
python train_model.py       # trains, validates, scores every player-season
python backtest.py          # does the mispricing signal predict future value?
```

### Using the analyzer

```bash
python analyze.py player "Kevin De Bruyne"
python analyze.py player "Bukayo Saka" --season 2020
python analyze.py rank undervalued --season 2022 --max-age 23 --top 15
python analyze.py rank overvalued --league "Premier League" --season 2022 --min-value 10000000
python analyze.py squad "Real Madrid" --season 2022
python analyze.py compare "Jude Bellingham" "Pedri" "Jamal Musiala" --season 2022
```

`--season` takes the **season end year** (2022 = the 2021/22 season). Other filters:
`--league --pos GK|DF|MF|FW --max-age --min-minutes --min-value --top`.

---

## How the valuation works

**Target:** `log(market value in EUR)`.

**Model:** gradient-boosted trees (`HistGradientBoostingRegressor`). Chosen because it
handles missing values natively — FBref's advanced passing/defending stats only start in
2017/18, so pre-2018 seasons have real gaps that would otherwise force either imputation
or throwing away half the history.

**36 features** across six groups:

- *Age & workload* — age, minutes, matches, share of starts, share of team minutes
- *Attacking output* — goals, assists, npxG, xAG, shots, shots on target, finishing vs xG (all per 90)
- *Creation & progression* — key passes, progressive actions, passes into the box, final-third entries, pass completion %
- *Defending* — tackles+interceptions, blocks, clearances, errors, high turnovers
- *Discipline* — yellows, reds per 90
- *Contract & context* — years left on contract, years at the club, height, foot, league, position, team points-per-match, on/off xG impact, and club quality

**Two things the model deliberately corrects for:**

1. **Market inflation.** Big-5 median value rose from €3.0m (2011) to €7.5m (2022). The
   model therefore predicts value *relative to its season's market level* and re-inflates
   afterwards, so a 2013 player and a 2022 player are judged on the same scale. Without
   this the model's error on held-out recent seasons nearly doubles.
2. **Club quality leakage.** Club strength is the single strongest predictor, measured as
   the squad's mean market value — computed **leave-one-out**, so a player's own value
   never feeds his own feature.

**Validation is strictly temporal:** train on 2010/11–2019/20, test on 2020/21 and
2021/22 — seasons the model has never seen. See `FINDINGS.md` for the numbers.

---

## The current pipeline

```bash
python fetch_fotmob.py            # per-player stats, 5 leagues x 11 seasons (cached)
python build_dataset_current.py   # join to Transfermarkt values
python train_model_positions.py   # four models, one per position
python export_dashboard_v2.py     # files for the dashboard
```

**A model per position.** A centre-back is not priced on goals and a striker is not
priced on clearances, so each position gets its own model and its own statistics:

| Position | Judged on |
|---|---|
| Goalkeepers | saves, goals conceded, clean sheets, distribution |
| Defenders | tackles, interceptions, clearances, blocks, recoveries, passing |
| Midfielders | passes, chances created, xA, tackles, interceptions, recoveries |
| Forwards | goals, assists, xG, shots on target, big chances, dribbles |

plus age, share of team minutes, and squad quality for everyone.

**Three decisions that made the difference:**

1. **The target is the end-of-season valuation, not the pre-season one.** Predicting a
   September price from the stats a player went on to produce has the causation backwards.
   Predicting what his season earned him is the honest direction — and it cut the median
   error from ~40% to ~29%.
2. **Playing time is a share of the team's minutes, never an absolute.** That is what lets
   a season eight matches old sit in the same model as a finished one.
3. **Per-90 rates are shrunk toward the position average**, weighted by minutes played
   (K = 450). A full season barely moves; eight matches moves a long way. Without it the
   early-season leaderboards are just small-sample noise.

One trap worth knowing: FotMob serves some stats as rates already ("Tackles per 90") and
others as season totals ("Assists"). Dividing an already-normalised rate by 90-minute units
again deflates every passing and defending number and is invisible unless you sanity-check
against a known player. `ALREADY_P90` and `SEASON_TOTALS` in `build_dataset_current.py`
keep the two apart.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Download → clean → join → feature engineering |
| `train_model.py` | Train, validate, score every player-season, save model |
| `backtest.py` | Tests whether "undervalued" flags predict future value growth |
| `analyze.py` | The CLI analyzer |
| `data/player_seasons.csv` | 30,814 player-seasons, 95 columns |
| `data/scored_player_seasons.csv` | Same + model value, range, gap, ratio |
| `data/metrics.json` | All validation metrics |
| `models/value_model.joblib` | Fitted model + p05/p95 range models |
| `FINDINGS.md` | Accuracy, what drives value, worked examples |

---

## Known limitations

- **The newest season available is 2025/26.** The Transfermarkt mirror was built on
  5 Sep 2026, before the 2026/27 season had any meaningful sample, so 2026/27 is not in it.
  Re-running `build_dataset_modern.py` picks up new seasons as the mirror refreshes.
- **The current-era model is much weaker.** Without xG it beats an age+minutes+club baseline
  by only 1.3 points of median error. Its rankings are directional, not precise.
- **The model shades the expensive end down.** Median model/market ratio is 1.11 for
  players under €5m and 0.83–0.89 above €50m — standard regression-to-the-mean. Compare
  players *within* a value band rather than across the whole league.
- **It cannot see potential, injuries, or transfer gossip.** A 19-year-old's €80m price
  tag reflects what clubs think he will become; the model only sees what he has done.
  That is a real cause of "overvalued" flags on teenagers (Haaland 2021/22, Pedri 2021/22).
- **Market value is itself an estimate**, crowd-sourced by Transfermarkt — the model is
  learning to predict a well-informed opinion, not a realised transfer fee.
- Some club names from the source mirror have lost their accented characters
  (`1. FC K?ln`). Player names are reconstructed from URL slugs and are clean.

---

## Dashboard

`export_dashboard_data.py` writes flat files to `dashboard_data/` (player-season
valuations, accuracy headlines, feature importance, backtest results) which feed the
published dashboard at https://claude.ai/artifact/2ubbePstwQ35xEMD5pqjpg —
market-vs-model scatter, the mispriced tables, what drives value, and the backtest,
filterable by season, league and position.

```bash
python export_dashboard_data.py
```

Re-run it after rebuilding the model, then ask Claude to refresh the dashboard's files.
