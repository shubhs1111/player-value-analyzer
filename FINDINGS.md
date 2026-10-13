# Findings

30,814 player-seasons · 2010/11–2021/22 · Premier League, La Liga, Serie A, Bundesliga, Ligue 1.
Trained on 2010/11–2019/20, **validated on 2020/21 and 2021/22 (3,741 unseen player-seasons).**

## 1. How accurate is it?

| | Baseline (age + minutes + club + league) | Full model |
|---|---|---|
| R² (log value) | 0.777 | **0.823** |
| Median absolute error | 35.2% | **31.6%** |
| Within 30% of the real value | 42.2% | **47.9%** |
| Within 50% | 67.5% | **74.7%** |
| Rank correlation (Spearman) | 0.886 | **0.913** |

Stable across both held-out seasons (R² 0.822 / 0.825) and all four position groups
(GK 0.831, DF 0.815, MF 0.814, FW 0.830) — no hidden position where it falls apart.

The p05–p95 range the analyzer prints covers **78.7%** of held-out players. Calibration
across value deciles is clean: predicted vs actual medians sit within ±10% at every decile,
regression slope 1.03.

**Read the point estimate as "roughly a third off, typically", not as a price.** The range
is the honest output.

## 2. What actually sets a player's market value?

Permutation importance (drop in R² when the feature is shuffled):

| Driver | Importance |
|---|---|
| Club quality (squad value, leave-one-out) | 0.527 |
| **Age** | 0.443 |
| Minutes played | 0.129 |
| Share of starts | 0.028 |
| **Goals + assists per 90** | 0.027 |
| Team points per match | 0.020 |
| League | 0.013 |
| Shots on target per 90 | 0.011 |
| Position | 0.010 |
| Years at club | 0.008 |
| Pass completion % | 0.007 |
| Contract years remaining | 0.003 |

The uncomfortable headline: **who you play for and how old you are matter more than what
you do on the pitch.** Club quality and age together carry ~70% of the explanatory power;
every per-90 performance statistic combined contributes a fraction of that. Playing time
beats production — being trusted to start is worth more than what you do once you start.

Contract length registers weakly here, but that is partly a data artefact: contract expiry
is missing for 37% of rows, and Transfermarkt values are published pre-season, before most
expiries bite.

## 3. Does the mispricing signal mean anything?

A valuation model is only useful if the players it calls cheap go on to *become*
expensive. Grouping 11,164 player-seasons by mispricing quintile and measuring the
**next season's** value change, net of league-wide inflation:

| Model says | n | Median value change vs market | % who gained |
|---|---|---|---|
| Most overvalued | 2,233 | **−15.0%** | 30% |
| Overvalued | 2,233 | −14.3% | 32% |
| Fair | 2,232 | −11.1% | 37% |
| Undervalued | 2,233 | ±0.0% | 43% |
| Most undervalued | 2,233 | **+8.9%** | 55% |

Perfectly monotonic, a **23.9 percentage-point spread** between top and bottom quintile,
Spearman 0.245. And it is not just "young players get dearer" — the ordering survives
inside every age band:

| Age | Most overvalued | Fair | Most undervalued |
|---|---|---|---|
| ≤21 | +12.0% | +42.7% | **+130.8%** |
| 22–24 | 0.0% | +16.7% | **+45.8%** |
| 25–27 | −7.7% | 0.0% | **+22.3%** |
| 28–30 | −20.0% | −14.3% | **0.0%** |
| 31+ | −42.8% | −33.1% | **−18.6%** |

*Caveat:* part of this is mean reversion in Transfermarkt's own numbers — a player priced
temporarily low looks "undervalued" and then drifts back. The effect is real but it is
partly the market correcting itself, not purely foresight.

## 4. Worked examples across the decade

**Bukayo Saka — the model's best call.** 2019/20, age 18, Arsenal: market €20.0m,
model **€28.1m** (1.41x, flagged undervalued). The market agreed the following season —
he was revalued to €65.0m. By 2021/22 the model had caught up and called him fairly
priced (€61.5m vs €65.0m, 0.95x).

**Kalvin Phillips — the model's best warning.** 2021/22, age 26, Leeds: market €50.0m,
model **€15.2m** (0.30x, most overvalued in the league). The profile behind the price was
thin: 1,596 minutes, 0 goals, 0.12 npxG+xAG per 90 (19th percentile among midfielders).
A big-club move followed, and his value fell from there.

**Kevin De Bruyne — the superstar discount.** Model tracks his arc well (Bremen €11.6m at
22 → €131.6m at 29) but sits below market in 9 of 10 seasons. Elite players carry a
premium the model systematically shaves off.

**Eden Hazard — an honest miss.** 2015/16, after a poor title-defence season: market
€65.0m, model **€21.3m** (0.33x, flagged overvalued). The market was right to hold its
nerve — he rebounded to €150m by 2018/19. One bad season is not a trend, and the model,
seeing one season at a time, cannot tell the difference.

**Cristiano Ronaldo, 2021/22 — age does the work.** Age 37 at Juventus: market €30.0m,
model **€13.7m** (0.46x). Scoring output was still strong; the model priced the age curve
and the contract situation instead. He left on a free that winter.

**Moussa Dembélé, 2021/22 — a cheap striker.** Lyon, age 26: market €20.0m, model
**€45.1m** (2.25x) — a 22-goal-involvement season priced like a squad player.

**Erling Haaland and Jude Bellingham, 2021/22 — where the model breaks.** Haaland at 21
(market €150m, model €87.5m, 0.58x) and Bellingham at 19 (market €80m, model €46.1m,
0.58x) are both flagged overvalued. They were not. This is the model's hard limit: the
market was pricing what they would become, and a single season of output cannot show it.
**Treat "overvalued" on an under-22 as a question, not a verdict.**

## 5. Most mispriced players, 2021/22 (≥1,500 minutes)

**Undervalued**

| Player | Club | Pos | Age | Market | Model | Ratio |
|---|---|---|---|---|---|---|
| Alessandro Deiola | Cagliari | MF | 26.9 | €2.0m | €5.9m | 2.95 |
| Lorenzo Montipò | Hellas Verona | GK | 26.4 | €4.5m | €10.9m | 2.43 |
| Jean Onana | Lille | MF | 22.5 | €5.0m | €11.4m | 2.29 |
| Walace | Udinese | MF | 27.2 | €5.0m | €11.4m | 2.27 |
| Moussa Dembélé | Lyon | FW | 26.0 | €20.0m | €45.1m | 2.25 |
| Kiernan Dewsbury-Hall | Leicester | MF | 23.8 | €8.0m | €16.5m | 2.07 |

**Overvalued** (≥€10m only)

| Player | Club | Pos | Age | Market | Model | Ratio |
|---|---|---|---|---|---|---|
| Lucas Boyé | Elche | FW | 26.3 | €15.0m | €4.0m | 0.27 |
| Kalvin Phillips | Leeds | MF | 26.6 | €50.0m | €15.2m | 0.30 |
| Iñigo Martínez | Athletic Bilbao | DF | 31.1 | €25.0m | €9.2m | 0.37 |
| Nahitan Nández | Cagliari | MF | 26.5 | €13.0m | €5.1m | 0.39 |
| James Tarkowski | Burnley | DF | 29.6 | €22.0m | €9.2m | 0.42 |
| Antoine Griezmann | Barcelona | FW | 31.3 | €35.0m | €15.9m | 0.45 |

Two patterns run through the overvalued list: **defenders past 29** (the market pays for
reputation, the model prices the age curve) and **players at mid-table clubs whose price
was set by transfer speculation rather than output**.

---

*Market values are Transfermarkt estimates, not transfer fees. Outcomes described for
Saka, Phillips and Ronaldo after 2021/22 come from general knowledge, not from this
dataset, which ends at that season.*

---

# Part 2 — Extending to 2025/26

The FBref mirror was archived in September 2025. A second pipeline
(`build_dataset_modern.py`) rebuilds the dataset from Transfermarkt's own appearance
records, which are still refreshed monthly, adding **2022/23, 2023/24, 2024/25 and
2025/26** — 34,743 player-seasons from 2012/13 onward. The cost is xG, xAG,
progression and defensive actions, which that source does not carry.

## 1. What the missing stats cost

Trained through 2023/24, tested on 2024/25 and 2025/26 (3,849 unseen player-seasons):

| | FBref era (with xG) | Modern era (no xG) |
|---|---|---|
| R² (log value) | 0.823 | **0.662** |
| Median error | 31.6% | **39.8%** |
| Baseline (age + minutes + club) | 35.2% | 41.1% |
| Lift over baseline | 3.6 points | **1.3 points** |
| Rank correlation | 0.913 | 0.827 |

The obvious question is whether the drop is the thinner features or something harder
about recent seasons. It is the features. Running the **modern feature set on the
FBref model's own validation seasons** (train ≤2019/20, test 2020/21 + 2021/22) gives
R² 0.626 and 43.7% median error — worse than it scores on recent data. Recent seasons
are, if anything, slightly easier to price.

**So xG and progression were doing far more work than their permutation importance
suggested.** In the FBref model, goals+assists per 90 ranked fifth at 0.027 importance,
which reads as negligible. Remove that whole family of features and a quarter of the
model's edge over a naive baseline disappears. Importance scores split credit between
correlated features and understate a group of them measured one at a time.

## 2. What drives value now

| Driver | Importance |
|---|---|
| Club quality | 0.804 |
| Age | 0.188 |
| Minutes played | 0.040 |
| Detailed position | 0.038 |
| Minutes per match | 0.019 |
| League | 0.009 |
| Goals + assists /90 | 0.009 |

Club quality alone now carries four-fifths of the explanatory power — up from about half.
With no way to measure *how well* a player performs, the model falls back almost entirely
on *where* he plays and how old he is. This is the same finding as Part 1, pushed to its
unflattering conclusion.

## 3. The mispricing signal still works

| Model says | n | Median value change vs market | % who gained |
|---|---|---|---|
| Most overvalued | 2,741 | −16.7% | 28% |
| Overvalued | 2,740 | −11.9% | 33% |
| Fair | 2,740 | 0.0% | 42% |
| Undervalued | 2,740 | +7.1% | 53% |
| Most undervalued | 2,741 | **+75.0%** | 78% |

Monotonic again, and again it holds inside every age band. But read the +91.7-point spread
with suspicion rather than satisfaction: **a weaker model produces a wider spread here.**
Its residual is dominated by "this player is priced low relative to his minutes and club",
and cheap players mean-revert upward. The FBref model's tighter 23.9-point spread is the
more trustworthy measure of genuine foresight.

## 4. A data trap worth knowing about

The first run of the 2025/26 rankings returned a Strasbourg midfielder as 178x underpriced
on a €50k valuation against 2,137 minutes. That is not a market inefficiency — it is a
youth-era price that was never updated before the player broke into the first team. Any
valuation snapshot older than 300 days, or below €150k, is now dropped (641 player-seasons).
Without that guard the "most undervalued" list is just a list of stale records.

## 5. Most mispriced, 2025/26 (≥1,500 minutes)

**Underpriced** (≥€1m market value)

| Player | Club | Pos | Age | Market | Model | Ratio |
|---|---|---|---|---|---|---|
| Torbjørn Heggem | Bologna | DF | 26.7 | €1.2m | €10.1m | 8.38 |
| Jacobo Ramón | Como | DF | 20.7 | €1.0m | €7.3m | 7.29 |
| Yan Diomande | RB Leipzig | FW | 18.8 | €1.5m | €10.5m | 6.99 |
| Davide Bartesaghi | AC Milan | DF | 19.7 | €1.8m | €11.5m | 6.38 |
| James Hill | Bournemouth | DF | 23.7 | €3.0m | €13.6m | 4.53 |
| Jon Gorrotxategi | Real Sociedad | MF | 23.6 | €3.0m | €11.1m | 3.71 |

**Overpriced** (≥€20m market value)

| Player | Club | Pos | Age | Market | Model | Ratio |
|---|---|---|---|---|---|---|
| Guillaume Restes | Toulouse | GK | 20.5 | €20.0m | €3.7m | 0.18 |
| Zion Suzuki | Parma | GK | 23.1 | €20.0m | €3.8m | 0.19 |
| Óscar Mingueza | Celta Vigo | DF | 26.3 | €20.0m | €4.0m | 0.20 |
| Ayyoub Bouaddi | Lille | MF | 18.0 | €28.0m | €5.8m | 0.21 |
| Hugo Larsson | Eintracht Frankfurt | MF | 21.2 | €40.0m | €9.2m | 0.23 |
| Pietro Comuzzo | Fiorentina | DF | 20.6 | €25.0m | €6.3m | 0.25 |

Look at the overpriced list: seven of the eight are 23 or younger, most at mid-table clubs.
That is not eight market errors. It is the model's blind spot stated as a ranking — these are
prospects priced on what clubs expect them to become, at clubs whose squad value drags the
model's estimate down. **The current-era "overvalued" list should be read as "young players at
unfashionable clubs", not as a sell list.** The underpriced side is more useful, because
being underpriced relative to minutes and club is a judgement the model can actually make.

## 6. What is not available

**2026/27 is not in the data.** The Transfermarkt mirror was built on 5 September 2026,
before the new season had a usable sample, and it contains no 2026/27 fixtures. Re-running
`build_dataset_modern.py` picks it up once the mirror refreshes — though a model trained on
full seasons should not be trusted on eight matchdays anyway.

FBref itself is unreachable from this machine: it sits behind a Cloudflare bot challenge that
returns 403 to scripted requests and an unsolved interstitial in a real browser. No maintained
mirror of its Big-5 data past 2022/23 exists on GitHub. So closing the xG gap needs either a
network FBref permits, or a paid provider (Opta, StatsBomb, Wyscout).

---

# Part 3 — Position models, and the season in progress

FotMob's public stats host carries per-player tackles, interceptions, clearances,
recoveries, saves and distance covered for the Big 5, back to 2016/17 and **including
the season in progress**. That closed both gaps at once: current data, and the
position-specific statistics a valuation model actually needs.

29,422 player-seasons fetched, 95.2% matched to a Transfermarkt valuation.

## 1. Four models, one per position

| Position | n (test) | Age, minutes, club only | Plus his position's stats | Gain |
|---|---|---|---|---|
| Goalkeepers | 121 | 43.3% | **35.3%** | 8.0 pts |
| Defenders | 610 | 32.3% | **30.4%** | 1.9 pts |
| Midfielders | 504 | 33.8% | **28.6%** | 5.2 pts |
| Forwards | 421 | 33.8% | **28.7%** | 5.1 pts |

Tested on 2025/26, a completed season held out of training. R² between 0.73 and 0.84.

Goalkeepers gain most, which makes sense: without saves and clean sheets a keeper is
almost invisible to a model, since he has no goals, no tackles and no passing volume
to speak of. Defenders gain least — tackles and interceptions are genuinely weak price
signals, because the market pays centre-backs for composure and positioning, which no
counting stat captures.

## 2. The change that mattered most

The first version of this model predicted a player's **pre-season** valuation from the
stats he produced **during** that season. That has the causation backwards: the September
price cannot reflect a season that has not happened.

Switching the target to the **end-of-season** valuation — what his season actually earned
him — moved median error from roughly 40% to 29%, and more than doubled the measured value
of the statistics. Nothing else in the pipeline changed.

It also makes the product coherent: for the season in progress, the model is forecasting
what this form should be worth by May, set against what the player costs today.

## 3. Making a part-played season comparable

2026/27 is eight matches old. Two things make it usable:

- **Playing time as a share of the team's minutes**, not an absolute. A player on 80% of
  his side's minutes reads the same in October as in May.
- **Shrinking every per-90 rate toward the position average**, weighted by minutes
  (K = 450 minutes). A full season is barely touched; eight matches is pulled firmly
  toward the mean.

Before shrinkage the most "underpriced" player in 2026/27 was a 20-year-old with 200
minutes whose model value was ten times his price. After it, the current-season extremes
are defensible and the well-known names land where a fan would expect: Bellingham 1.02x,
Wirtz 0.79x.

## 4. A bug that would have poisoned everything

FotMob publishes some statistics as rates already ("Tackles per 90", "Accurate passes per
90") and others as season totals ("Assists", "Expected goals"). Treating all of them as
totals divides the rates by 90-minute units a second time.

It is invisible in aggregate — the model still trains, the metrics still look plausible —
and it surfaced only on checking one known player: Jude Bellingham at 5.1 accurate passes
per 90, when a Real Madrid midfielder makes about 36. Every passing and defending feature
was deflated by a factor of roughly the player's matches played.

The lesson is cheap to state and easy to skip: after any join or unit conversion, read one
row you know by heart.

## 5. What is on the dashboard now

The scatter plot is gone. In its place:

- **Three steps** in plain language explaining what the model does before any number appears.
- **A player card**: search any player, see his market price and the model's price as two
  bars on one scale with the likely range marked, then the six stats his position's model
  was given — each with his percentile among others in that position.
- **Two lists**, who looks cheap and who looks dear this season, filterable by position and
  league, clicking through to the card.
- **An accuracy chart** showing, per position, how far off the model is with and without
  the stats — so a reader can see for themselves whether knowing the numbers helps.

## 6. Still true, still worth saying

Distance covered is **Premier League only**, and only from 2025/26 — FotMob does not
publish physical data for the other four leagues. It appears on the card for PL players
and is deliberately kept out of the model, where it would act as a league marker rather
than a measure of work rate.

And the model still cannot see potential. A teenager priced at €60m is being paid for what
clubs believe he will become; the model sees only what he has already done.
