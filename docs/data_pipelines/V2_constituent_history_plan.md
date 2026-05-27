# data_pipelines — V2 plan: NSE index constituent history (`universe_members(as_of=)`)

Branch: `data-constituent-history-plan` · Status: **design only — no implementation in this PR.**

This is the implementation specification for the next coherent slice of
`data_pipelines` work after v1.7: persisting NSE index *membership over time*
in the cache, plus the loader API that lets downstream modules (notably
`backtesting/`) ask "who was in NIFTY 100 as of 2020-03-12?" and get an
honest answer.

Out-of-scope follow-ups parked during this branch will land in
`docs/data_pipelines/V2_TBD.md` as a new sub-section ("V2 constituent
history follow-ups") rather than this doc — keep this spec tight.

Related context already in the repo (cited rather than duplicated):

- `docs/data_pipelines/goal.md` § "Honest about failure", "Cache-first,
  two-layer" — the non-negotiables this plan must respect.
- `docs/data_pipelines/V2_TBD.md` — current TBD list. There is **no
  existing entry** for constituent history; the parking-lot reference in
  the universe YAMLs ("Point-in-time historical membership is explicitly
  out of scope — V2_TBD open q.1") refers to a placeholder that was
  rolled into this plan instead.
- `docs/data_pipelines/universe_yaml_spec.md` § 1 — explicit "current-snapshot
  contract" framing the YAML answers, which this plan is the long-term
  successor to *for the historical-query path* (the YAMLs stay around as
  the cold-start seed + the lint-checked source of truth for the *current*
  list).
- `docs/data_pipelines/adding_a_domain.md` § "Seams the framework
  deliberately leaves to domains" — "Universe maintenance: domains that
  need point-in-time / drifting universes will likely want their own
  scraper or vendor feed — that's a per-domain concern." This plan is the
  concrete realization of that note for `nse_equities`.
- `docs/backtesting/goal.md` § "Look-ahead-bias elimination is structural,
  not conventional." — the customer for this work.
- `.claude/memories/project-nse-data-quirks.md` — `archives.nseindia.com`
  is the reliable path for current-list CSVs; jugaad/nselib are blocked
  much of the time. DUMMY tickers must be filtered. These constraints
  carry forward to the refresh job.

---

## § 1. Problem statement

### 1.1 The survivorship-bias gap (concrete)

Every NSE universe YAML under
`configs/data_pipelines/domains/nse_equities/universe_*.yaml` carries
exactly one constituent snapshot, pinned by a single date:

```yaml
universe: nifty50
listed_at: 2026-05-24
indices:
  - "NIFTY:50"
tickers:
  - "NSE:ADANIENT"
  - ...
  - "NSE:JIOFIN"     # IPO 2023-08
  - "NSE:ETERNAL"    # listed under new name 2024
  - ...
```

When `gbdt` (or anything else) loads this universe for a 2018–2026 panel,
it gets the **2026** constituent set applied to every date in the window:

- **False positives** — JIOFIN (Jio Financial Services, listed
  2023-08-21) appears in the universe for dates back to 2018 even though
  the company didn't trade publicly. The `min_rows` gate on the
  cache-side load (`gbdt.data.load_panel`, see
  `project-nse-data-quirks.md` § 5) silently drops it from training data
  for pre-IPO dates — but the *universe membership* is wrong, and the
  per-day cross-sectional features (rank/zscore across the panel) are
  computed against an artificially-shrunk panel that doesn't match what a
  causal observer in 2018 would have seen.
- **False negatives (the harder bias)** — any company that *was* in
  NIFTY 50 between 2018 and 2026 but has since been removed
  (e.g., delistings, mergers, replaced-by-rebalance) is **missing
  entirely** from the universe. The strategy never sees them, so any
  metric computed on the historical panel is conditioned on
  "stocks that survived to 2026". Classic survivorship bias.

The same problem exists for every NSE universe shipped by v1.7 / the
Sectoral catalog (PR #24) / the Thematic catalog (PR #40) — 30+ files at
the time of writing. All point at 2026 snapshots; none capture history.

### 1.2 Why the current YAML approach falls short

The YAML format was designed (correctly, given v1.7 scope) as a
"current-snapshot contract". `universe_yaml_spec.md` § 1 makes this
explicit. Bolting historical intervals onto the YAML would:

- **Inflate file size dramatically.** NIFTY 500 with 10 years of quarterly
  membership history is a ~5000-line YAML; the lint test
  (`tests/data_pipelines/test_universe_yaml_lint.py`) would have to grow
  schema validation for interval semantics, derivation invariants (e.g.,
  `nifty500 == nifty100 ∪ nifty_midcap_150 ∪ nifty_smallcap_250` must
  hold per-date, not just per-current-snapshot), and date overlap rules.
- **Break the YAML's role as a hand-editable source.** Today a refresh is
  "rerun the curl, regenerate the file, commit." A history-aware YAML
  would require a diff-and-merge step that's painful to do by hand and
  hard to lint.
- **Conflate two different access patterns.** The current YAML is read
  *once at module import* (cheap, in-process cache via `lru_cache`).
  Point-in-time membership queries are date-indexed and benefit from
  the SQL `WHERE valid_from <= ? AND (valid_to IS NULL OR ? < valid_to)`
  pattern. Different storage shapes for different workloads.

The right shape is: **YAML stays the current-snapshot + lint-checked
authoritative source of the present list; SQL captures the temporal
dimension.**

### 1.3 Why this matters specifically for backtesting v1

`docs/backtesting/goal.md` opens with:

> **Look-ahead-bias elimination is structural, not conventional.** The
> engine must make it impossible to trade on information the caller
> hasn't observed, rather than relying on the caller to avoid doing so.

The engine itself enforces price-data look-ahead via the master timeline
and two-phase step lifecycle. But it has nothing to enforce
*universe-membership* look-ahead — that's a property of the caller's
data input. If the caller passes "NIFTY 50 as of 2026" as the universe
for a 2018 backtest, the engine accepts it; the bias is upstream.

So: backtesting v1's structural guarantee is **load-bearing on the
universe-loader being honest about membership-at-date**. Today the loader
isn't honest; this plan fixes that.

### 1.4 Why this is deferrable for gbdt v1 screening

For the current `gbdt` H=25 / H=100 screening work, the bias is
**tolerable, but worth tracking**:

- The pooled-panel CatBoost model with asset-agnostic features and
  uniqueness weights is not making per-stock identity-bound bets — it
  learns rank/zscore patterns that should generalize across the universe.
  A survivorship-biased panel does shift the base rate and the
  positive-class composition, but it doesn't catastrophically invalidate
  the model's screening utility.
- The metric of interest is forward-looking: deploy the model on the
  *current* universe and ask "which top-K names hit the threshold over
  the next H days?" The current universe is *not* survivorship-biased
  (those are the actual tradeable names today).
- Historical training-set bias affects the AUC / Brier estimates that
  `gbdt` reports, but those are decision-support metrics for the user,
  not deployment numbers. The "PASS / FAIL" call in `report.md` is
  qualitative.

For backtesting v1, by contrast, the *reported PnL* is the deliverable.
A backtest run on a survivorship-biased universe produces a number that
looks like a strategy result but isn't — it's a strategy result
conditioned on perfect hindsight about which stocks would survive. That
class of bug is unacceptable for the deliverable.

So: **gbdt can wait for V2 to land; backtesting v1 cannot ship
trustworthy numbers without it.** The priority follows from the
customer.

---

## § 2. Scope

### 2.1 IN scope (this plan)

1. **Persistent storage** of NSE index membership-over-time in the
   `data_pipelines` cache (table location + interval representation per
   § 4).
2. **Refresh mechanism** that (a) pulls the current constituent CSV from
   `archives.nseindia.com`, (b) diffs it against the last-known
   membership rows in the DB, (c) inserts only the deltas. Idempotent.
3. **Universe-loader API extension**:
   `data_pipelines.universe_members(universe_name, as_of=<date>) -> list[str]`
   returning the constituent list valid on `as_of`. Plus
   `universe_members_with_meta(...)` for the agent-tool surface.
4. **Migration of existing YAMLs** — first-pull of every currently-shipped
   universe seeds one initial `valid_from=<listed_at in YAML>, valid_to=NULL`
   row per ticker. The YAML stays in the repo as the authoritative *current*
   list + lint-checked seed; the DB owns the temporal dimension.
5. **Scheduling** — how the refresh job actually gets called on a cadence
   (per § 5).
6. **Test coverage** specified per phase (per § 8).

### 2.2 OUT of scope (explicit non-goals)

- **US universes (S&P 500, NASDAQ-100, Russell 1000)** — the
  domain-specific data sources are completely different (S&P Dow Jones
  is paywalled; NASDAQ publishes via PR; Russell is FTSE). Each US index
  family is a separate vendor relationship and a separate scraping
  story. Deferred to a per-domain plan once an actual `backtesting` user
  asks for it. The schema designed here (§ 4) is domain-agnostic enough
  that adding `us_equities_members` later is a copy-paste + new
  populator, not a refactor.
- **Ticker renames / corporate identity changes.** If a company changes
  its ticker symbol (HDFC → HDFCBANK merger, the ETERNAL renaming) the
  *ticker* changes; both old and new are valid identifiers at different
  times. We will store them as separate `(ticker, valid_from, valid_to)`
  rows — the temporal join "happens to work" — but we do **not** ship a
  symbol-identity-mapping table in this plan. That's a separate problem
  ("which ticker today is the successor of which ticker yesterday?")
  that interacts with corporate-action data we don't carry yet.
- **Corporate-action adjustments to OHLCV** — already handled by the
  `adj_close` column in the price-data layer.
- **Float-adjusted weights / market-cap weights** — schema has a nullable
  `weight` column (§ 4) for future-proofing, but no weight populator in
  this plan. NSE doesn't publish per-constituent weights on the CSV
  endpoints; lifting weights would require a separate index-fact-sheet
  scrape.
- **Sub-daily membership changes.** All NSE rebalances apply at market
  open on a published date; membership is a daily concept. Sub-daily
  interval representation is not needed.
- **An agent-callable "/refresh-constituents" skill.** Refresh is a
  cron / scheduled-task concern (per § 5), not an interactive verb. If a
  use case emerges, lift later.

---

## § 3. Forward-only vs historical-backfill — the **required choice**

### 3.1 The two paths

**Path A — Forward-only.** Start tracking membership from the date the
refresh job is first deployed. The very first run writes one
`(index, ticker, valid_from=DEPLOY_DATE, valid_to=NULL)` row per
constituent of the current YAML. Subsequent refreshes detect deltas and
emit `valid_to` updates + new rows.

For any `as_of` *before* `DEPLOY_DATE`, the loader falls back to the YAML
snapshot (with a logged warning) — the YAML's `listed_at` date is the
boundary, and dates earlier than that get the current snapshot. This is
*exactly today's behavior* for old dates, just made explicit and
warning-flagged.

- Cost: ~1 day of implementation (schema + populator + loader). Refresh
  runs are a few KB of writes per quarter.
- Reliability: high. Single source (the live CSV endpoint that already
  works in production today, per `sectoral_fetch.py`).
- Backtesting value: **zero immediate, growing linearly over time.** The
  first year of backtests still runs on a hindsight universe; year five
  has four years of honest history and one of hindsight.
- Half-broken risk: low. The mechanism either works (data flows) or
  doesn't (data doesn't flow, log warns, loader falls back to YAML).

**Path B — Historical backfill.** In addition to forward-only, also
*reconstruct* historical membership back to some target date (proposal:
2015, to match the deep-history price seed in the v1.7 plan).

Sources for backfill — researched briefly during this plan
(see § 3.2):

1. **`niftyindices.com/press-release`** — official rebalance announcements
   as PDFs (`ind_prs<DDMMYYYY>.pdf`). Authoritative, lots of irrelevant
   PRs interleaved (Shariah-index reviews, methodology updates, factsheet
   releases), per-PR PDF parsing needed, naming convention drift over the
   ~10-year window. Estimated: ~40 relevant PRs per year × 10 years × 5–10
   min hand-parsing each = **30–80 hours of one-time work** for NIFTY 50 +
   100 + 500 + sectorals + thematics, with low confidence the parser
   generalizes.
2. **Wikipedia "NIFTY 50" / "NIFTY 500" articles** — community-maintained
   historical change lists. Semi-structured (HTML tables). Covers NIFTY
   50 back to 1996 with reasonable density. Coverage of NIFTY Midcap 150
   / Smallcap 250 / sectorals / thematics is **sparse-to-nonexistent**.
3. **NSE archives PDF index methodology + fact sheets** — give the
   *current* constituents at the document's publication date; older
   editions of the same document live under stable URLs only inconsistently.
4. **Commercial vendor (e.g., NSE Data Vending Feed, Refinitiv, Bloomberg)**
   — definitive, paywalled, not on the table for an unfunded research repo.

- Cost: 30–80 hours of scraping + parser work; ongoing parser maintenance
  as NSE renames documents.
- Reliability: **medium for NIFTY 50, low for everything else.** Wikipedia
  is the only semi-structured source and its coverage drops sharply
  outside the flagship index. PDF parsing is brittle (NSE has changed
  PR formatting at least twice in the 2015–2026 window per quick
  inspection).
- Backtesting value: **high if it works.** A 10-year backtest with
  point-in-time NIFTY-50 membership is the gold-standard input.
- Half-broken risk: **high.** If we backfill NIFTY 50 cleanly but NIFTY
  Midcap 150 only partially, downstream modules either silently use
  partial data (worst case — they think they have membership and they
  don't, for missing dates) or have to know which (universe, date)
  combinations are reliable. That's exactly the kind of
  "conventional rather than structural" bias-elimination that the
  backtesting goal doc rejects.

### 3.2 Brief feasibility check on the backfill path (research done)

Confirmed during this plan:

- `https://www.niftyindices.com/press-release` exists and is the
  canonical place. URL pattern `Press_Release/ind_prs<DDMMYYYY>.pdf` is
  stable for at least 2022–2025.
- Sample PRs returned by web search:
  - `ind_prs15092023.pdf` (Sep 15, 2023): Shariah-index replacements +
    other.
  - `ind_prs20102022_1.pdf` (Oct 20, 2022): replacements in Shariah and
    SME Emerge. Note the `_1` suffix — multiple PRs per day need
    disambiguation.
- NIFTY 50 itself rebalances **semi-annually** (Jan 31 / Jul 31 cutoff,
  effective end of Mar / Sep) per the methodology PDF. Other indices
  (NIFTY 500, Midcap 150, etc.) rebalance on different cadences. So
  scraping NIFTY 50 alone is ~20 PRs over 10 years; doing the full 30+
  universes is closer to ~200 PRs.
- Wikipedia "NIFTY 50" article has dated historical constituent entries
  back to 1996 (e.g., "Dr. Reddy's Laboratories from 22 April 1996 to 14
  May 1997"). The talk page acknowledges full change-history is on the
  Nifty Indices website.

### 3.3 Recommendation

**Path A — Forward-only, with a documented limitation flag in the
backtesting loader.**

Reasoning:

1. **The bias-vs-effort trade-off is asymmetric the wrong way for
   backfill.** Spending 30–80 hours on a parser that lands incomplete
   coverage (NIFTY 50 well, midcaps poorly, sectorals worse) replaces
   one known bias (survivorship) with two (survivorship for the
   non-flagship indices + parser-error noise for the ones we did
   scrape). The latter is harder to reason about than the former.
2. **The clock starts now either way.** A forward-only deploy in week 1
   has captured *some* honest history by month 12 — and that 12 months
   is the freshest, most-relevant slice for any current strategy
   research. Backfill would deliver more of the *historical* slice but
   doesn't accelerate when we have honest data for *2026 onward*.
3. **Backtesting v1 can structurally encode the limitation.** Per the
   `goal.md` "look-ahead is structural" rule, the right move is for the
   backtest engine to **refuse to start** with `start_date < DEPLOY_DATE`
   when the universe-membership coverage is forward-only-since-`DEPLOY_DATE`.
   This makes the bias *impossible*, not just *unlikely*. Until that
   refusal is wired up, the loader returns a `coverage_warning="forward_only;
   pre_2026-MM-DD membership reflects current YAML snapshot"` field
   alongside the ticker list, which backtesting v1 can read and refuse
   on.
4. **The forward-only path doesn't preclude backfill later.** If a
   funded need emerges (an actual strategy that requires 2015 onward
   point-in-time membership), we can write a one-off backfill script
   that inserts `(valid_from, valid_to)` rows into the same table the
   forward-only mechanism is already populating. The schema (§ 4)
   accommodates this without change. Path A → Path B is a clean
   superset; Path B → Path A doesn't even make sense.
5. **Path A respects the data_pipelines "honest about failure" rule.**
   The coverage gap is explicit (logged warning, returned in meta);
   nobody can accidentally consume historical membership and think it's
   authoritative.

**Runner-up trigger** ("if X changes, we'd switch to historical
backfill"):

> A backtesting v1 user proposes a strategy whose validation requires
> ≥3 years of point-in-time NIFTY 50 membership AND is willing to fund
> the parser work (or someone donates a clean Wikipedia → CSV export of
> the change history). At that point, scope a separate `V3_PLAN.md` for
> NIFTY-50-only backfill (not the full universe family); leave the
> non-flagship indices on forward-only.

---

## § 4. Schema

### 4.1 Table location

The new tables live in **`data/processed.db`** (the same SQLite file
that holds `nse_equities_data` / `nse_equities_meta`). Reasoning:

- One database file is one backup unit. Splitting membership into a
  sibling DB doubles the operational footprint without buying anything.
- The single-writer-per-`data_root` contract (`CLAUDE.md` § Data and
  configs) already applies. Membership writes are infrequent (once per
  quarter at most, per universe) and short — they will not contend with
  the price-data writer in practice. The threading.Lock in
  `cache.py::_get_lock` is keyed on the DB path and serializes any
  concurrent in-process write.
- Joining membership ⨯ price data in SQL is enabled out of the box (e.g.,
  agent-tool surface: "give me close prices for current NIFTY 50
  members on 2020-03-12" is one query).

### 4.2 DDL

Two tables, both auto-created on first refresh-job run (mirroring the
auto-create pattern in `cache.py::_ensure_tables`):

```sql
CREATE TABLE IF NOT EXISTS nse_index_members (
    -- The universe this row belongs to. Matches the universe YAML name
    -- (e.g., 'nifty50', 'nifty_bank') — NOT the cache-prefixed index
    -- ticker. Loader translates from universe_name to index_ticker via
    -- the existing gbdt registry block when needed.
    universe_name TEXT NOT NULL,

    -- The constituent ticker, fully prefixed ('NSE:RELIANCE'),
    -- matching the YAML's `tickers:` list convention.
    ticker        TEXT NOT NULL,

    -- Inclusive lower bound on membership validity (ISO date string,
    -- YYYY-MM-DD). The constituent is in the universe on this date.
    valid_from    TEXT NOT NULL,

    -- Exclusive upper bound (ISO date string). NULL = still a member.
    -- Stored as TEXT to keep symmetry with valid_from and to make
    -- range queries straightforward (TEXT compares ISO dates
    -- lexicographically, same as a real ORDER BY).
    valid_to      TEXT,

    -- Optional per-constituent weight at last observation. Currently
    -- always NULL — NSE CSV endpoints don't publish weights. Reserved
    -- so a future fact-sheet scraper can populate without schema
    -- migration.
    weight        REAL,

    -- Provenance.
    source_csv_url   TEXT NOT NULL,
    fetched_at_utc   TEXT NOT NULL,

    PRIMARY KEY (universe_name, ticker, valid_from)
);

CREATE INDEX IF NOT EXISTS nse_index_members_lookup
    ON nse_index_members (universe_name, valid_from, valid_to);

CREATE TABLE IF NOT EXISTS nse_index_members_meta (
    -- One row per universe. Tracks refresh-job status.
    universe_name           TEXT PRIMARY KEY,

    -- ISO date of the most recent successful refresh.
    last_refresh_date       TEXT NOT NULL,
    last_refresh_at_utc     TEXT NOT NULL,

    -- ISO date of the earliest valid_from row in the data table for
    -- this universe. The coverage_start for downstream callers.
    coverage_start          TEXT NOT NULL,

    -- Did the most recent refresh insert any delta rows?
    -- (For human-readable logs / dashboard.)
    last_refresh_delta_count INTEGER NOT NULL DEFAULT 0,

    -- Failures during the most recent attempt (JSON-serialized list of
    -- {url, http_status, error}). Empty list = clean run.
    last_refresh_errors_json TEXT NOT NULL DEFAULT '[]'
);
```

### 4.3 Why these choices

- **`(universe_name, ticker, valid_from)` as composite PK.** Rules out
  duplicate-insert bugs in the refresh job (the same constituent being
  inserted twice with the same `valid_from` is a programmer error). Does
  *not* rule out "ticker leaves the universe, then rejoins later" — that
  inserts a new row with a new `valid_from`, which is what we want.
- **`valid_from` inclusive, `valid_to` exclusive.** The standard temporal
  representation. `WHERE valid_from <= ? AND (valid_to IS NULL OR ? <
  valid_to)` is the canonical "as-of" query.
- **TEXT for dates.** SQLite's `TIMESTAMP` affinity is advisory only;
  storing dates as ISO `YYYY-MM-DD` strings preserves lexicographic
  ordering equal to date ordering and matches the
  `range_start`/`range_end` columns already in `nse_equities_meta`.
- **`weight REAL NULL`.** Future-proofing; no populator in this plan.
- **`source_csv_url` + `fetched_at_utc` per row.** Provenance per § "Honest
  about failure". A row's source is visible without joining to the meta
  table — useful when debugging "where did this membership claim come
  from".
- **Index on `(universe_name, valid_from, valid_to)`.** The hot query is
  per-universe + date-range; this index makes the planner happy without
  doing a full table scan once the table grows past a few thousand rows
  (~3 universes × 100 tickers × 4 quarters/year × 10 years ≈ 12 k rows
  for the full Nifty family if backfill ever lands).

### 4.4 Migration story

First time the refresh script runs against an existing `processed.db`:

1. `CREATE TABLE IF NOT EXISTS` runs cleanly (no conflict with existing
   tables — names are new).
2. For each universe in the configured set: load the YAML to get the
   current `tickers:` list, INSERT one row per ticker with
   `valid_from = <YAML's listed_at>`, `valid_to = NULL`,
   `source_csv_url = '<yaml-path>'` (marker that this row was seeded
   from the YAML, not from a live fetch),
   `fetched_at_utc = <now>`. This is the seed; subsequent refreshes
   detect deltas against it.
3. Insert one row in `nse_index_members_meta` per universe with
   `coverage_start = <YAML's listed_at>` and `last_refresh_delta_count = 0`.

Re-running the schema creation on a migrated DB is a no-op (the
`CREATE TABLE IF NOT EXISTS` is idempotent; the seed step **must** be
guarded by `SELECT COUNT(*) FROM nse_index_members WHERE universe_name = ?`
to avoid re-seeding on top of forward-only delta rows).

---

## § 5. Refresh mechanism

### 5.1 Cadence

**Monthly**, on the 5th of each month, UTC. Justification:

- NSE rebalances on different cadences per index (NIFTY 50 semi-annual,
  NIFTY 500 quarterly, sectorals semi-annual). Running monthly catches
  any of them within ~30 days regardless of which index changed when.
- Monthly is cheap (~30 HTTP requests/month against `archives.nseindia.com`
  — well below any rate-limit threshold).
- Aligning with rebalance quarters (Mar/Jun/Sep/Dec) is tempting but
  fragile — NSE's "effective date" of a rebalance is announced a few weeks
  in advance, and announcements occasionally move. A monthly polling job
  is more robust than guessing the announcement schedule.
- A weekly job buys nothing — rebalances never happen weekly.

**The 5th of the month** is chosen to be safely past any quarter-end
processing on NSE's side (end-of-Mar rebalances typically settle on the
website within a few business days).

### 5.2 Where the script lives

```
scripts/data_pipelines/refresh_nse_constituents.py
```

Mirroring the existing pattern (`sectoral_fetch.py`, `broad_market_fetch.py`).
Surface:

```
python -m scripts.data_pipelines.refresh_nse_constituents
    [--universes nifty50,nifty100,...]      # default: all NSE universes
    [--dry-run]                              # log diffs but don't write
    [--data-root data]
```

The script:

1. Loads the list of registered NSE universes (filter the
   `configs/data_pipelines/domains/nse_equities/universe_*.yaml` directory).
2. For each universe: looks up the `archives.nseindia.com` CSV URL from a
   small in-repo lookup table (per-universe `csv_slug` — mirroring the
   `SECTORAL_INDICES` tuple in `sectoral_fetch.py`). The lookup table
   lives in
   `src/data_pipelines/domains/nse_equities/constituent_sources.py` so
   it's importable and testable.
3. Curls the CSV (with `Mozilla/5.0` UA per the NSE-quirks memory),
   filters `DUMMY*` tickers, parses to the current ticker set.
4. Reads existing membership rows from `nse_index_members` for the
   universe where `valid_to IS NULL` → that's the "currently-believed
   membership."
5. Computes:
   - `added = current_csv - currently_believed`
   - `removed = currently_believed - current_csv`
6. If `len(added) == 0 and len(removed) == 0`: idempotent no-op. Update
   `last_refresh_at_utc` in the meta row; do **not** touch the data
   table. Log "no-op."
7. Otherwise, in one SQLite transaction:
   - For each `removed` ticker: `UPDATE nse_index_members SET valid_to = <today>
     WHERE universe_name = ? AND ticker = ? AND valid_to IS NULL`.
   - For each `added` ticker: `INSERT INTO nse_index_members (...)` with
     `valid_from = <today>, valid_to = NULL`.
   - `UPDATE nse_index_members_meta SET last_refresh_date = <today>,
     last_refresh_at_utc = <now>, last_refresh_delta_count = ?,
     last_refresh_errors_json = '[]'`.
8. Failure mode (HTTP non-200, parse error, etc.): log the failure,
   append it to `last_refresh_errors_json` for the affected universe,
   move on to the next universe. **Do not delete or modify any existing
   rows.** A failed refresh is a no-op on the data table.
9. Exit code: 0 if all universes refreshed successfully or were no-ops;
   1 if any universe failed (so a `cron` wrapper can email on failure).

### 5.3 Detection — diff against `valid_to IS NULL` rows

This is the only safe diff: comparing the live CSV against the set of
currently-believed members. Comparing against the YAML's `tickers:` list
would re-seed forever on day 2 of forward-only operation
(the YAML lags the DB).

### 5.4 YAML-DB drift handling

A subtle gotcha: the YAML's `tickers:` list and the DB's
"currently-believed members" *can drift apart* over time (the DB updates
monthly; the YAML updates only when a human runs `sectoral_fetch.py
--write-yamls` and commits the result).

Policy:

- **YAML is authoritative for the schema-lint test** (today's behavior
  — lint walks the YAML).
- **DB is authoritative for `universe_members(as_of=<today>)`** queries.
- When the human-driven YAML refresh PR lands, the DB is *already* in
  sync (because the monthly refresh job has been tracking deltas all
  along). The YAML commit just brings the human-facing file up to date.
- The loader (§ 6) prefers DB over YAML for any `as_of` ≥ the universe's
  `coverage_start`; falls back to YAML for older `as_of`.

### 5.5 Scheduling

Two paths, pick whichever fits the operational reality:

- **`/schedule` skill** (preferred — already exists in the project per
  the available-skills list). Register the refresh as a cron-style
  routine: `0 6 5 * * uv run python -m scripts.data_pipelines.refresh_nse_constituents`.
  This gets the script triggered without touching the OS-level cron and
  centralizes scheduling under Claude Code.
- **System cron / systemd timer** as the fallback for a deployed
  context. The same CLI is the entry point either way.

Document both in the script's docstring; pick one when the phase ships.

---

## § 6. Universe loader integration

### 6.1 Existing surface

Today:

```python
# src/data_pipelines/domains/nse_equities/universe.py
def load_universe(name: str = "nifty50", ...) -> list[str]:
    """Return the list of identifiers (tickers + indices) in the named universe."""
```

Returns `tickers + indices` concatenated (the format the gbdt panel
loader expects). Cached via `lru_cache` on the YAML path.

### 6.2 New surface

Add at the package level
(`src/data_pipelines/__init__.py`) so consumers don't need to know about
the domain layout:

```python
def universe_members(
    universe_name: str,
    as_of: date | str | None = None,
    *,
    data_root: str | Path = "data",
    include_index_tickers: bool = False,
) -> list[str]:
    """Constituent tickers in `universe_name` as of `as_of`.

    Parameters
    ----------
    universe_name : str
        Universe slug matching the YAML filename
        (`nifty50`, `nifty_bank`, ...).
    as_of : date | str | None
        Date to query. `None` means "today" (default).
    data_root : str | Path
        Cache root, same convention as `fetch()`.
    include_index_tickers : bool
        If True, append the universe's index tickers (`NIFTY:50`, ...)
        per the historical `load_universe` behavior. Default False because
        the typical caller (a panel loader) wants only constituents and
        treats the index as a separate fetch.

    Returns
    -------
    list[str]
        Fully-prefixed constituent identifiers (`NSE:RELIANCE`, ...),
        sorted alphabetically for deterministic output.

    Resolution order
    ----------------
    1. Look up the universe in the YAML registry to validate the name
       and pull `indices` + `coverage_start_yaml = listed_at`.
    2. If `as_of >= coverage_start_db` for this universe (per
       `nse_index_members_meta`):
          SELECT ticker FROM nse_index_members
          WHERE universe_name = ?
            AND valid_from <= ?
            AND (valid_to IS NULL OR ? < valid_to)
          ORDER BY ticker
       Return that list. This is the honest path.
    3. Else (`as_of` predates DB coverage): emit a *single* warning
       through `logging.getLogger("data_pipelines.universe_members")`
       with text:
          "as_of=<date> predates DB coverage_start=<date> for
           universe=<name>; falling back to YAML snapshot
           (listed_at=<date>) — point-in-time membership not available
           for this date"
       Then return `load_universe(name)` (the current behavior). This
       is the YAML fallback. The warning is per-process-per-(universe,
       date) deduped via a module-level set so we don't spam logs in
       loops.
    4. If `universe_name` is not in the YAML registry: raise
       `FileNotFoundError` (matching `load_universe`'s current behavior).
    """
```

Plus a meta-flavored sibling for the agent surface:

```python
@dataclass
class UniverseMembersMeta:
    universe_name: str
    as_of: date
    source: Literal["db_pit", "yaml_fallback"]
    coverage_start: date           # earliest reliable date
    row_count: int
    coverage_warning: str | None   # "forward_only;..." or None

def universe_members_with_meta(
    universe_name: str, as_of: date | str | None = None, *,
    data_root: str | Path = "data", include_index_tickers: bool = False,
) -> tuple[list[str], UniverseMembersMeta]:
    """Same as `universe_members` plus a JSON-serializable meta object
    describing the resolution path. Use this from the backtesting engine
    (it can refuse to start if source == 'yaml_fallback' AND the
    coverage_warning is set)."""
```

### 6.3 What stays the same

- `nse_equities.universe.load_universe(name)` — unchanged. v1.7
  consumers (the existing gbdt path that doesn't care about historical
  membership) keep working bit-for-bit. The new function is **additive**.
- `tests/data_pipelines/test_universe_yaml_lint.py` — unchanged. The
  YAML is still the authoritative file for the lint.
- The gbdt registry block in `configs/gbdt/default.yaml` — unchanged. It
  references the YAML, not the DB.

### 6.4 What changes for `gbdt`

Nothing in this plan. The gbdt panel loader (`gbdt.data.load_panel`)
will continue calling `load_universe()` and getting the current snapshot.
A separate follow-up plan (likely `gbdt/V2_PLAN.md`) would migrate it to
`universe_members(as_of=<panel_end_date>)` once both modules are ready.
The new function does not destabilize gbdt by virtue of being unused.

### 6.5 What changes for `backtesting`

`backtesting` v1 is encouraged to call `universe_members_with_meta(...)`
in its setup phase and refuse to start when
`meta.source == "yaml_fallback"` for the backtest's `start_date`. This is
the structural enforcement of the look-ahead-bias rule. The mechanism
lives in this plan; the policy enforcement is `backtesting`'s call.

---

## § 7. Phasing

Five phases. Each is its own PR; each builds on the previous. Total
effort estimate: **6–9 hours of focused work** spread across the phases.

| Phase | Deliverable                                              | Success criteria                                                                 | Effort  | Depends on    |
|-------|----------------------------------------------------------|----------------------------------------------------------------------------------|---------|---------------|
| 1     | Schema + migration + low-level writer + tests            | DDL applied on first call; idempotent re-run; happy-path insert+query unit tests | 1.5 h   | —             |
| 2     | Seed script for 5 priority indices                       | NIFTY 50/100/Midcap-150/500/Next-50 seeded; meta row written per universe        | 1 h     | Phase 1       |
| 3     | Refresh script + scheduling                              | Re-running the refresh on the seed-state is a no-op; manual delta test passes    | 1.5 h   | Phase 1, 2    |
| 4     | `universe_members(as_of=)` loader + meta + tests         | YAML-fallback warns; DB-PIT returns correct list at interval boundaries          | 1.5 h   | Phase 1, 2    |
| 5     | Roll out refresh to remaining ~25 universes              | All 30+ NSE universes in `nse_index_members_meta` with coverage_start ≤ today   | 1 h     | Phase 3, 4    |

### Phase 1 — Schema + writer

**Files added:**
- `src/data_pipelines/domains/nse_equities/members_cache.py` — pure
  schema/CRUD module mirroring the shape of `cache.py`. Functions:
  `ensure_tables`, `read_members(universe_name, as_of)`,
  `apply_refresh(universe_name, current_set, source_url, today)`,
  `read_meta(universe_name)`, `seed_from_yaml(universe_name, yaml_path)`.
- `src/data_pipelines/domains/nse_equities/constituent_sources.py` — the
  per-universe `csv_slug` lookup table (extracted from `sectoral_fetch.py`
  for re-use).
- `tests/data_pipelines/test_members_cache.py` — happy path + idempotency.

**Success:** unit tests pass on a fresh DB; schema diff against a
manually-created DB matches the DDL in § 4.

### Phase 2 — Seed script

**Files added:**
- `scripts/data_pipelines/seed_nse_constituents.py` — loops over the
  five priority universes, calls `seed_from_yaml`. Idempotent.

**Success:** running the script twice produces no extra rows the second
time. `sqlite3 data/processed.db "SELECT universe_name, COUNT(*) FROM
nse_index_members GROUP BY universe_name"` shows expected ticker counts
per universe.

### Phase 3 — Refresh script + scheduling

**Files added:**
- `scripts/data_pipelines/refresh_nse_constituents.py` — per § 5.
- `tests/data_pipelines/test_refresh_nse_constituents.py` — mocks the
  CSV fetch; verifies the diff-and-apply logic on synthetic deltas.

**Scheduling task** (separate doc, not a code file): update
`.claude/skills/schedule/README.md` (if one exists) or
`docs/data_pipelines/README.md` with the recommended cron expression.

**Success:** mock-based delta tests pass; manual `--dry-run` against the
real archives URL shows "no delta detected" on day 1 post-seed.

### Phase 4 — Loader integration

**Files added:**
- `src/data_pipelines/__init__.py` — exports `universe_members`,
  `universe_members_with_meta`, `UniverseMembersMeta`.
- `src/data_pipelines/universe_resolver.py` — actual implementation
  (kept out of the domain folder because the public API is package-level,
  and routing-by-domain logic may need to grow for future
  `us_equities_members` work).
- `tests/data_pipelines/test_universe_members.py` — interval boundaries,
  YAML fallback path, dedup-warn behavior.

**Success:** all four resolution-order branches in the docstring (§ 6.2)
have a test. Existing `tests/data_pipelines/test_universe_yaml_lint.py`
unchanged + still passing.

### Phase 5 — Full rollout

**Changes:**
- Extend `constituent_sources.py` to cover the remaining ~25 NSE
  universes from the Sectoral + Thematic catalogs.
- Re-run the seed script for the new ones.
- One-off: take a snapshot of `nse_index_members_meta` and commit it as
  a fixture for the regression test.

**Success:** `nse_index_members_meta` has 30+ rows; the first
forward-only delta detected during the monthly refresh updates the right
universe without disturbing the others.

---

## § 8. Tests required per phase

### Phase 1 tests

- **`test_ensure_tables_idempotent`** — call `ensure_tables` twice on
  the same DB; verify no error, no schema drift.
- **`test_insert_and_query_happy_path`** — insert 3 members with
  open-ended `valid_to`; query with `as_of=today` returns all 3.
- **`test_interval_boundary_inclusive_lower`** — insert a member with
  `valid_from = 2025-01-15`; query with `as_of = 2025-01-15` includes
  them (inclusive lower).
- **`test_interval_boundary_exclusive_upper`** — insert with
  `valid_to = 2025-06-30`; `as_of = 2025-06-29` includes, `as_of =
  2025-06-30` excludes (exclusive upper, the standard convention).
- **`test_pk_collision_raises`** — two inserts with same
  `(universe_name, ticker, valid_from)` raises `IntegrityError`.
- **`test_seed_from_yaml_idempotent`** — call `seed_from_yaml` twice;
  the second call inserts zero rows (and does not error).

### Phase 2 tests

- **`test_seed_script_creates_meta_row`** — after the seed runs, every
  seeded universe has a `nse_index_members_meta` row with non-empty
  `coverage_start`.
- **`test_seed_script_filters_dummy_tickers`** — fixture YAML
  contains `NSE:DUMMYVEDL` (a real artifact per
  `project-nse-data-quirks.md`); verify it doesn't land in
  `nse_index_members`.

### Phase 3 tests

- **`test_refresh_no_delta_is_noop`** — mock the live CSV to return the
  exact current member set; verify `last_refresh_delta_count == 0` and
  no INSERT statements ran.
- **`test_refresh_add_then_query`** — mock the live CSV to return
  `current_set + {NEWMEM}`; verify a new row with `valid_from = today,
  valid_to = NULL` is inserted; query at `as_of = yesterday` does not
  include it; query at `as_of = today` does.
- **`test_refresh_remove_closes_interval`** — mock the live CSV to
  return `current_set - {OLDMEM}`; verify the existing row gets `valid_to
  = today`; query at `as_of = today` excludes it (exclusive upper); query
  at `as_of = today - 1day` includes it.
- **`test_refresh_replace_is_add_plus_remove`** — joint case; both row
  updates must commit atomically (within one transaction).
- **`test_refresh_csv_404_does_not_modify_data`** — mock the live CSV
  to return HTTP 404 for one universe; verify *that universe's*
  `nse_index_members` rows are unchanged and the error is appended to
  `last_refresh_errors_json`; verify other universes still process.
- **`test_refresh_idempotency_after_delta`** — apply a delta, run the
  refresh again immediately; second run is a no-op (`current_set` now
  matches `currently_believed`).

### Phase 4 tests

- **`test_universe_members_db_pit_path`** — seed + run, query at a date
  inside the coverage window; assert source == `db_pit` and the set
  matches the DB rows.
- **`test_universe_members_yaml_fallback_path`** — query with `as_of`
  before `coverage_start`; assert source == `yaml_fallback`, the set
  matches `load_universe(name)`, and a warning was emitted exactly
  once for the (universe, date) pair across two repeat calls.
- **`test_universe_members_invalid_name_raises`** — unknown universe
  name raises `FileNotFoundError`.
- **`test_load_universe_unchanged_when_db_present`** — install a DB
  with members; assert `nse_equities.universe.load_universe()` still
  returns the YAML-based list (no silent rerouting of the legacy API).
- **`test_existing_yaml_lint_still_passes`** — re-run the existing
  `test_universe_yaml_lint.py` and confirm zero regressions.

### Phase 5 tests

- **`test_all_priority_universes_in_meta`** — assert `len(read_meta_all())
  == len(known_universes)` after rollout.
- **`test_refresh_run_against_all_universes`** — mocked end-to-end
  smoke; one delta per universe; all commit atomically and independently.

---

## § 9. Risks + mitigations

### R1 — NSE archive page format changes

`archives.nseindia.com/content/indices/ind_<slug>.csv` has been stable
for years per the empirical record in `sectoral_fetch.py` and the NSE
quirks memory. But it is an archival page on a website NSE redesigns
occasionally.

- **Likelihood:** medium over a 12-month window.
- **Impact:** the refresh script logs HTTP 404 / parse error per
  universe; existing rows are untouched (per § 5.8); the
  `last_refresh_errors_json` field surfaces the failure; the loader
  falls back to YAML for `as_of >= today`. **No data corruption.**
- **Mitigation:** the refresh script must structurally distinguish
  "page returned HTML help-text instead of CSV" from "page returned a
  CSV with zero rows" (matching the pattern in
  `StooqAdapter._is_apikey_required`). A help-text response is a
  hard failure, not a no-op.

### R2 — Rate limiting

The script makes ~30 GET requests against `archives.nseindia.com` per
monthly run. Sustained traffic levels needed to trigger NSE's anti-bot
gate are an order of magnitude higher. The NSE quirks memory documents
that `jugaad`/`nselib` get blocked (live JSON endpoints) — but the
*archives* endpoint has been reliable.

- **Mitigation:** the script polite-spaces requests at 2 s between
  fetches; uses the documented `User-Agent: Mozilla/5.0` header; runs
  once a month.

### R3 — Backtesting v1 hasn't been used yet against any real strategy

We don't know what the *actual* point-in-time query patterns look like.
The API in § 6 is a best guess. May need to add bulk-query support
(`universe_members_history(...)` returning all `(date, member_set)`
tuples over a range) before backtesting actually runs.

- **Mitigation:** keep Phase 4 minimal (single-date queries only).
  Defer bulk-query to a backtesting-driven follow-up plan once the
  first real strategy lands. The SQL underlying the bulk query is
  straightforward (one INNER JOIN against a date series); the *API
  shape* is the hard part and we can't get it right without a real
  caller.

### R4 — Index renames / splits

NIFTY renames sub-indices occasionally (a recent example: certain
thematic indices renamed in 2023). If the YAML name in the repo doesn't
match the renamed `csv_slug`, the refresh script will 404.

- **Mitigation:** the `csv_slug` is a separate field from the universe
  name (per `constituent_sources.py` design). Renames update the slug
  without touching `universe_name`. The PR that lands the rename also
  updates the slug.
- **Mitigation 2:** the script logs the URL it attempted; a 404 is a
  recoverable hand-edit, not a silent bug.

### R5 — Forward-only coverage gap is a known limitation

The product of Path A (per § 3) is that backtests starting before
`DEPLOY_DATE` use the YAML snapshot. This is documented; the loader
warns; the meta object flags it.

- **Mitigation (policy):** backtesting v1 should refuse to start when
  the meta says `source == "yaml_fallback"` AND the strategy is being
  evaluated for performance numbers (not just smoke-testing the engine).
  This is a `backtesting` concern, not `data_pipelines`'s. Document the
  expectation here so backtesting's V2_PLAN.md picks it up.
- **Mitigation (decision deferred):** in 12 months, if we have a backlog
  of strategies that need pre-2026 honest membership, run the trigger
  in § 3.3 and scope `V3_PLAN.md` for selective backfill.

### R6 — Single-writer contract collision with bulk seeds

If a monthly refresh runs at the same time as a bulk `data_pipelines
fetch ... --back-extend` seed (e.g., a long-running gbdt experiment is
re-seeding deep history), both processes contend on `processed.db`. WAL
mode prevents corruption; the in-process lock prevents intra-process
races; but cross-process the loser waits.

- **Mitigation:** the refresh script's total wall-clock time is
  small (seconds to a couple of minutes for 30 universes). Even a
  blocking wait of a few minutes during a bulk seed is acceptable. No
  code change beyond ensuring the refresh uses the same `cache.py`
  helpers (so it inherits the WAL/lock contract).

---

## § 10. Decisions log

A compact table of the load-bearing choices in this plan + their
reasoning. Future maintainers can scan this without reading the full doc.

| # | Decision                                                    | Why                                                                                                 | Alternatives considered                                                                  |
|---|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1 | Forward-only, no historical backfill (§ 3)                  | Half-broken backfill is worse than known-bounded gap; cost 30–80 h vs 6–9 h                          | Full backfill via Wikipedia + NSE PR PDFs; mixed (NIFTY 50 only)                          |
| 2 | Tables live in `data/processed.db`, not a sibling DB (§ 4.1)| One backup unit; cross-table joins; refresh writes are infrequent so contention is a non-issue       | Separate `data/membership.db`; YAML-only persistence (rejected per § 1.2)                |
| 3 | `valid_from` inclusive, `valid_to` exclusive (§ 4.3)        | Standard temporal SQL convention; `WHERE valid_from <= ? AND (NULL OR ? < valid_to)` is canonical    | Both inclusive (semantically ambiguous at the boundary); separate `is_current` flag (loses interval semantics) |
| 4 | Dates stored as ISO TEXT strings, not as `DATE` type (§ 4.3)| SQLite's `DATE` affinity is advisory; ISO text gives lexicographic = chronological ordering for free; matches `nse_equities_meta` convention | `INTEGER` Julian day (compact but opaque); `TIMESTAMP` (overkill, dates are sub-daily-trivial) |
| 5 | YAML stays authoritative for *current* + lint; DB authoritative for *historical* (§ 5.4) | Two access patterns, two storage shapes; YAML lint contract preserved unchanged                       | Migrate YAMLs to DB-only (loses human-editable cold-start); DB authoritative for everything (forces lint-on-DB refactor) |
| 6 | Refresh monthly, not weekly or quarterly (§ 5.1)            | Captures all rebalance cadences with one schedule; cheap; tolerant of NSE announcement-date slips    | Quarterly aligned with NSE rebalance dates (fragile to schedule shifts); weekly (no value)        |
| 7 | `universe_members()` warns on YAML fallback, doesn't raise (§ 6.2) | Gbdt and other current consumers shouldn't break when calling the new API for old dates; loud-failure goes to `_with_meta` flavor for callers that want it | Raise on fallback (would force `try/except` in callers that already work fine on current snapshot); silently fall back (violates "honest about failure") |
| 8 | New per-domain mechanism, not framework-level (§ 4, § 6)    | Per `adding_a_domain.md` § "Universe maintenance is a per-domain concern"; us_equities will get its own; framework crystallization waits for domain #3 to apply | Lift to framework now (premature; would crystallize on one example) |
| 9 | API at `data_pipelines.universe_members(...)` package level, implementation in `universe_resolver.py` (§ 6.2) | Consumers don't need to know about per-domain layout; matches the `fetch()` precedent           | Per-domain export `nse_equities.universe_members(...)` (forces consumers to know the domain) |
| 10 | Phase 1–5 are 5 separate PRs (§ 7)                         | Each phase ships an independently-useful slice (schema testable before populator runs; populator testable before loader uses it); easier to review | One mega-PR (harder to review, harder to revert one phase if it goes wrong) |

---

## Open questions for review

These are decisions this plan deliberately *didn't* lock in — they need
a sign-off from someone with broader context before the implementation
PRs start.

1. **Should `listed_at:` in the YAMLs be deprecated** once the DB
   authoritative for historical queries? Current proposal: keep it, as
   it's the lint-checked + human-readable proof of when this snapshot was
   pulled. But if it confuses consumers ("is the YAML date or the DB date
   the source of truth?") we might prefer to drop it from the YAML and
   surface it only from `nse_index_members_meta.coverage_start`.

2. **Scheduling mechanism** — `/schedule` skill vs system cron — see
   § 5.5. Both work; the project has a `/schedule` skill already, which
   argues for using it. But `/schedule` runs depend on the harness being
   alive; a long unattended period would skip refreshes silently. System
   cron is hand-off-able. Recommend `/schedule` for now, with a note in
   the script docstring that it can also be invoked from cron.

3. **`weight REAL` column** — keep nullable + unused, or drop until a
   weight populator exists? Current proposal: keep (zero cost; lets the
   weight populator land as an Edit, not a schema migration).

4. **Backtesting policy enforcement** — should backtesting v1 *refuse to
   start* on `yaml_fallback` source, or merely *warn loudly*? This is a
   `backtesting` decision, not a `data_pipelines` one — flagging it
   here so backtesting v1's plan picks up the question rather than
   silently inheriting whichever default the loader gives.

5. **US universes** — the scope (§ 2.2) excludes US universes. Is that
   acceptable indefinitely, or do we want an explicit V3 plan stub
   committed alongside this one ("when an `us_equities` consumer of
   point-in-time membership appears, scope V3")? Current proposal: no
   stub; cross that bridge when a real US backtesting strategy lands.
