---
name: project-nse-data-quirks
description: NSE adapter chain quirks surfaced during NIFTY universe self-service — jugaad/nselib often blocked, curl from archives.nseindia.com works; processed.db-wal can corrupt after FS wedge needing /tmp workaround; constituent lists carry DUMMY* placeholders that must be filtered.
metadata:
  type: project
---

Observed during 2026-05-26 v1 GBDT experiment universe self-service runs. These are real ops facts about working with NSE data via this project's `data_pipelines` adapter chain.

**1. Live API calls are often blocked.**
- `jugaad_data.NSELive.live_index('NIFTY NEXT 50')` returns non-JSON HTML interstitial (NSE anti-bot gate). Raises `JSONDecodeError`.
- `nselib.capital_market.niftynext50_equity_list()` fails with `ssl.SSLCertVerificationError` on some systems (Python urllib doesn't trust the cert chain).
- Both presented during nifty100 universe self-service. Same risk for nifty_midcap_150 and nifty500.

**Reliable fallback:** `curl -L -A "Mozilla/5.0" https://archives.nseindia.com/content/indices/<list>.csv` returns clean CSV. Confirmed working URLs:
- `ind_nifty50list.csv`
- `ind_niftynext50list.csv`
- `ind_nifty100list.csv`
- `ind_niftymidcap150list.csv`
- `ind_nifty500list.csv`

**2. Constituent lists carry placeholder tickers.**
The official archives.nseindia.com lists include 4× `DUMMYVEDL*` entries (Vedanta demerger pseudo-tickers — non-tradeable). Filter them out before universe registration: `if symbol.startswith("DUMMY"): skip`. Otherwise `data_pipelines.fetch()` will spend ~80 s per dummy retrying-then-failing.

**3. `processed.db-wal` can be filesystem-corrupted after a disk wedge.**
After the 2026-05-26 disk wedge (see `[[feedback-disk-wedge-pattern]]`), `data/processed.db-wal` returned I/O errors on stat/read/unlink — independent of the underlying DB integrity. Symptoms: `sqlite3 data/processed.db 'PRAGMA quick_check'` returns `unable to open database file`. The DB itself (`processed.db`) is fine when copied to a clean FS.

**Current scratch path** (as of 2026-05-27): a persistent, non-tmpfs scratch directory referenced as `${SCRATCH_CACHE}` in project memories — the literal lives in per-user memory `scratch-cache-path`. `/tmp/exp_data` is symlinked to `${SCRATCH_CACHE}` for script back-compat. **Do NOT put the cache on raw `/tmp` (tmpfs)** — we lost ~12 hr of NSE Broad+Sectoral back-extends to a tmpfs wipe on 2026-05-27. The original tmpfs-warning note in this memory was prescient and is now paid for.

**Workaround / re-bootstrap** (if the WAL corrupts again or the scratch path is lost):
```bash
# ${SCRATCH_CACHE} = persistent scratch dir on a non-corrupted FS; per-user memory `scratch-cache-path`.
mkdir -p "${SCRATCH_CACHE}" && cp data/processed.db "${SCRATCH_CACHE}/processed.db"
# Link raw/ from main checkout (preserves immutable provider downloads).
# Substitute <main-checkout-abs> with the repo's absolute path on this machine.
ln -sfn <main-checkout-abs>/data/raw "${SCRATCH_CACHE}/raw"
# Re-symlink /tmp/exp_data for any scripts hardcoded to the old path
ln -sfn "${SCRATCH_CACHE}" /tmp/exp_data
# Then in each worktree:
rm -rf data && ln -s /tmp/exp_data data
git update-index --skip-worktree data/.gitkeep
```

**Symptom + recovery decision tree**:
- `sqlite3 data/processed.db 'SELECT 1'` errors → WAL corrupted → re-bootstrap as above (no data loss; just need to copy main DB to fresh scratch).
- `ls data/processed.db` returns "no such file" → cache lost (tmpfs wipe or similar) → restore from main checkout's `data/processed.db` (preserved as a snapshot from 2026-05-26 morning at minimum) + re-run any post-snapshot fetches via `python -m scripts.data_pipelines.broad_market_fetch` (post PR #38 — structured logs; see § 7 below).

**4. Per-ticker fetch wall time.**
With `nselib` blocked by single-day rejection errors, `data_pipelines.fetch()` on a new NSE ticker takes ~30–80 s per ticker (retry storms on holiday-adjacent dates + delisting placeholders + Yahoo "possibly delisted" warnings). For a 150-stock universe, that's ~2–3 hours sequential. Use per-ticker 120 s hard timeout (mirror `scripts/seed_nifty50_deep.sh` from PR #6).

**5. Shallow-cache silent drop trap.**
Many tickers in the cache from earlier `data-seed-nifty-total` work were seeded with `start=2019-01-01` (~1500–1588 rows). The default `min_rows=1600` gate in `gbdt.data.load_panel` silently drops them. For wide-universe experiments where many tickers fall in this gap (Exp 2 dropped 53/100 silently), the universe self-service phase should also call `data_pipelines.fetch(..., back_extend=True, start='2015-01-01')` on cached-but-short tickers to push them above the bar. The `/gbdt-experiment` skill encodes this as policy: `back_extend=True` is the unconditional default on every per-ticker fetch during Pre-flight § 3 (the library `data_pipelines.fetch()` keeps `back_extend=False` to stay surprise-free for non-gbdt callers — the default flip lives in the skill, not the library).

**6. NSE jugaad payload has in-payload duplicate dates for 3 known dates (FIXED in PR #37).**
The jugaad NSE bhav archive carries re-issued/corrected entries for at least 3 historical dates: **2011-05-06**, **2011-08-09**, **2011-10-24**. Before PR #37, this triggered `sqlite3.IntegrityError: UNIQUE constraint failed` on `back_extend=True` fetches for any ticker with rows around those dates (RELIANCE was the canonical repro). Post PR #37, `data_pipelines.cache.merge_cache()` dedupes by `(ticker, date)` before write + emits a WARNING-level log naming the source provider. Future occurrences are visible in the log — if you see `WARN merge_cache: dropped N duplicate rows from <provider>` for non-jugaad providers, that's a NEW data-quality issue worth filing.

**7. Broad fetch script invocation post PR #38.**
The fetch driver script must be invoked as a module: `python -m scripts.data_pipelines.broad_market_fetch [start_idx] [--verbose]`. Direct script invocation (`python scripts/data_pipelines/broad_market_fetch.py`) fails with `ModuleNotFoundError: No module named 'scripts'` because PR #38 introduced `from scripts.data_pipelines._fetch_logging import ...` which needs the package context. Same applies to `sectoral_fetch.py` and `thematic_fetch.py` if they adopt the helper module. The new structured-log format is one line per ticker with status bucket (`OK` / `OK_NOOP` / `FAIL_EMPTY` / `FAIL_OVERLAP` / `FAIL_OTHER`), rolling summary every 50 tickers, and a `summary.json` written on completion (even via Ctrl-C / SIGTERM).

**8. Fetch plan regeneration recipe.**
`/tmp/fetch_plan.json` is the input to the broad fetch script. If it's missing or corrupted (e.g. overwritten by a sanity test), regenerate from current cache state:
```python
import json, sqlite3, yaml, glob, os
sectoral = {'nifty_bank','nifty_it', ...}  # see configs/data_pipelines/domains/nse_equities/ for the full set
thematic_prefixes = ('nifty_alpha','nifty_low_vol', ...)  # see post-PR-#40 set
all_tickers = set()
for path in glob.glob('configs/data_pipelines/domains/nse_equities/universe_*.yaml'):
    name = os.path.basename(path).replace('universe_','').replace('.yaml','')
    if name in sectoral or any(name.startswith(p) for p in thematic_prefixes):
        continue
    all_tickers.update(yaml.safe_load(open(path)).get('tickers', []))
con = sqlite3.connect('/tmp/exp_data/processed.db')
cur = con.cursor()
placeholders = ','.join('?' for _ in all_tickers)
cur.execute(f'SELECT ticker, row_count FROM nse_equities_meta WHERE ticker IN ({placeholders})', sorted(all_tickers))
cache = dict(cur.fetchall())
plan = {'cold': [], 'back_extend': [], 'already_deep': []}
for t in sorted(all_tickers):
    bucket = 'cold' if t not in cache else ('back_extend' if cache[t] < 2000 else 'already_deep')
    plan[bucket].append(t)
scratch = os.environ['SCRATCH_CACHE']  # per-user memory `scratch-cache-path`
json.dump(plan, open(f'{scratch}/fetch_plan.json', 'w'), indent=2)
os.symlink(f'{scratch}/fetch_plan.json', '/tmp/fetch_plan.json')
```
Persistent location (`${SCRATCH_CACHE}/`) survives reboots; `/tmp/` symlink keeps script compat.

**How to apply:**

- When registering a new NSE universe: start with the curl fallback (don't waste time on jugaad/nselib).
- Filter `DUMMY*` from constituent lists at registration time.
- Pre-flight `sqlite3 data/processed.db 'PRAGMA quick_check'` before any cache-dependent work. If it fails, set up a scratch dir and reroute the data symlink (use `${SCRATCH_CACHE}`, NOT raw `/tmp/`).
- For bulk universe seeds: sequential with per-ticker 120 s hard timeout.
- For cached-but-short tickers: `back_extend=True` on the gap-fill pass.
- For broad fetch: `python -m scripts.data_pipelines.broad_market_fetch` (module form, NOT direct script).

See `[[feedback-disk-wedge-pattern]]` for the FS wedge root cause, `[[feedback-agent-pkill-antipattern]]` for the concurrent-process lesson, `[[project-r-precision-methodology]]` for the gbdt cross-cell metric framework, and `CLAUDE.md` § Data and configs for the broader cache layout contract.
