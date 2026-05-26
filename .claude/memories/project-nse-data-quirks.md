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

**Workaround** (until the host's FS is repaired / fsck'd):
```bash
# Pick a non-NTFS scratch path (tmpfs, ext4 home, etc.).
SCRATCH="/tmp/exp_data"
mkdir -p "$SCRATCH" && cp data/processed.db "$SCRATCH/processed.db"
cp data/NASDAQ100.csv "$SCRATCH/" 2>/dev/null  # if present
mkdir -p "$SCRATCH/raw" "$SCRATCH/processed"; touch "$SCRATCH/.gitkeep"
# Then in each worktree:
rm -rf data && ln -s "$SCRATCH" data
```
The scratch copy is writable, so new fetches (from universe self-service) write to it without corruption. Note: tmpfs is RAM-backed; data persists only until reboot. For long runs, pick a non-tmpfs scratch (e.g. `~/exp_data`).

**4. Per-ticker fetch wall time.**
With `nselib` blocked by single-day rejection errors, `data_pipelines.fetch()` on a new NSE ticker takes ~30–80 s per ticker (retry storms on holiday-adjacent dates + delisting placeholders + Yahoo "possibly delisted" warnings). For a 150-stock universe, that's ~2–3 hours sequential. Use per-ticker 120 s hard timeout (mirror `scripts/seed_nifty50_deep.sh` from PR #6).

**5. Shallow-cache silent drop trap.**
Many tickers in the cache from earlier `data-seed-nifty-total` work were seeded with `start=2019-01-01` (~1500–1588 rows). The default `min_rows=1600` gate in `gbdt.data.load_panel` silently drops them. For wide-universe experiments where many tickers fall in this gap (Exp 2 dropped 53/100 silently), the universe self-service phase should also call `data_pipelines.fetch(..., back_extend=True, start='2015-01-01')` on cached-but-short tickers to push them above the bar.

**How to apply:**

- When registering a new NSE universe: start with the curl fallback (don't waste time on jugaad/nselib).
- Filter `DUMMY*` from constituent lists at registration time.
- Pre-flight `sqlite3 data/processed.db 'PRAGMA quick_check'` before any cache-dependent work. If it fails, set up a scratch dir and reroute the data symlink.
- For bulk universe seeds: sequential with per-ticker 120 s hard timeout.
- For cached-but-short tickers: `back_extend=True` on the gap-fill pass.

See `[[feedback-disk-wedge-pattern]]` for the FS wedge root cause and `CLAUDE.md` § Data and configs for the broader cache layout contract.
