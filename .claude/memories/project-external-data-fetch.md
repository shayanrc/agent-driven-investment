---
name: project-external-data-fetch
description: How to fetch external/network data on this host — sandbox-off + short/inline commands + SSL_CERT_FILE; FRED egress is flaky here, use the primary sources (Treasury/NY Fed/Yahoo) instead.
metadata:
  type: project
---

Fetching external HTTP data on the project host (the `data_pipelines`/gbdt machine) has
three non-obvious traps that cost hours when re-derived (discovered 2026-06-21, migrated
from per-user memory 2026-07-15):

1. **Network needs the sandbox disabled, AND the command must stay inline.** The Bash
   sandbox blocks all network. `dangerouslyDisableSandbox: true` works ONLY while the
   command runs inline — if it runs long enough to be auto-backgrounded, the backgrounded
   process is **re-sandboxed and loses network** (every request then times out). So keep
   each network command SHORT (a few seconds), fetch in small batches across multiple
   calls, save raw to `/tmp`, and do parsing/seeding in a separate no-network step.
2. **Python `ssl` needs `SSL_CERT_FILE`.** The `.venv` Python doesn't find the system CA
   bundle by default → `CERTIFICATE_VERIFY_FAILED`. Export
   `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` (or `certifi.where()`).
   `ssl.create_default_context()` (urllib + the adapters) honors it. Also send
   `Accept: */*` + `Connection: close` headers — some CDN front-ends hang an HTTP/1.1 GET
   without them (this is why the FRED adapter has those headers).
3. **FRED egress (`stlouisfed.org`) is unreliable on this host** — it has timed out for
   hours straight (read-timeout after TLS handshake) while GitHub/PyPI/Yahoo/Treasury/
   NY Fed all worked. When FRED is down, the same macro series are available from **the
   primary sources FRED redistributes** (real values, not proxies) — used to seed 8/9 of
   the F17 macro panel (`docs/gbdt/_259_macro_real_fred.md`):
   - **U.S. Treasury** daily CSV (`home.treasury.gov/.../daily-treasury-rates.csv/<YEAR>/all?type=daily_treasury_yield_curve` and `...real_yield_curve`): DGS10/DGS3MO/DGS2 (→ T10Y2Y = 10Y−2Y), DFII10 (TIPS 10Y), T10YIE = nom10 − real10. Per-year only (`all/all` is 403).
   - **NY Fed** EFFR (`markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=&endDate=`): DFF — full history in one call.
   - **Yahoo** chart API (`query1.finance.yahoo.com/v8/finance/chart/<sym>?period1=&period2=&interval=1d`): `^VIX`→VIXCLS, `DX-Y.NYB`→DTWEXBGS, `^TNX`/10≈DGS10, `^IRX`≈DGS3MO. NB: `range=max` silently downsamples to monthly — use explicit `period1/period2` for daily.
   - **No free non-FRED source for `BAMLH0A0HYM2`** (ICE BofA HY credit OAS) — it stays absent. DBnomics does NOT carry FRED; Stooq's CSV is apikey-gated (returns an HTML gate page).

**How to apply:** for any external fetch, set `SSL_CERT_FILE`, keep network commands
short + inline (sandbox off), and for FRED macro data go straight to Treasury/NY Fed/
Yahoo rather than waiting on `stlouisfed.org`. Related: [[project-venv-stale-shebang]],
[[project-gbdt-macro-features-f17]].
