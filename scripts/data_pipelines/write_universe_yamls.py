"""Materialize universe YAMLs from /tmp/universe_symbols.json.

Each YAML is a flat ticker list, no `compose:` directive. The header
documents derivation lineage where applicable.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "configs" / "data_pipelines" / "domains" / "nse_equities"
DATESTAMP = "2026-05-26"

with open("/tmp/universe_symbols.json") as f:
    parsed = json.load(f)

# Universe metadata: (filename_slug, universe_key, NSE_index_ticker,
#                     description, csv_url_slug, derivation_note)
UNIVERSES = [
    # Tier 1 — cold-fetched lists
    ("nifty_smallcap_250", "nifty_smallcap_250", "NIFTY:SMALLCAP250",
     "NIFTY Smallcap 250 — top 251-500 by mcap",
     "ind_niftysmallcap250list", None, "smallcap250"),
    ("nifty_microcap_250", "nifty_microcap_250", "NIFTY:MICROCAP250",
     "NIFTY Microcap 250 — top 501-750 by mcap",
     "ind_niftymicrocap250_list", None, "microcap250"),
    # Tier 2 — derived (still curl'd authoritative list to avoid rebalance edge cases)
    ("nifty_next_50", "nifty_next_50", "NIFTY:NEXT50",
     "NIFTY Next 50 — top 51-100 by mcap (Nifty 100 minus Nifty 50)",
     "ind_niftynext50list",
     "Equivalent to: nifty100 \\ nifty50.", "next50"),
    ("nifty100", "nifty100", "NIFTY:100",
     "NIFTY 100 — top 100 by mcap",
     "ind_nifty100list",
     "Equivalent to: nifty50 ∪ nifty_next_50.", "nifty100"),
    ("nifty200", "nifty200", "NIFTY:200",
     "NIFTY 200 — top 200 by mcap",
     "ind_nifty200list",
     "Equivalent to: nifty100 ∪ (next 100 large/midcap names).", "nifty200"),
    ("nifty500", "nifty500", "NIFTY:500",
     "NIFTY 500 — top 500 by mcap (covers ~96% of free-float mcap)",
     "ind_nifty500list",
     "Equivalent to: nifty100 ∪ nifty_midcap_150 ∪ nifty_smallcap_250.",
     "nifty500"),
    ("nifty_midcap_150", "nifty_midcap_150", "NIFTY:MIDCAP150",
     "NIFTY Midcap 150 — top 101-250 by mcap",
     "ind_niftymidcap150list", None, "midcap150"),
    ("nifty_midsmallcap_400", "nifty_midsmallcap_400", "NIFTY:MIDSMALLCAP400",
     "NIFTY MidSmallcap 400 — top 101-500 by mcap (midcap + smallcap union)",
     "ind_niftymidsmallcap400list",
     "Equivalent to: nifty_midcap_150 ∪ nifty_smallcap_250.",
     "midsmallcap400"),
    ("nifty_largemidcap_250", "nifty_largemidcap_250", "NIFTY:LARGEMIDCAP250",
     "NIFTY LargeMidcap 250 — top 250 by mcap (large + midcap union)",
     "ind_niftylargemidcap250list",
     "Equivalent to: nifty100 ∪ nifty_midcap_150.",
     "largemidcap250"),
    ("nifty_total_market", "nifty_total_market", "NIFTY:TOTALMARKET",
     "NIFTY Total Market — top 750 by mcap (broad market coverage)",
     "ind_niftytotalmarket_list",
     "Equivalent to: nifty500 ∪ nifty_microcap_250.",
     "totalmarket"),
    ("nifty_midcap_50", "nifty_midcap_50", "NIFTY:MIDCAP50",
     "NIFTY Midcap 50 — top 50 most liquid midcaps",
     "ind_niftymidcap50list",
     "Subset of nifty_midcap_150.",
     "midcap50"),
    ("nifty_midcap_100", "nifty_midcap_100", "NIFTY:MIDCAP100",
     "NIFTY Midcap 100 — top 100 by mcap within midcap segment",
     "ind_niftymidcap100list",
     "Subset of nifty_midcap_150 (first 100).",
     "midcap100"),
    ("nifty_smallcap_50", "nifty_smallcap_50", "NIFTY:SMALLCAP50",
     "NIFTY Smallcap 50 — top 50 most liquid smallcaps",
     "ind_niftysmallcap50list",
     "Subset of nifty_smallcap_250.",
     "smallcap50"),
    ("nifty_smallcap_100", "nifty_smallcap_100", "NIFTY:SMALLCAP100",
     "NIFTY Smallcap 100 — top 100 by mcap within smallcap segment",
     "ind_niftysmallcap100list",
     "Subset of nifty_smallcap_250 (first 100).",
     "smallcap100"),
]


def render(filename: str, universe_key: str, index_ticker: str,
           description: str, csv_slug: str, derivation: str | None,
           symbols: list[str]) -> str:
    lines = [
        f"# {description} (as of {DATESTAMP}).",
        f"# Source: https://archives.nseindia.com/content/indices/{csv_slug}.csv",
        f"# Refresh by re-running that fetch; NSE rebalances quarterly/semi-annually.",
        f"# Point-in-time historical membership is explicitly out of scope (open q.1).",
    ]
    if derivation:
        lines.append(f"# Derivation: {derivation}")
    lines.append("")
    lines.append(f"universe: {universe_key}")
    lines.append(f"listed_at: {DATESTAMP}")
    lines.append("indices:")
    lines.append(f'  - "{index_ticker}"')
    lines.append("tickers:")
    for sym in sorted(symbols):
        lines.append(f'  - "NSE:{sym}"')
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for entry in UNIVERSES:
        filename_slug, key, idx, desc, csv_slug, derivation, parsed_key = entry
        symbols = parsed[parsed_key]
        out_path = OUT_DIR / f"universe_{filename_slug}.yaml"
        out_path.write_text(render(filename_slug, key, idx, desc,
                                    csv_slug, derivation, symbols))
        print(f"wrote {out_path}  ({len(symbols)} tickers)")


if __name__ == "__main__":
    main()
