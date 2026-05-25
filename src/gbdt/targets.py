"""Binary target construction for the 18-cell direction × threshold × horizon lattice.

The lattice is hard-coded in v1 (not user-configurable — see V1_PLAN anti-goal #2).
Implementation lands in V1_PLAN Stage 3.
"""

# v1 target lattice: 6 directions × 3 horizons = 18 binary classifiers per asset.
# Each cell is the question "did close move by ``threshold`` in direction ``direction``
# at any point within ``horizon`` trading periods after origin?"
DIRECTIONS = ("up", "down")
THRESHOLDS_PCT = (10, 20, 50)   # absolute percent moves
HORIZONS_DAYS = (10, 20, 50)    # forward trading-day windows

# Target name convention: f"{direction}_{threshold}_h{horizon}", e.g. "up_10_h20".
