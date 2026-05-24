"""GBDT classifier wrapper with calibration.

Library choice + wrapper interface finalized in V1_PLAN Stage 4 (default lean:
lightgbm). Calibration method finalized in V1_PLAN Stage 5 (default lean:
sklearn ``IsotonicRegression``). Persisted artifacts contain model binary +
calibration map + feature schema + fit metadata.
"""
