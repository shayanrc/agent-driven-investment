"""Causal feature engineering.

Every feature at row ``t`` uses only data from rows strictly before ``t``.
Implementation + final feature set finalized in V1_PLAN Stage 2; the
candidate list and lookback windows are documented there.
"""
