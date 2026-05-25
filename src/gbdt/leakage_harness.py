"""Synthetic-data harness for detecting causal-feature violations.

Generates an OHLCV-shaped DataFrame with a known "leak signal" planted at a
future row. Any feature function that incorporates the leak achieves perfect
AUC on the synthetic data; any causally-correct feature achieves chance AUC.
Gates every new feature in CI. Implementation lands in V1_PLAN Stage 1.
"""
