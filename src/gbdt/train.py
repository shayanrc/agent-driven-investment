"""Walk-forward training driver.

For each fold and each of the 18 targets: fit GBDT with early stopping on a
validation slice, fit calibration on the same validation slice, score the
test slice, persist the per-(fold, target) artifact. Implementation lands in
V1_PLAN Stage 6.
"""
