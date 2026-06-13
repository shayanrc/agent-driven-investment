"""isotonic — placeholder for migrated gbdt isotonic calibration.

**V1 ships a placeholder only.** gbdt's existing ``conditional_isotonic``
calibration lives inside ``src/gbdt/`` and is NOT migrated in this branch
(plan D3; calibration/goal.md "What this module is *not*"). Migrating it
here — to avoid two-location calibration state — is a deliberate future
follow-up, not v1 scope.

This module exists so the package layout is complete and the migration has
an obvious home. It intentionally contains no implementation.
"""

from __future__ import annotations

__all__: list[str] = []
