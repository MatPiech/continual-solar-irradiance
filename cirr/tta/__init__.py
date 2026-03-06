"""Test-Time Adaptation (TTA) methods for solar irradiance forecasting."""

from cirr.tta.dual_phase import DualPhaseAdapter
from cirr.tta.memo_reg import MEMOReg
from cirr.tta.note_reg import NOTEReg

__all__ = ["DualPhaseAdapter", "MEMOReg", "NOTEReg"]
