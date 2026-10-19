"""Smart-persistence forecast: clear-sky-index persistence with naive fallback.

Single implementation shared by ``scripts/smart_persistence_model.py`` and the
federated benchmark, so the Forecast Skill denominator cannot drift between them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from cirr.physics.clear_sky import resolve_clear_sky_value

KC_CLAMP = (0.0, 2.0)


def predict_periods(
    periods: list[dict[str, Any]],
    clear_sky_lookup: dict[str, float],
    horizon_minutes: int = 15,
    period_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Predict every period by clear-sky-index persistence.

    ``clear_sky_lookup`` is keyed ``YYYYMMDD_HHMMSS``; predictions and targets
    come back in the units of ``clear_sky_lookup`` when *period_scale* converts
    the period irradiances into those units (e.g. ``1/1000`` for W/m² periods
    against a kW/m² lookup).  Falls back to naive persistence whenever either
    clear-sky value is unavailable, matching the published baseline.
    """
    preds, tgts = [], []
    naive = 0
    for period in periods:
        last = period["history"][-1]
        stamp = last["image_name"].rsplit(".", 1)[0]
        last_irr = float(last["irradiance"]) * period_scale

        ghi_now = resolve_clear_sky_value(clear_sky_lookup, stamp)
        ghi_target = resolve_clear_sky_value(clear_sky_lookup, _shift(stamp, horizon_minutes))

        if ghi_now is not None and ghi_target is not None and ghi_now > 0:
            kc = min(max(last_irr / ghi_now, KC_CLAMP[0]), KC_CLAMP[1])
            preds.append(kc * ghi_target)
        else:
            preds.append(last_irr)
            naive += 1
        tgts.append(float(period["target_irradiance"]) * period_scale)

    return np.array(preds), np.array(tgts), naive


def _shift(stamp: str, minutes: int) -> str:
    return (datetime.strptime(stamp, "%Y%m%d_%H%M%S") + timedelta(minutes=minutes)).strftime("%Y%m%d_%H%M%S")


def _demo() -> None:
    lookup = {"20160101_120000": 0.8, "20160101_121500": 0.6}
    periods = [{"history": [{"image_name": "20160101_120000.jpg", "irradiance": 400.0}], "target_irradiance": 300.0}]
    preds, tgts, naive = predict_periods(periods, lookup, 15, period_scale=1 / 1000)
    assert naive == 0
    assert abs(preds[0] - 0.5 * 0.6) < 1e-9, preds  # kc = 0.4/0.8 = 0.5
    assert abs(tgts[0] - 0.3) < 1e-9
    # missing target clear-sky -> naive persistence
    preds, _, naive = predict_periods(periods, {"20160101_120000": 0.8}, 15, period_scale=1 / 1000)
    assert naive == 1 and abs(preds[0] - 0.4) < 1e-9
    print("smart_persistence self-check ok")


if __name__ == "__main__":
    _demo()
