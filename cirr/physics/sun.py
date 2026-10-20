"""Sun position in the sky-image plane, per site.

Projection follows the IEP paper (Piechocki & Kraft): stereographic-like
``r = 2 f tan(zenith / 2)`` with an azimuth offset for camera orientation.
Coordinates are normalised to the *square centre crop* used by the datamodule
(``u, v`` in ``[0, 1]``), so the same parameters hold at any resize.  ``cx, cy``
allow for an optical centre that is not the crop centre, ``mirror`` for cameras
whose image is left-right flipped relative to the Folsom convention, and ``k3``
adds ``k3 * zenith_rad**3`` to the radius for lenses that depart from the
paper's projection near the horizon (0 reproduces the paper exactly).

``clock`` is a correction in minutes added to the image-name time while daylight
saving time is / is not in effect, for cameras whose clock does not follow the
local DST switch (found at the Polish sites, see fit_sun_camera.py).

Camera parameters are fitted by ``scripts/fit_sun_camera.py``; see that script's
report for residuals.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pvlib

# lat/lon/tz as in scripts/bash_scripts/generate_clear_sky_all.sh.
# offset [deg], focal [crop-radius units], mirror (+1/-1), cx/cy [normalised],
# k3 [radius / rad^3], clock [min] as (during DST, during standard time).
SITES: dict[str, dict] = {
    # Fitted on the 1520 crop (results/sun_camera/folsom_params.txt): 88 % inliers,
    # median 0.8 px at 128.  Paper values were 165 deg / 0.48 on the full 1536 frame.
    "folsom": {
        "lat": 38.642,
        "lon": -121.148,
        "tz": "US/Pacific",
        "offset": 163.72,
        "focal": 0.5044,
        "mirror": 1,
        "cx": 0.4937,
        "cy": 0.4822,
        "k3": 0.0,
        "clock": (0.0, 0.0),
    },
    # Polish sites (results/sun_camera/*_params.txt): the lens needs k3 and the image
    # clock is ~70 min off in one DST regime; only the difference between regimes is
    # identifiable, a constant clock bias is absorbed by ``offset``.
    # Poznan: 81.5 % inliers, median 0.85 px at 128.
    "poznan": {
        "lat": 52.4031,
        "lon": 16.9539,
        "tz": "Europe/Warsaw",
        "offset": 297.39,
        "focal": 0.6596,
        "mirror": 1,
        "cx": 0.5397,
        "cy": 0.4128,
        "k3": -0.0824,
        "clock": (0.0, 68.3),
    },
    # Karpicko: 80.0 % inliers, median 0.79 px at 128.
    "karpicko": {
        "lat": 52.1294,
        "lon": 16.1231,
        "tz": "Europe/Warsaw",
        "offset": 197.60,
        "focal": 0.6251,
        "mirror": 1,
        "cx": 0.4417,
        "cy": 0.4771,
        "k3": -0.0574,
        "clock": (-70.2, 0.0),
    },
}


def site_name(data_root: Path) -> str:
    return Path(data_root).name.split("_")[0].lower()


def _localize(naive: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    # The DST fall-back hour is ambiguous; it is at night at every site, where the
    # mask is empty anyway, so it is resolved as standard time.
    return naive.tz_localize(tz, ambiguous=np.zeros(len(naive), dtype=bool), nonexistent="shift_forward")


def solar_angles(site: str, image_names: list[str], cfg: dict | None = None) -> dict[str, tuple[float, float]]:
    """Map image name -> (zenith, azimuth) in degrees.  Image names are naive local time."""
    cfg = cfg or SITES[site]
    names = sorted(set(image_names))
    naive = pd.DatetimeIndex(pd.to_datetime([n[:15] for n in names], format="%Y%m%d_%H%M%S"))
    dst_min, std_min = cfg.get("clock", (0.0, 0.0))
    if dst_min or std_min:
        is_dst = np.array([t.dst().total_seconds() != 0 for t in _localize(naive, cfg["tz"])])
        naive = naive + pd.to_timedelta(np.where(is_dst, dst_min, std_min), unit="min")
    local = _localize(naive, cfg["tz"])
    pos = pvlib.solarposition.get_solarposition(local, cfg["lat"], cfg["lon"])
    return dict(zip(names, zip(pos["apparent_zenith"].to_numpy(), pos["azimuth"].to_numpy(), strict=True), strict=True))


def project(zenith, azimuth, offset, focal, mirror=1, cx=0.5, cy=0.5, k3=0.0):
    """Normalised (u, v) of the sun in the square crop.  Works on scalars or arrays."""
    theta = np.deg2rad(zenith)
    r = 2 * focal * np.tan(theta / 2) + k3 * theta**3
    a = np.deg2rad(np.asarray(azimuth) - offset)
    return cx + 0.5 * mirror * r * np.sin(a), cy + 0.5 * r * np.cos(a)


def sun_mask(shape: tuple[int, int], crop: int, zenith: float, azimuth: float, cfg: dict) -> np.ndarray:
    """Binary uint8 mask at the *raw* image shape, disc radius crop/16 (paper: size//16).

    Drawn at raw resolution so it can go through the same albumentations call as
    the image (centre crop, resize, augmentation) as a ``mask`` target.
    """
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if zenith >= 90.0:
        return mask
    u, v = project(zenith, azimuth, cfg["offset"], cfg["focal"], cfg["mirror"], cfg["cx"], cfg["cy"], cfg.get("k3", 0.0))
    x = int(round((w - crop) / 2 + u * crop))
    y = int(round((h - crop) / 2 + v * crop))
    cv2.circle(mask, (x, y), crop // 16, 1, -1)
    return mask


def _demo() -> None:
    # Folsom paper params: sun due south at noon lands below-left of centre, inside the disc.
    u, v = project(30.0, 180.0, 165.0, 0.48)
    assert 0 < u < 1 and 0 < v < 1
    assert sun_mask((100, 120), 100, 95.0, 180.0, SITES["folsom"]).sum() == 0
    assert sun_mask((100, 120), 100, 30.0, 180.0, SITES["folsom"]).sum() > 0
    angles = solar_angles("folsom", ["20150621_131000.jpg"])
    zen, _ = angles["20150621_131000.jpg"]
    assert 14 < zen < 17, zen  # solar noon (13:10 PDT) near solstice at 38.6 N
    # clock correction applies only in its DST regime (Karpicko: -70.2 min in summer)
    summer, winter = "20250615_120000.jpg", "20250115_120000.jpg"
    plain = solar_angles("karpicko", [summer, winter], {**SITES["karpicko"], "clock": (0.0, 0.0)})
    fixed = solar_angles("karpicko", [summer, winter])
    assert fixed[winter] == plain[winter] and fixed[summer] != plain[summer]
    print("sun self-check ok")


if __name__ == "__main__":
    _demo()
