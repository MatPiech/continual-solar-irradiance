"""Fit sun-mask camera parameters (offset, focal, cx, cy, mirror) for a site.

Detects the sun as the brightest blob in clear-sky frames and fits the IEP
projection (cirr/physics/sun.py) by robust least squares against pvlib's
solar position.  Writes an overlay montage for visual verification.

    python scripts/fit_sun_camera.py --site poznan --out results/sun_camera
"""

from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import least_squares

from cirr.physics.sun import SITES, project, solar_angles

SIZE = 512  # detection resolution
INLIER = 0.02  # normalised radius (~10 px at 512) counted as an inlier


def load_crop(path: Path) -> np.ndarray:
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]
    c = min(h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    return cv2.resize(im[y0 : y0 + c, x0 : x0 + c], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def detect_sun(im: np.ndarray) -> tuple[float, float] | None:
    """Centroid of the largest saturated blob, only when that blob is compact.

    Overcast or hazy frames that slip through the CSI filter saturate large areas;
    rejecting them (and tiny glints) keeps the fit on frames with a visible disc.
    """
    sat = (im.min(axis=2) >= 250).astype(np.uint8)
    if sat.mean() > 0.02:
        return None
    n, _, stats, cent = cv2.connectedComponentsWithStats(sat)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if not 30 <= stats[k, cv2.CC_STAT_AREA] <= 3000:
        return None
    return cent[k][0] / SIZE, cent[k][1] / SIZE


def clear_candidates(root: Path, site: str, n: int, seed: int) -> list[str]:
    frame = pd.read_csv(root / "cleaned_irradiance.csv")
    col = "ghi" if "ghi" in frame.columns else "irradiance"
    cs = pd.read_csv(next(root.glob("*clear_sky.csv")))
    cs = dict(zip(pd.to_datetime(cs["datetime"]).dt.strftime("%Y%m%d_%H%M%S"), cs["ghi_clear"], strict=False))
    frame["cs"] = frame["image_name"].str[:15].map(cs)
    frame = frame[(frame["cs"] > 200) & (frame[col] / frame["cs"]).between(0.9, 1.1)]
    frame = frame[frame["image_name"].map(lambda n: (root / "images" / n).exists())]
    # spread over days: at most 3 frames a day
    frame = frame.groupby(frame["image_name"].str[:8], group_keys=False).apply(
        lambda g: g.sample(min(len(g), 3), random_state=seed)
    )
    return frame.sample(min(n, len(frame)), random_state=seed)["image_name"].tolist()


PARAMS = ("offset", "focal", "cx", "cy", "k3")
LOWER, UPPER = [-np.inf, 0.2, 0.3, 0.3, -1.0], [np.inf, 1.5, 0.7, 0.7, 1.0]


def _err(za, uv, p, mirror):
    u, v = project(za[:, 0], za[:, 1], p[0], p[1], mirror, p[2], p[3], p[4])
    return np.hypot(u - uv[:, 0], v - uv[:, 1])


def _lsq(za, uv, x0, mirror):
    def resid(p):
        u, v = project(za[:, 0], za[:, 1], p[0], p[1], mirror, p[2], p[3], p[4])
        return np.concatenate([u - uv[:, 0], v - uv[:, 1]])

    return least_squares(resid, np.clip(x0, np.array(LOWER) + 1e-6, np.array(UPPER) - 1e-6), bounds=(LOWER, UPPER)).x


def ransac(za, uv, rng, iters=200):
    """Geometry (offset, focal, cx, cy, k3, mirror) robust to mis-detected frames."""
    best_n, best = -1, None
    for _ in range(iters):
        idx = rng.choice(len(uv), 5, replace=False)
        for mirror in (1, -1):
            for off0 in range(0, 360, 45):
                p = _lsq(za[idx], uv[idx], [off0, 0.5, 0.5, 0.5, 0.0], mirror)
                n = int((_err(za, uv, p, mirror) < INLIER).sum())
                if n > best_n:
                    best_n, best = n, (p, mirror)
    p, mirror = best
    inl = _err(za, uv, p, mirror) < INLIER
    p = _lsq(za[inl], uv[inl], p, mirror)
    return p, mirror


def angles_at(site, names, clock):
    a = solar_angles(site, list(names), {**SITES[site], "clock": clock})
    return np.array([a[n] for n in names])


def fit(site, det, rng):
    """Shared geometry, plus a clock correction for whichever DST regime disagrees.

    The regime with more inliers under the nominal clock is the reference (its
    correction is 0, so any constant clock bias there is absorbed by the azimuth
    offset); the other regime's correction is scanned in 5-min steps and then
    refined jointly with the geometry.
    """
    names = det["name"].to_numpy()
    uv = det[["u", "v"]].to_numpy()
    is_dst = np.array([t.dst().total_seconds() != 0 for t in pd.DatetimeIndex(
        pd.to_datetime([n[:15] for n in names], format="%Y%m%d_%H%M%S")).tz_localize(
        SITES[site]["tz"], ambiguous=False, nonexistent="shift_forward")])
    za0 = angles_at(site, names, (0.0, 0.0))

    groups = {g: m for g, m in (("dst", is_dst), ("std", ~is_dst)) if m.sum() >= 10}
    fits = {g: ransac(za0[m], uv[m], rng) for g, m in groups.items()}
    counts = {g: int((_err(za0[m], uv[m], *fits[g]) < INLIER).sum()) for g, m in groups.items()}
    ref = max(counts, key=counts.get)
    p, mirror = fits[ref]
    clock = {"dst": 0.0, "std": 0.0}
    for g, m in groups.items():
        if g == ref:
            continue
        scan = {dt: int((_err(angles_at(site, names[m], (dt, dt)), uv[m], p, mirror) < INLIER).sum())
                for dt in range(-150, 151, 5)}
        clock[g] = float(max(scan, key=scan.get))

    # joint refinement on inliers of both regimes
    za = angles_at(site, names, (clock["dst"], clock["std"]))
    inl = _err(za, uv, p, mirror) < INLIER
    free = [g for g in groups if g != ref]

    def resid(x):
        c = dict(clock)
        for g, dt in zip(free, x[5:], strict=True):
            c[g] = dt
        z = angles_at(site, names[inl], (c["dst"], c["std"]))
        u, v = project(z[:, 0], z[:, 1], x[0], x[1], mirror, x[2], x[3], x[4])
        return np.concatenate([u - uv[inl, 0], v - uv[inl, 1]])

    x = least_squares(resid, [*p, *[clock[g] for g in free]], diff_step=1e-3,
                      bounds=(LOWER + [-180.0] * len(free), UPPER + [180.0] * len(free))).x
    for g, dt in zip(free, x[5:], strict=True):
        clock[g] = round(float(dt), 1)
    params = dict(zip(PARAMS, map(float, x[:5]), strict=True))
    params["offset"] %= 360
    return {**params, "mirror": mirror, "clock": (clock["dst"], clock["std"])}


@click.command()
@click.option("--site", type=click.Choice(sorted(SITES)), required=True)
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
@click.option("--n", type=int, default=400)
@click.option("--seed", type=int, default=0)
@click.option("--out", type=click.Path(path_type=Path), default=Path("results/sun_camera"))
def main(site, data_root, n, seed, out):
    root = data_root or Path(f"data/{site}_prepared")
    out.mkdir(parents=True, exist_ok=True)
    names = clear_candidates(root, site, n, seed)
    angles = solar_angles(site, names)

    rows = []
    for name in names:
        uv = detect_sun(load_crop(root / "images" / name))
        if uv is not None:
            rows.append((name, *angles[name], *uv))
    det = pd.DataFrame(rows, columns=["name", "zenith", "azimuth", "u", "v"])
    print(f"{site}: {len(names)} clear candidates, sun detected in {len(det)}")

    params = fit(site, det, np.random.default_rng(seed))
    cfg = {**SITES[site], **params}
    za = angles_at(site, det["name"].to_numpy(), params["clock"])
    det["zenith"], det["azimuth"] = za[:, 0], za[:, 1]
    err = _err(za, det[["u", "v"]].to_numpy(),
               [cfg["offset"], cfg["focal"], cfg["cx"], cfg["cy"], cfg["k3"]], cfg["mirror"])
    det["err_px512"] = err * SIZE
    inl = err < INLIER
    by_month = det.assign(inlier=inl).groupby(det["name"].str[:6])["inlier"].agg(["size", "mean"]).round(2)
    summary = (f"params: {params}\n"
               f"inliers (<{INLIER * SIZE:.0f}px@512): {inl.mean():.1%}   median err {np.median(err[inl]) * SIZE:.1f}px@512 "
               f"(~{np.median(err[inl]) * 128:.2f}px at 128; mask radius 8px)\n"
               f"per month (n, inlier rate):\n{by_month.to_string()}\n")
    print(summary)
    det.to_csv(out / f"{site}_detections.csv", index=False)
    (out / f"{site}_params.txt").write_text(summary)
    params = {k: params[k] for k in ("offset", "focal", "mirror", "cx", "cy", "k3")}

    tiles = []
    # one frame per month so both DST regimes show up
    for _, row in det.groupby(det["name"].str[:6]).head(1).head(12).iterrows():
        im = load_crop(root / "images" / row["name"]).copy()
        pu, pv = project(row["zenith"], row["azimuth"], **params)
        cv2.circle(im, (int(row["u"] * SIZE), int(row["v"] * SIZE)), 6, (0, 255, 0), 2)
        cv2.circle(im, (int(pu * SIZE), int(pv * SIZE)), SIZE // 16, (255, 0, 0), 2)
        cv2.putText(im, row["name"][:13], (5, 20), 0, 0.6, (255, 255, 0), 2)
        tiles.append(cv2.resize(im, (256, 256)))
    grid = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles) - len(tiles) % 4, 4)])
    Image.fromarray(grid).save(out / f"{site}_overlay.jpg")
    print(f"overlay: {out / f'{site}_overlay.jpg'}  (green = detected, red = mask)")


if __name__ == "__main__":
    main()
