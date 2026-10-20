"""Interactively select sun-mask camera settings for a site.

Shows clear-sky frames (one per month first, so both DST regimes are visible)
with the sun mask drawn from the current settings, and lets you tune them with
sliders.  Starts from the fitted values in cirr/physics/sun.py (fit_sun_camera.py).

    python scripts/select_sun_mask.py --site poznan
    python scripts/select_sun_mask.py --site karpicko --headless   # just write the overlay grid
    python scripts/select_sun_mask.py --site poznan --headless --set offset=299 --set clock_std=65

On a remote machine without a display use --headless: open the written JPG (e.g. in
VSCode), adjust with --set, repeat; add --save to write the settings file.

Keys:  s = save settings   n = new frames   q / Esc = quit
Saved to results/sun_camera/<site>_manual_params.txt as a SITES entry to paste into
cirr/physics/sun.py.  Needs a display (OpenCV Qt window); --headless does not.
"""

from pathlib import Path

import click
import cv2
from fit_sun_camera import SIZE, clear_candidates, detect_sun, load_crop
import numpy as np

from cirr.physics.sun import SITES, project, solar_angles

TILE = 256
COLS = 4
WIN = "sun mask"

# slider name -> (max position, position -> value, value -> position)
SLIDERS = {
    "offset x0.1deg": (3600, lambda p: p / 10, lambda v: int(round((v % 360) * 10))),
    "focal x1e-3": (1500, lambda p: p / 1000, lambda v: int(round(v * 1000))),
    "cx x1e-3": (1000, lambda p: p / 1000, lambda v: int(round(v * 1000))),
    "cy x1e-3": (1000, lambda p: p / 1000, lambda v: int(round(v * 1000))),
    "k3 (x1e-3)-1": (2000, lambda p: (p - 1000) / 1000, lambda v: int(round(v * 1000 + 1000))),
    "mirror 0:+1 1:-1": (1, lambda p: -1 if p else 1, lambda v: int(v == -1)),
    "clock DST min+180": (360, lambda p: p - 180, lambda v: int(round(v + 180))),
    "clock STD min+180": (360, lambda p: p - 180, lambda v: int(round(v + 180))),
}
KEYS = ("offset", "focal", "cx", "cy", "k3", "mirror", "clock_dst", "clock_std")


def pick_frames(site: str, n: int, seed: int) -> list[str]:
    names = clear_candidates(Path(f"data/{site}_prepared"), 10 * n, seed)
    by_month: dict[str, list[str]] = {}
    for name in names:
        by_month.setdefault(name[:6], []).append(name)
    picked = [v[0] for _, v in sorted(by_month.items())][:n]
    rest = [x for x in names if x not in picked]
    return picked + rest[: n - len(picked)]


def settings_from_cfg(cfg: dict) -> dict:
    return {
        **{k: cfg[k] for k in ("offset", "focal", "cx", "cy", "k3", "mirror")},
        "clock_dst": cfg["clock"][0],
        "clock_std": cfg["clock"][1],
    }


class Viewer:
    def __init__(self, site: str, names: list[str]):
        self.site = site
        self.names = names
        self.images = [load_crop(Path(f"data/{site}_prepared/images") / n) for n in names]
        self.detected = [detect_sun(im) for im in self.images]
        self._angles: dict[tuple[float, float], dict] = {}

    def angles(self, clock: tuple[float, float]) -> dict:
        if clock not in self._angles:
            self._angles[clock] = solar_angles(self.site, self.names, {**SITES[self.site], "clock": clock})
        return self._angles[clock]

    def render(self, s: dict) -> np.ndarray:
        ang = self.angles((float(s["clock_dst"]), float(s["clock_std"])))
        tiles = []
        for name, im, det in zip(self.names, self.images, self.detected, strict=True):
            im = im.copy()
            zen, az = ang[name]
            if zen < 90:
                u, v = project(zen, az, s["offset"], s["focal"], s["mirror"], s["cx"], s["cy"], s["k3"])
                cv2.circle(im, (int(u * SIZE), int(v * SIZE)), SIZE // 16, (255, 0, 0), 3)
            if det is not None:
                cv2.circle(im, (int(det[0] * SIZE), int(det[1] * SIZE)), 6, (0, 255, 0), 2)
            cv2.putText(im, f"{name[:13]} z{zen:.0f}", (6, 24), 0, 0.8, (255, 255, 0), 2)
            tiles.append(cv2.resize(im, (TILE, TILE)))
        while len(tiles) % COLS:
            tiles.append(np.zeros_like(tiles[0]))
        return np.vstack([np.hstack(tiles[i : i + COLS]) for i in range(0, len(tiles), COLS)])


def save(site: str, s: dict, out: Path) -> None:
    entry = (
        f'"offset": {s["offset"]:.2f}, "focal": {s["focal"]:.4f}, "mirror": {s["mirror"]}, '
        f'"cx": {s["cx"]:.4f}, "cy": {s["cy"]:.4f},\n'
        f'"k3": {s["k3"]:.4f}, "clock": ({float(s["clock_dst"]):.1f}, {float(s["clock_std"]):.1f}),'
    )
    path = out / f"{site}_manual_params.txt"
    path.write_text(entry + "\n")
    print(f"saved {path}:\n{entry}")


@click.command()
@click.option("--site", type=click.Choice(sorted(SITES)), required=True)
@click.option("--n", type=int, default=12, help="frames shown")
@click.option("--seed", type=int, default=0)
@click.option("--headless", is_flag=True, help="write the overlay grid for the current settings and exit")
@click.option(
    "--set", "overrides", multiple=True, metavar="KEY=VALUE", help=f"override a setting, repeatable; keys: {', '.join(KEYS)}"
)
@click.option("--save", "save_settings", is_flag=True, help="with --headless: also write the settings file")
@click.option("--out", type=click.Path(path_type=Path), default=Path("results/sun_camera"))
def main(site, n, seed, headless, overrides, save_settings, out):
    out.mkdir(parents=True, exist_ok=True)
    viewer = Viewer(site, pick_frames(site, n, seed))
    s = settings_from_cfg(SITES[site])
    for item in overrides:
        key, _, value = item.partition("=")
        if key not in KEYS:
            raise click.BadParameter(f"unknown key {key!r}; use one of {', '.join(KEYS)}", param_hint="--set")
        s[key] = int(value) if key == "mirror" else float(value)
    print("red = mask from current settings, green = detected sun")

    if headless:
        path = out / f"{site}_manual_overlay.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(viewer.render(s), cv2.COLOR_RGB2BGR))
        print(f"wrote {path}")
        print("settings:", {k: s[k] for k in KEYS})
        if save_settings:
            save(site, s, out)
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for (label, (top, _, to_pos)), key in zip(SLIDERS.items(), KEYS, strict=True):
        cv2.createTrackbar(label, WIN, to_pos(s[key]), top, lambda _: None)
    while True:
        for (label, (_, to_val, _)), key in zip(SLIDERS.items(), KEYS, strict=True):
            s[key] = to_val(cv2.getTrackbarPos(label, WIN))
        cv2.imshow(WIN, cv2.cvtColor(viewer.render(s), cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(50) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            save(site, s, out)
        if k == ord("n"):
            seed += 1
            viewer = Viewer(site, pick_frames(site, n, seed))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
