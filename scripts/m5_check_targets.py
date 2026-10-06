"""M5 check: target hits agree with the targets drawn into the video.

For every practice shot with a landing, each applicable target is projected into the image
through the calibrated camera (for the hitter's end), and the bounce pixel is tested against
that outline with OpenCV. This shares nothing with ``analysis.practice`` but the target
definitions: it checks the hitter's-frame mirroring, rectangle/circle containment, absolute
targets and stroke filters against plain image geometry.

usage: uv run python scripts/m5_check_targets.py <session_id> [...] [--sets builtin]

``--sets builtin`` evaluates a fixed battery of target sets (relative and absolute, rects and
circles, a serve-only filter) instead of the session's own targets; the session's targets
are restored afterwards.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from swingvision import services
from swingvision.court import calibration as calib
from swingvision.settings import load_settings
from swingvision.storage import tables
from swingvision.storage.schemas import Target

#: Margin (px) around an outline treated as "on the line": a disagreement within it is a
#: rounding case, not an error. Reported separately.
EDGE_PX = 1.0

BUILTIN: dict[str, list[Target]] = {
    "serve boxes (relative)": [
        Target(id="deuce", name="Deuce box", shape="rect", x0=-4.115, y0=0, x1=0, y1=6.4),
        Target(id="ad", name="Ad box", shape="rect", x0=0, y0=0, x1=4.115, y1=6.4),
    ],
    "T and wide corners (relative, serves only)": [
        Target(id="t_d", name="Deuce T", shape="rect", x0=-1.2, y0=4.4, x1=0, y1=6.4,
               strokes=["serve"]),
        Target(id="w_d", name="Deuce wide", shape="rect", x0=-4.115, y0=4.4, x1=-2.9, y1=6.4,
               strokes=["serve"]),
        Target(id="w_a", name="Ad wide", shape="rect", x0=2.9, y0=4.4, x1=4.115, y1=6.4,
               strokes=["serve"]),
    ],
    "circles (relative)": [
        Target(id="c1", name="Body", shape="circle", cx=-2.0, cy=5.0, r=1.0),
        Target(id="c2", name="Deep", shape="circle", cx=0.0, cy=9.5, r=1.5),
    ],
    "absolute halves": [
        Target(id="near_l", name="Near left", shape="rect", frame="absolute",
               x0=-4.115, y0=-11.885, x1=0, y1=0),
        Target(id="far_l", name="Far left", shape="rect", frame="absolute",
               x0=-4.115, y0=0, x1=0, y1=11.885),
    ],
    "forehand only (no strokes until M6)": [
        Target(id="fh", name="FH deep", shape="rect", x0=-4.115, y0=8, x1=4.115, y1=11.885,
               strokes=["forehand"]),
    ],
}  # fmt: skip


def outline(t: Target, side: int | None) -> np.ndarray:
    """Ground polygon (N, 3) of the target where it lies for a hitter on ``side``."""
    sx = -1.0 if (t.frame == "relative" and side == 1) else 1.0
    if t.shape == "circle":
        a = np.linspace(0, 2 * np.pi, 361)
        pts = np.column_stack([t.cx + t.r * np.cos(a), t.cy + t.r * np.sin(a)])
    else:
        x0, x1 = sorted((t.x0, t.x1))
        y0, y1 = sorted((t.y0, t.y1))
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        # Densify so lens distortion bends the edges like the real lines.
        pts = np.concatenate(
            [np.linspace(pts[i], pts[(i + 1) % 4], 50, endpoint=False) for i in range(4)]
        )
    pts = pts * sx
    return np.column_stack([pts, np.zeros(len(pts))])


def check(session, targets: list[Target]) -> dict:
    cal = calib.load(session.calibration_path)
    rows = tables.read_table(session.practice_path).to_pylist()
    segs = {r["segment_id"]: r for r in tables.read_table(session.segments_path).to_pylist()}
    n = agree = edge = hits = 0
    bad = []
    for r in rows:
        if r["landing_x"] is None or r["side"] is None or r["outcome"] == "net":
            continue
        cam = calib.camera_at(cal, segs[r["segment_id"]]["t_end"] or r["t_contact"])
        px = cam.project(np.array([[r["landing_x"], r["landing_y"], 0.0]]))[0]
        stroke = r["stroke_type"] or ("serve" if r["shot_kind"] == "serve" else None)
        applicable = [t for t in targets if not t.strokes or stroke in t.strokes]
        if not applicable:
            if r["in_target"] is not None:
                bad.append((r["t_contact"], "has a result but no target applies"))
            continue
        dists = []
        for t in applicable:
            poly = cam.project(outline(t, r["side"])).astype(np.float32)
            dists.append(cv2.pointPolygonTest(poly, (float(px[0]), float(px[1])), True))
        hit = max(dists) >= 0
        n += 1
        hits += hit
        if bool(r["in_target"]) == hit:
            agree += 1
        elif min(abs(d) for d in dists) <= EDGE_PX:
            edge += 1
        else:
            bad.append((r["t_contact"], f"practice says {r['in_target']}, image says {hit}"))
    return {"landings": n, "agree": agree, "hits": hits, "on_edge": edge, "disagree": bad}


def main(argv: list[str]) -> int:
    settings = load_settings()
    builtin = "--sets" in argv and argv[argv.index("--sets") + 1] == "builtin"
    ids = [a for a in argv if not a.startswith("--") and a != "builtin"]
    ok = True
    for sid in ids:
        session = services.session_by_id(settings, sid)
        if session is None:
            print(f"{sid}: unknown session")
            return 2
        config = session.load_config()
        original = list(config.practice.targets) if config.practice else []
        sets = BUILTIN if builtin else {"session targets": original}
        try:
            for name, targets in sets.items():
                services.set_practice(settings, sid, targets=targets)
                services.refresh_practice(settings, sid)
                res = check(session, targets)
                ok &= not res["disagree"]
                print(
                    f"{sid} {name}: {res['agree']}/{res['landings']} landings agree "
                    f"({res['hits']} in a target)"
                    + (
                        f", {res['on_edge']} within {EDGE_PX} px of an edge"
                        if res["on_edge"]
                        else ""
                    )
                    + "".join(f"\n  DISAGREE {t:.2f}: {why}" for t, why in res["disagree"])
                )
        finally:
            services.set_practice(settings, sid, targets=original)
            services.refresh_practice(settings, sid)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
