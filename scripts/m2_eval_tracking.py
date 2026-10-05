"""M2 exit-criterion check: is the player tracked during hitting time, and only the player?

Usage:  uv run python scripts/m2_eval_tracking.py SESSION_ID [--out DIR] [--samples 48]

* **Hitting time** = ±``--window`` s around strong audio onsets (racket impacts and bounces;
  ball-collection noise mostly falls below the strength threshold). Frames too dark to see
  anything are excluded, as everywhere else.
* **Coverage** = share of processed frames in hitting time where "me" was detected or bridged.
* **Contact sheets** of random hitting moments (``--out``): the "me" box in yellow, other
  people in red (inside the court area) or gray (outside it), static objects in blue, for a
  visual check that the box is on the player and never on someone else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc

from swingvision import services
from swingvision.io.frames import grab_frame
from swingvision.players.movement import usable_frames
from swingvision.settings import load_settings
from swingvision.storage import tables

COLORS = {"me": (0, 215, 255), "other": (40, 40, 255), "outside": (150, 150, 150),
          "static": (255, 140, 40)}  # fmt: skip


def hitting_windows(onsets_t: np.ndarray, strength: np.ndarray, min_strength: float, half: float):
    """Merged [t0, t1] intervals around strong onsets."""
    t = np.sort(onsets_t[strength >= min_strength])
    out: list[list[float]] = []
    for x in t:
        if out and x - half <= out[-1][1]:
            out[-1][1] = x + half
        else:
            out.append([x - half, x + half])
    return np.array(out).reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--min-strength", type=float, default=8.0)
    ap.add_argument("--window", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = load_settings()
    session = services.session_by_id(settings, args.session_id)
    if session is None:
        raise SystemExit(f"Unknown session {args.session_id}")
    config = session.load_config()
    frames = tables.read_table(session.pass1_frames_path)
    tracks = tables.read_table(session.player_tracks_path)
    movement = tables.read_table(session.movement_path)
    onsets = tables.read_table(session.audio_onsets_path)

    ft = frames.column("t_s").to_numpy()
    ff = frames.column("frame").to_numpy()
    p = settings.processing
    lit = usable_frames(frames, p.dark_luma, p.view_min)
    tracked = np.isin(ff, movement.column("frame").to_numpy())
    detected = np.isin(ff, tracks.filter(pc.equal(tracks.column("role"), "me")).column("frame"))

    win = hitting_windows(
        onsets.column("t_s").to_numpy(),
        onsets.column("strength").to_numpy(),
        args.min_strength,
        args.window,
    )
    lo = np.searchsorted(win[:, 0], ft, side="right") - 1
    in_hit = (lo >= 0) & (ft <= win[np.clip(lo, 0, None), 1]) if len(win) else np.zeros_like(lit)
    sel = in_hit & lit
    report = {
        "session": config.name,
        "processed_frames": len(ft),
        "lit_frames": int(lit.sum()),
        "hitting_windows": len(win),
        "hitting_time_s": round(float(np.sum(win[:, 1] - win[:, 0])), 1) if len(win) else 0.0,
        "hitting_frames_lit": int(sel.sum()),
        "coverage_hitting_tracked": round(float(tracked[sel].mean()), 4) if sel.any() else None,
        "coverage_hitting_detected": round(float(detected[sel].mean()), 4) if sel.any() else None,
        "coverage_all_lit_tracked": round(float(tracked[lit].mean()), 4),
        "roles": dict(
            zip(*np.unique(tracks.column("role").to_pylist(), return_counts=True), strict=True)
        ),
    }
    report["roles"] = {str(k): int(v) for k, v in report["roles"].items()}
    # Untracked hitting frames, grouped into spans, to look at.
    miss_t = ft[sel & ~tracked]
    spans = []
    for x in miss_t:
        if spans and x - spans[-1][1] < 1.0:
            spans[-1][1] = x
        else:
            spans.append([x, x])
    report["missed_spans"] = [[round(a, 2), round(b, 2)] for a, b in spans][:60]
    print(json.dumps(report, indent=2))

    if args.out is None:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero(sel)
    pick = np.sort(rng.choice(cand, size=min(args.samples, len(cand)), replace=False))
    tt = tracks.column("t_s").to_numpy()
    info = config.video
    tiles = []
    for i in pick:
        t = float(ft[i])
        img = cv2.cvtColor(grab_frame(settings.ffmpeg(), Path(config.source.path), t, info),
                           cv2.COLOR_RGB2BGR)  # fmt: skip
        rows = np.flatnonzero(np.abs(tt - t) < 1e-4)
        for r in rows:
            role = tracks.column("role")[r].as_py()
            b = [int(tracks.column(c)[r].as_py()) for c in ("x0", "y0", "x1", "y1")]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), COLORS[role], 8 if role == "me" else 5)
        small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        status = "TRACKED" if tracked[i] else "MISSED"
        cv2.putText(small, f"{t:8.2f}s {status}", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)  # fmt: skip
        tiles.append(small)
    for k in range(0, len(tiles), 12):
        chunk = tiles[k : k + 12]
        while len(chunk) % 4:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[j : j + 4]) for j in range(0, len(chunk), 4)])
        cv2.imwrite(
            str(args.out / f"sheet_{k // 12:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
    print(f"Wrote {len(tiles)} samples to {args.out}")


if __name__ == "__main__":
    main()
