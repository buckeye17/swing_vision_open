"""M7b spike: whole-body pose models with toe keypoints on the frames around serve contacts.

Usage:
    uv run python scripts/spikes/toe_pose_bench.py run <session_id> MODEL_DIR [--out DIR]
    uv run python scripts/spikes/toe_pose_bench.py score <session_id>... [--out DIR]

``run`` decodes every frame from 0.35 s before to 0.12 s after each near serve's contact,
crops the player (the ``movement`` box, padded 1.25 like MMPose), and runs every candidate
ONNX model found in MODEL_DIR (ONNX Runtime, CUDA when available). Keypoints are stored per
serve in ``<out>/<session>_<swing>.npz``. ``score`` compares the front (right) toe with the
hand-labeled toe tips in ``training/serve_contact/<session>.json``: the error on the ground
at the keypoint's height above the sole (0 and 2 cm), the bias along / across the foot, and
what's left after removing the bias. Results: ``docs/spikes/m7b-toe-pose.md``.

Candidates (download them into MODEL_DIR; file names as below):

* ``rtmw-l.onnx``: RTMW-l 384×288 distilled from RTMW-x, COCO-WholeBody (OpenMMLab
  ``rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip``, ``end2end.onnx``)
* ``halpe26-x.onnx``: RTMPose-x 384×288, Halpe-26 body + feet (OpenMMLab
  ``rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip``)
* ``rtmw3d-x.onnx``: RTMW3D-x 384×288 (2D + relative depth, H3WB; rtmlib's Hugging Face copy)
* ``vitpose-l-wb.onnx``: ViTPose-L whole-body (a third-party ONNX export on Hugging Face)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

KIND = {
    "rtmw-l": ("simcc", (288, 384), (20, 22)),
    "halpe26-x": ("simcc", (288, 384), (21, 25)),
    "rtmw3d-x": ("simcc", (288, 384), (20, 22)),
    "vitpose-l-wb": ("heatmap", (192, 256), (20, 22)),
}
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def _warp(img, box, size, padding=1.25):
    import cv2

    x0, y0, x1, y1 = box
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    s = np.array([x1 - x0, y1 - y0]) * padding
    aspect = size[0] / size[1]
    s = np.array([max(s[0], s[1] * aspect), max(s[1], s[0] / aspect)])
    A = np.array(
        [
            [size[0] / s[0], 0, -(c[0] - s[0] / 2) * size[0] / s[0]],
            [0, size[1] / s[1], -(c[1] - s[1] / 2) * size[1] / s[1]],
        ],
        np.float32,
    )
    return cv2.warpAffine(img, A, size, flags=cv2.INTER_LINEAR), c, s


def _decode(name, outs, size):
    from swingvision.pose.feet import decode_simcc
    from swingvision.pose.pose2d import decode_heatmaps

    kind = KIND[name][0]
    if kind == "simcc":
        kp = decode_simcc(outs[0], outs[1])
        return kp
    hm = outs[0]
    kp = decode_heatmaps(hm)
    h, w = hm.shape[-2:]
    kp[..., 0] = (kp[..., 0] + 0.5) * size[0] / w
    kp[..., 1] = (kp[..., 1] + 0.5) * size[1] / h
    return kp


def run(session_id: str, model_dir: Path, out: Path) -> None:
    import onnxruntime as ort
    import pyarrow.parquet as pq
    import torch  # noqa: F401  (CUDA libraries for ONNX Runtime)

    from swingvision import services
    from swingvision.io.frames import open_source
    from swingvision.pipeline.stages.pose import box_track, me_movement
    from swingvision.settings import load_settings

    settings = load_settings()
    session = services.session_by_id(settings, session_id)
    config = session.load_config()
    models = {}
    for name in KIND:
        path = model_dir / f"{name}.onnx"
        if path.exists():
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            models[name] = ort.InferenceSession(
                str(path), opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
    boxes = box_track(me_movement(session))
    swings = pq.read_table(session.swings_path).to_pylist()
    serves = [r for r in swings if r["stroke_type"] == "serve" and r["side"] == -1]
    out.mkdir(parents=True, exist_ok=True)
    with open_source(Path(config.source.path), config.video) as src:
        for r in serves:
            path = out / f"{session_id}_{r['swing_id']}.npz"
            if path.exists():
                continue
            t0 = time.time()
            frames = list(src.frames(r["t_contact"] - 0.35, r["t_contact"] + 0.12))
            data = {
                "frames": np.array([f.index for f in frames]),
                "t": np.array([f.t_s for f in frames]),
            }
            imgs = [f.image.permute(1, 2, 0).cpu().numpy() for f in frames]
            bxs = [boxes.at(f.t_s) for f in frames]
            for name, sess in models.items():
                size = KIND[name][1]
                kps = np.full((len(frames), 133, 3), np.nan)
                for i, (img, b) in enumerate(zip(imgs, bxs, strict=True)):
                    if b is None:
                        continue
                    crop, c, s = _warp(img, b, size)
                    x = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
                    outs = sess.run(None, {sess.get_inputs()[0].name: x})
                    kp = _decode(name, outs, size)[0]
                    kp[:, 0] = kp[:, 0] * s[0] / size[0] + c[0] - s[0] / 2
                    kp[:, 1] = kp[:, 1] * s[1] / size[1] + c[1] - s[1] / 2
                    kps[i, : len(kp)] = kp
                data[f"kp_{name}"] = kps.astype(np.float32)
            np.savez_compressed(path, **data)
            print(session_id, r["swing_id"], f"{time.time() - t0:.1f}s", flush=True)


def score(session_ids: list[str], out: Path) -> None:
    from swingvision import services
    from swingvision.court import calibration as calib
    from swingvision.settings import load_settings
    from swingvision.training.serve_labels import load_labels

    settings = load_settings()
    rows = {name: [] for name in KIND}
    for sid in session_ids:
        session = services.session_by_id(settings, sid)
        cal = calib.load(session.calibration_path)
        labels = load_labels(settings.output_root, sid) or {"serves": []}
        for lab in labels["serves"]:
            if lab.get("toe") is None:
                continue
            path = out / f"{sid}_{lab['swing_id']}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            frame = lab.get("toe_frame", lab.get("frame"))
            k = list(d["frames"]).index(frame)
            cam = calib.camera_at(cal, float(d["t"][k]))
            g = cam.image_to_ground(np.asarray(lab["toe"], float))[:2]
            for name, (_kind, _size, (toe, heel)) in KIND.items():
                if f"kp_{name}" not in d:
                    continue
                kp = np.nanmedian(d[f"kp_{name}"][max(0, k - 2) : k + 3], axis=0)
                rows[name].append((cam, kp[toe, :2], kp[heel, :2], g))
    for name, rr in rows.items():
        if not rr:
            continue
        for h in (0.0, 0.02):
            e = []
            for cam, toe, heel, g in rr:
                tg = cam.image_to_ground(toe, h)[:2]
                hg = cam.image_to_ground(heel, 0.0)[:2]
                u = (tg - hg) / np.linalg.norm(tg - hg)
                v = np.array([-u[1], u[0]])
                d = g - tg
                e.append((d @ u, d @ v, np.linalg.norm(d)))
            e = np.array(e)
            bias = np.median(e[:, :2], axis=0)
            rest = np.hypot(e[:, 0] - bias[0], e[:, 1] - bias[1])
            print(
                json.dumps(
                    {
                        "model": name,
                        "kp_height_m": h,
                        "n": len(e),
                        "error_median_cm": round(100 * float(np.median(e[:, 2])), 1),
                        "bias_along_cm": round(100 * float(bias[0]), 1),
                        "bias_across_cm": round(100 * float(bias[1]), 1),
                        "after_bias_median_cm": round(100 * float(np.median(rest)), 1),
                        "after_bias_p90_cm": round(100 * float(np.percentile(rest, 90)), 1),
                    }
                )
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("session")
    r.add_argument("model_dir", type=Path)
    r.add_argument("--out", type=Path, default=Path("toe_pose_runs"))
    s = sub.add_parser("score")
    s.add_argument("sessions", nargs="+")
    s.add_argument("--out", type=Path, default=Path("toe_pose_runs"))
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.session, a.model_dir, a.out)
    else:
        score(a.sessions, a.out)


if __name__ == "__main__":
    main()
