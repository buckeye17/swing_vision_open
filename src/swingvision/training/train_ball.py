"""Train the slim U-Net ball detector on labeled clips (``sv train ball``, PLAN.md §10).

Samples are labeled frames of the clips in the label store. Each sample is a patch of the
three input frames (``t - 2``, ``t``, ``t + 2``) at the model's scale, read straight from the
clips' JPEG frame cache (decoded at half size by libjpeg, which is the model scale for 4K).

* **Positives**: patches containing the labeled ball (random offset).
* **Negatives**: frames labeled ``none``, random places, and the player's box (rackets,
  limbs and shoes moving across court lines look like balls to a motion detector).
* ``occluded`` frames are skipped.

The target is a Gaussian at the ball on the half-resolution heatmap; the loss is the
CenterNet focal loss. Augmentation: horizontal flip, time reversal (physics is symmetric),
brightness/contrast/gamma (dusk), and a little scale jitter.

The train/validation split is by session (``--val-session``) or by the clips' ``split``.
Weights go to ``<output_root>/models/ball/<name>.pt`` with a model card ``<name>.json``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np

from swingvision.ball.detectors.unet import OFFSETS, build_model, card_path, weights_path
from swingvision.storage.fsutil import atomic_write_json
from swingvision.training import clipcache as cc
from swingvision.training.labels import BallClip, LabelStore

PATCH = 320  # model pixels
SCALE = 0.5  # model pixels per full-resolution pixel (1920 for a 3840-wide crop)
SIGMA = 1.0  # heatmap pixels


@dataclass
class Sample:
    clip: BallClip
    frame: int
    vis: str
    x: float | None
    y: float | None


@dataclass
class TrainConfig:
    name: str
    epochs: int = 30
    batch: int = 16
    lr: float = 2e-3
    width: int = 32
    pos_frac: float = 0.65
    player_frac: float = 0.2
    #: Share of samples centered on an earlier model's false detection (when there is one).
    hard_frac: float = 0.25
    workers: int = 8
    seed: int = 0
    #: Keep the epoch with the lowest validation loss (only when validation clips are
    #: independent of the final test set); otherwise the last epoch.
    select_best: bool = False


def collect(store: LabelStore, clips: list[BallClip]) -> list[Sample]:
    out = []
    for c in clips:
        if not cc.is_cached(store, c):
            continue
        for f in c.frames:
            lab = c.label(f)
            if lab is None or lab.vis == "occluded":
                continue
            out.append(Sample(c, f, lab.vis, lab.x, lab.y))
    return out


def player_boxes_for(clips: list[BallClip], settings) -> dict[str, object]:
    """Session id → PlayerBoxes (for hard-negative patches)."""
    from swingvision import services
    from swingvision.ball.trajectory import PlayerBoxes
    from swingvision.storage import tables

    out = {}
    for sid in {c.session_id for c in clips}:
        s = services.session_by_id(settings, sid)
        if s is not None and s.movement_path.exists():
            out[sid] = PlayerBoxes.from_movement(tables.read_table(s.movement_path))
    return out


class BallDataset:
    """Indexable dataset (works with torch's DataLoader)."""

    def __init__(self, store, samples, boxes, train: bool, cfg: TrainConfig, seed: int = 0):
        self.store = store
        self.samples = samples
        self.boxes = boxes
        self.train = train
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # Hard negatives: where an earlier model predicted a ball that isn't one (stored
        # with the clip's frame cache when the clip was created or re-predicted).
        self.hard: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
        for c in {(x.clip.session_id, x.clip.clip_id): x.clip for x in samples}.values():
            self.hard[(c.session_id, c.clip_id)] = _wrong_predictions(store, c)

    def __len__(self) -> int:
        return len(self.samples)

    def _read(self, s: Sample, f: int) -> np.ndarray:
        p = half_path(self.store, s.clip, f)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
        if img is None:  # no half-size copy: decode the full frame at half size
            p = cc.frame_path(self.store, s.clip, f)
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            raise FileNotFoundError(p)
        return img  # BGR

    def __getitem__(self, i: int):
        rng = np.random.default_rng(None if self.train else 1000 + i)
        s = self.samples[i]
        frames = [s.frame + o for o in OFFSETS]
        if self.train and rng.random() < 0.5:
            frames = frames[::-1]  # time reversal
        imgs = [self._read(s, f) for f in frames]
        H, W = imgs[0].shape[:2]
        zoom = rng.uniform(0.8, 1.25) if self.train else 1.0
        size = round(PATCH / zoom)
        # Patch center (model pixels at SCALE).
        has_ball = s.vis == "visible"
        bx = (s.x + 0.5) * SCALE - 0.5 if has_ball else None
        by = (s.y + 0.5) * SCALE - 0.5 if has_ball else None
        mode = rng.random()
        hard = self.hard.get((s.clip.session_id, s.clip.clip_id), {}).get(s.frame)
        if self.train and hard is not None and rng.random() < self.cfg.hard_frac:
            cx = (hard[0] + 0.5) * SCALE - 0.5 + rng.uniform(-0.3, 0.3) * size
            cy = (hard[1] + 0.5) * SCALE - 0.5 + rng.uniform(-0.3, 0.3) * size
        elif has_ball and (mode < self.cfg.pos_frac or not self.train):
            cx = bx + rng.uniform(-0.4, 0.4) * size
            cy = by + rng.uniform(-0.4, 0.4) * size
        elif mode < self.cfg.pos_frac + self.cfg.player_frac and s.clip.session_id in self.boxes:
            pb = self.boxes[s.clip.session_id]
            t = s.clip.t0_s + (s.frame - s.clip.frame0) / 60.0
            k = int(np.clip(np.searchsorted(pb.t, t), 0, len(pb.t) - 1)) if len(pb.t) else -1
            if k >= 0 and abs(pb.t[k] - t) < 0.2:
                x0, y0, x1, y1 = pb.box[k] * SCALE
                cx = rng.uniform(x0, x1)
                cy = rng.uniform(y0, y1)
            else:
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        else:
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        x0 = int(np.clip(round(cx - size / 2), 0, max(0, W - size)))
        y0 = int(np.clip(round(cy - size / 2), 0, max(0, H - size)))
        crops = [im[y0 : y0 + size, x0 : x0 + size] for im in imgs]
        if size != PATCH:
            crops = [cv2.resize(c, (PATCH, PATCH), interpolation=cv2.INTER_AREA) for c in crops]
        x = np.concatenate([c[..., ::-1] for c in crops], axis=2).astype(np.float32) / 255.0
        hh = PATCH // 2
        target = np.zeros((hh, hh), np.float32)
        if has_ball:
            # Ball in patch model coordinates, then heatmap coordinates.
            u = (bx - x0) * (PATCH / size)
            v = (by - y0) * (PATCH / size)
            hu, hv = (u + 0.5) / 2 - 0.5, (v + 0.5) / 2 - 0.5
            if -3 <= hu < hh + 3 and -3 <= hv < hh + 3:
                gy, gx = np.mgrid[0:hh, 0:hh]
                target = np.exp(-((gx - hu) ** 2 + (gy - hv) ** 2) / (2 * SIGMA**2)).astype(
                    np.float32
                )
                # The focal loss's positive is the pixel nearest the ball (exactly 1).
                iu, iv = round(hu), round(hv)
                if 0 <= iu < hh and 0 <= iv < hh:
                    target[iv, iu] = 1.0
        if self.train:
            if rng.random() < 0.5:
                x = x[:, ::-1]
                target = target[:, ::-1]
            gain = rng.uniform(0.5, 1.3)
            bias = rng.uniform(-0.08, 0.08)
            gamma = rng.uniform(0.7, 1.4)
            x = np.clip(np.power(np.clip(x * gain + bias, 0, 1), gamma), 0, 1)
        return (
            np.ascontiguousarray(x.transpose(2, 0, 1)),
            np.ascontiguousarray(target),
        )


def _wrong_predictions(store: LabelStore, clip: BallClip) -> dict[int, tuple[float, float]]:
    """Frame → stored prediction that is not the labeled ball (≥ 25 px off, or no ball)."""
    from swingvision.training.assist import load_predictions

    out = {}
    for f, (x, y, _) in load_predictions(store, clip).items():
        lab = clip.label(f)
        if lab is None or lab.vis == "occluded":
            continue
        if lab.vis == "none" or np.hypot(lab.x - x, lab.y - y) > 25:
            out[f] = (x, y)
    return out


def focal_loss(logits, target):
    """CenterNet focal loss (target: Gaussian peaks of 1 at the ball)."""
    import torch

    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.99
    neg_w = (1 - target) ** 4
    pos_loss = (torch.log(p) * (1 - p) ** 2)[pos].sum()
    neg_loss = (torch.log(1 - p) * p**2 * neg_w)[~pos].sum()
    n_pos = pos.sum().clamp_min(1)
    return -(pos_loss + neg_loss) / n_pos


def peak_accuracy(logits, target, tol: float = 2.0, thr: float = 0.3) -> tuple[int, int, int]:
    """(correct, missed, false) of the per-patch maximum against the target peak."""
    import torch

    p = torch.sigmoid(logits)
    flat = p.flatten(1)
    v, k = flat.max(1)
    w = p.shape[-1]
    py, px = (k // w).float(), (k % w).float()
    tf = target.flatten(1)
    tv, tk = tf.max(1)
    ty, tx = (tk // w).float(), (tk % w).float()
    has = tv > 0.5
    det = v > thr
    close = torch.hypot(px - tx, py - ty) <= tol
    correct = int((has & det & close).sum())
    missed = int((has & ~(det & close)).sum())
    false = int((~has & det).sum() + (has & det & ~close).sum())
    return correct, missed, false


def train(
    store: LabelStore,
    settings,
    cfg: TrainConfig,
    train_clips: list[BallClip],
    val_clips: list[BallClip],
    log: Callable[[str], None] = print,
    device: str | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    tr = collect(store, train_clips)
    va = collect(store, val_clips)
    if not tr:
        raise RuntimeError("No labeled training frames (are the clips' frames cached?)")
    ensure_half_cache(store, tr + va, log)
    boxes = player_boxes_for(train_clips + val_clips, settings)
    ds_tr = BallDataset(store, tr, boxes, True, cfg, cfg.seed)
    ds_va = BallDataset(store, va, boxes, False, cfg)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        drop_last=True,
        persistent_workers=cfg.workers > 0,
    )
    dl_va = DataLoader(ds_va, batch_size=cfg.batch, num_workers=cfg.workers) if va else None
    model = build_model(cfg.width, len(OFFSETS)).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = cfg.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    log(
        f"Training {cfg.name}: {len(tr)} train frames ({len({s.clip.clip_id for s in tr})} clips), "
        f"{len(va)} val frames, {cfg.epochs} epochs"
    )
    history = []
    best = (math.inf, None)
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.split(":")[0], enabled=device.startswith("cuda")
            ):
                out = model(x)
            loss = focal_loss(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        row = {"epoch": ep + 1, "train_loss": round(tot / max(1, len(dl_tr)), 4)}
        if dl_va is not None:
            model.eval()
            vl, c, m, fa = 0.0, 0, 0, 0
            with torch.inference_mode():
                for x, y in dl_va:
                    x = x.to(device).contiguous(memory_format=torch.channels_last)
                    y = y.to(device)
                    out = model(x).float()
                    vl += float(focal_loss(out, y))
                    a, b, d = peak_accuracy(out, y)
                    c, m, fa = c + a, m + b, fa + d
            row.update(
                val_loss=round(vl / max(1, len(dl_va)), 4),
                val_correct=c,
                val_missed=m,
                val_false=fa,
            )
            if cfg.select_best and row["val_loss"] < best[0]:
                best = (
                    row["val_loss"],
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
        history.append(row)
        log(json.dumps(row))
    state = best[1] if best[1] is not None else model.state_dict()
    root = store.root.parent.parent
    wp = weights_path(root, cfg.name)
    wp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in state.items()}, wp)
    card = {
        "name": cfg.name,
        "kind": "unet",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "offsets": list(OFFSETS),
        "width": cfg.width,
        "input_px": round(3840 * SCALE),
        "scale": SCALE,
        "threshold": 0.1,
        "train_clips": sorted(f"{c.session_id}/{c.clip_id}" for c in train_clips),
        "val_clips": sorted(f"{c.session_id}/{c.clip_id}" for c in val_clips),
        "train_frames": len(tr),
        "val_frames": len(va),
        "epochs": cfg.epochs,
        "history": history,
        "train_seconds": round(time.time() - t0, 1),
    }
    atomic_write_json(card_path(root, cfg.name), card)
    log(f"Saved {wp}")
    return card


def half_path(store: LabelStore, clip: BallClip, frame: int):
    """Half-size copy of a cached frame (training reads these: 4× fewer pixels to decode)."""
    return store.cache_dir(clip.session_id, clip.clip_id) / "half" / f"f{frame:07d}.jpg"


def ensure_half_cache(store: LabelStore, samples: list[Sample], log=print) -> None:
    """Write half-size copies of every frame the samples need (threaded; cv2 drops the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    need = {
        (s.clip.session_id, s.clip.clip_id, s.frame + o): s.clip for s in samples for o in OFFSETS
    }
    todo = [(c, f) for (_, _, f), c in need.items() if not half_path(store, c, f).exists()]
    if not todo:
        return
    log(f"Writing {len(todo)} half-size frames for training")

    def one(item):
        c, f = item
        img = cv2.imread(str(cc.frame_path(store, c, f)), cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            return
        out = half_path(store, c, f)
        out.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(buf.tobytes())
            tmp.replace(out)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, todo))


def ensure_cached(store: LabelStore, settings, clips: list[BallClip], log=print) -> None:
    from swingvision import services

    for c in clips:
        if cc.is_cached(store, c):
            continue
        s = services.session_by_id(settings, c.session_id)
        if s is None:
            continue
        log(f"Decoding frames of {c.session_id}/{c.clip_id}")
        cc.ensure_cached(store, c, s.load_config(), settings.processing.decode_backend)
