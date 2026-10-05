"""Ball detector from a COCO YOLO's "sports ball" class (zero-shot baseline).

The same pretrained Ultralytics weights as person detection, run on the ball crop scaled to
``input_px`` (default 1920) on single frames; box centers become candidates with the class
confidence as score. COCO's sports balls are mostly large and close, so this mainly shows
what an off-the-shelf detector does with a 6-20 px tennis ball.
"""

from __future__ import annotations

import numpy as np

from swingvision.ball.detectors.base import BallDetector, DetectorSpec, register

SPORTS_BALL = 32
CONF = 0.05
BATCH = 4


class YoloBallDetector(BallDetector):
    context = 0
    max_candidates = 8

    def __init__(self, spec: DetectorSpec, device: str | None = None):
        super().__init__(spec, device)
        from swingvision.players.detect import YoloPersonDetector

        self.name = spec.name
        px = spec.input_px or 1920
        if spec.input_px is None:
            self.spec = DetectorSpec(spec.name, spec.weights, px)
        # Reuse the person detector's GPU preprocessing and weights handling.
        self._yolo = YoloPersonDetector(spec.name, input_size=px, conf=CONF, device=self.device)

    def detect(self, frames, targets):
        import torch
        from ultralytics.utils.nms import non_max_suppression

        assert self.crop is not None, "prepare() first"
        out: list[np.ndarray] = []
        y = self._yolo
        for b in range(0, len(targets), BATCH):
            imgs = [frames[i] for i in targets[b : b + BATCH]]
            x, scale = y._prepare(imgs, self.crop)
            with torch.inference_mode():
                preds = y.model(x)
                preds = preds[0] if isinstance(preds, (list, tuple)) else preds
                dets = non_max_suppression(
                    preds.float(),
                    conf_thres=CONF,
                    iou_thres=0.5,
                    classes=[SPORTS_BALL],
                    max_det=self.max_candidates,
                    max_time_img=1.0,
                )
            for d in dets:
                d = d[:, :5].cpu().numpy().astype(np.float64)
                cx = (d[:, 0] + d[:, 2]) / 2 / scale + self.crop.x0
                cy = (d[:, 1] + d[:, 3]) / 2 / scale + self.crop.y0
                out.append(np.column_stack([cx, cy, d[:, 4]]).astype(np.float32))
        return out


def _register() -> None:
    from swingvision.models.registry import REGISTRY

    for name, spec in REGISTRY.items():
        if spec.task.startswith("person"):
            register(name)(lambda s, device=None: YoloBallDetector(s, device))


_register()
