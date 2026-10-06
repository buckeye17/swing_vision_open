"""Learned stroke classifier (``sv train strokes``, PLAN.md §7.7, §10).

Training examples are the swings of every session with stroke labels: the ground truth in
``training/strokes/<id>.json`` (a labeled stroke's nearest swing gets its class, every other
swing in the labeled spans is ``other``) and the user's corrections on the Swings page
(``edits.json``), which win. Each example is the swing's hitter-frame joint sequence around
the contact plus its ball features (:func:`swingvision.pose.strokes.sequence`).

The model (:class:`swingvision.pose.strokes.StrokeModel`, a small bidirectional GRU) is
scored by leaving one session out at a time against the rules on the same swings; it is
marked ``validated`` (and then used by the ``swings`` stage, setting ``stroke_model: auto``)
only when it learned from ``MIN_TRAIN`` swings, saw at least two classes ``MIN_PER_CLASS``
times, and beat the rules on held-out sessions. It only takes over for the classes it was
trained on: a swing the rules call something else keeps the rules' answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from swingvision.pose import strokes as st
from swingvision.storage.fsutil import atomic_write_json

MIN_TRAIN = 100
MIN_PER_CLASS = 10
LABEL_MATCH_S = 0.5


@dataclass
class Examples:
    seqs: list[np.ndarray] = field(default_factory=list)
    balls: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    session: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)


def swing_labels(
    swings: list[dict], labels: dict | None, edits: list[tuple[float, str]]
) -> dict[int, str]:
    """swing_id → class, from the ground truth and the user's corrections."""
    out: dict[int, str] = {}
    if labels:
        spans = labels.get("spans") or []
        strokes = [s for s in labels["strokes"] if s.get("visible", True)]
        hidden = [s["t"] for s in labels["strokes"] if not s.get("visible", True)]
        for r in swings:
            t = r["t_contact"]
            if spans and not any(a <= t <= b for a, b in spans):
                continue
            if any(abs(t - h) <= 0.8 for h in hidden):
                continue
            out[r["swing_id"]] = "other"
        for s in strokes:
            near = [r for r in swings if abs(r["t_contact"] - s["t"]) <= LABEL_MATCH_S]
            if near:
                best = min(near, key=lambda r: abs(r["t_contact"] - s["t"]))
                out[best["swing_id"]] = s["stroke"]
    for t, stroke in edits:
        near = [r for r in swings if abs(r["t_contact"] - t) <= 0.25]
        if near:
            out[min(near, key=lambda r: abs(r["t_contact"] - t))["swing_id"]] = stroke
    return out


def collect(session, labels: dict | None, edits: list[tuple[float, str]], ex: Examples) -> int:
    """Add one session's labeled swings to ``ex``; returns how many."""
    from swingvision.pipeline.stages.pose import ball_context
    from swingvision.pose import swings as sw
    from swingvision.pose.kinematics import frame_kinematics
    from swingvision.storage import tables
    from swingvision.storage.fsutil import read_json

    rows = tables.read_table(session.swings_path).to_pylist()
    hand = read_json(session.swings_summary_path)["racket_hand"]
    ball = ball_context(session)
    pose = sw.PoseSeries.from_table(tables.read_table(session.pose3d_path))
    lab = swing_labels(rows, labels, edits)
    n = 0
    for r in rows:
        y = lab.get(r["swing_id"])
        if y is None or r["stroke_rules"] is None:
            continue
        idx = pose.around(r["t_contact"], st.SEQ_BEFORE_S + 0.2, st.SEQ_AFTER_S + 0.2)
        if len(idx) < 10:
            continue
        kin = frame_kinematics(pose.t[idx], pose.joints[idx], r["side"], hand)
        seq = st.sequence(kin, r["t_contact"])
        if seq is None:
            continue
        hit = r["hit_event_id"]
        incoming, bounced = (
            ball.incoming.get(hit, (None, None)) if hit is not None else (None, None)
        )
        f = st.features(kin, r["t_contact"], r["t_backswing_end"], incoming, bounced)
        ex.seqs.append(seq)
        ex.balls.append(f.ball_vector())
        ex.labels.append(y)
        ex.rules.append(r["stroke_rules"])
        ex.session.append(session.path.name)
        n += 1
    return n


def fit(ex: Examples, idx: np.ndarray, epochs: int = 60, seed: int = 0, hidden: int = 64):
    """Train a :class:`StrokeModel` network on ``ex[idx]`` → (net, classes)."""
    import torch

    torch.manual_seed(seed)
    classes = sorted({ex.labels[i] for i in idx})
    cid = {c: k for k, c in enumerate(classes)}
    X = torch.from_numpy(np.stack([ex.seqs[i] for i in idx])).float()
    B = torch.tensor([ex.balls[i] for i in idx], dtype=torch.float32)
    y = torch.tensor([cid[ex.labels[i]] for i in idx])
    counts = torch.bincount(y, minlength=len(classes)).float()
    weight = (counts.sum() / (len(classes) * counts.clamp(min=1))).clamp(max=10)
    net = st._net(X.shape[-1], len(classes), hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(y), generator=g).split(64):
            opt.zero_grad()
            loss = loss_fn(net(X[b], B[b]), y[b])
            loss.backward()
            opt.step()
    net.eval()
    return net, classes


def _predict(net, classes: list[str], ex: Examples, idx: np.ndarray) -> list[str]:
    model = st.StrokeModel("tmp", classes, {}, net)
    labels, _ = model.predict(
        np.stack([ex.seqs[i] for i in idx]), np.array([ex.balls[i] for i in idx], np.float32)
    )
    return [
        m if r in classes else r for m, r in zip(labels, (ex.rules[i] for i in idx), strict=True)
    ]


def cross_validate(ex: Examples, epochs: int = 60) -> dict:
    """Leave one session out: model vs rules accuracy on the held-out swings."""
    sessions = sorted(set(ex.session))
    folds = []
    if len(sessions) < 2:
        return {"folds": [], "note": "one labeled session: no held-out score"}
    for s in sessions:
        test = np.array([i for i, x in enumerate(ex.session) if x == s])
        train = np.array([i for i, x in enumerate(ex.session) if x != s])
        net, classes = fit(ex, train, epochs)
        pred = _predict(net, classes, ex, test)
        truth = [ex.labels[i] for i in test]
        rules = [ex.rules[i] for i in test]
        folds.append(
            {
                "session": s,
                "n": len(test),
                "model_acc": float(np.mean([a == b for a, b in zip(pred, truth, strict=True)])),
                "rules_acc": float(np.mean([a == b for a, b in zip(rules, truth, strict=True)])),
            }
        )
    n = sum(f["n"] for f in folds)
    return {
        "folds": folds,
        "model_acc": sum(f["model_acc"] * f["n"] for f in folds) / n,
        "rules_acc": sum(f["rules_acc"] * f["n"] for f in folds) / n,
    }


def train_and_save(
    output_root: Path, name: str, ex: Examples, epochs: int = 60
) -> tuple[Path, dict]:
    """Cross-validate, train on everything, save the weights and the card."""
    import torch

    cv = cross_validate(ex, epochs)
    net, classes = fit(ex, np.arange(len(ex)), epochs)
    counts = {c: ex.labels.count(c) for c in sorted(set(ex.labels))}
    enough = sum(v >= MIN_PER_CLASS for v in counts.values()) >= 2
    beat = bool(cv.get("folds")) and cv["model_acc"] > cv["rules_acc"]
    card = {
        "name": name,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": classes,
        "counts": counts,
        "sessions": sorted(set(ex.session)),
        "n_train": len(ex),
        "epochs": epochs,
        "hidden": 64,
        "cross_validation": cv,
        "validated": bool(len(ex) >= MIN_TRAIN and enough and beat),
    }
    d = st.models_dir(output_root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.pt"
    torch.save(net.state_dict(), str(path))
    atomic_write_json(path.with_suffix(".json"), card)
    return path, card
