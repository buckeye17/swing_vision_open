"""Learned event classifier (``sv train events``, PLAN.md §7.5).

Every *kink* the event detector finds on a predicted ball track (see
:mod:`swingvision.ball.events`) is a training example: its features (motion break, velocity
change, position relative to the player's box and the net, audio onset after delay
compensation, …) and a class from the labeled events of the ground-truth clips: ``hit``,
``bounce``, ``net``, or ``none`` when no labeled event is within ±2 frames.

The model is scikit-learn's histogram gradient boosting (the LightGBM algorithm; it handles
missing features natively). It is saved to ``<output_root>/models/events/<name>.pkl`` with a
model card; the ``events`` stage uses the newest one marked ``default`` instead of the rules.
A model is marked ``default`` only when it learned from ``MIN_TRAIN_KINKS`` kinks or more
(label events on more clips first: a dozen examples don't beat the rules).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from swingvision.ball.events import EVENT_KINDS, FEATURE_NAMES
from swingvision.storage.fsutil import atomic_write, atomic_write_json

CLASSES = ("none", *EVENT_KINDS)
MIN_CONF = 0.5
#: A trained classifier replaces the rules only when it learned from at least this many kinks.
MIN_TRAIN_KINKS = 200


def models_dir(output_root: Path) -> Path:
    return Path(output_root) / "models" / "events"


@dataclass
class EventModel:
    name: str
    clf: object

    def predict_kinds(self, feats: list[dict]) -> list[tuple[str | None, float]]:
        if not feats:
            return []
        X = features_matrix(feats)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        out = []
        for row in proba:
            k = int(np.argmax(row))
            kind = classes[k]
            conf = float(row[k])
            out.append((None if kind == "none" or conf < MIN_CONF else kind, conf))
        return out


def features_matrix(feats: list[dict]) -> np.ndarray:
    return np.array(
        [[np.nan if f.get(n) is None else float(f[n]) for n in FEATURE_NAMES] for f in feats],
        dtype=np.float64,
    )


def active_model_name(output_root: Path | None) -> str | None:
    if output_root is None:
        return None
    d = models_dir(output_root)
    if not d.exists():
        return None
    cards = []
    for p in d.glob("*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if c.get("default", False) and (d / f"{p.stem}.pkl").exists():
            cards.append(c)
    if not cards:
        return None
    return max(cards, key=lambda c: c.get("created_at", ""))["name"]


def load_model(output_root: Path, name: str) -> EventModel:
    with (models_dir(output_root) / f"{name}.pkl").open("rb") as f:
        clf = pickle.load(f)  # our own file, written by save_model
    return EventModel(name, clf)


def load_active_model(output_root: Path | None) -> EventModel | None:
    name = active_model_name(output_root)
    return load_model(output_root, name) if name else None


def label_kinks(feats: list[dict], gt_events: list[tuple[int, str]], tol: int = 2) -> list[str]:
    """Class per kink: the labeled event within ±``tol`` frames (nearest), else ``none``."""
    labels = ["none"] * len(feats)
    used = set()
    pairs = sorted(
        (abs(f["frame"] - g), i, j)
        for i, f in enumerate(feats)
        for j, (g, _) in enumerate(gt_events)
        if abs(f["frame"] - g) <= tol
    )
    taken = set()
    for _, i, j in pairs:
        if i in taken or j in used:
            continue
        taken.add(i)
        used.add(j)
        labels[i] = gt_events[j][1]
    return labels


def fit(feats: list[dict], labels: list[str], seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(features_matrix(feats), np.array(labels))
    return clf


def save_model(output_root: Path, name: str, clf, card: dict) -> Path:
    d = models_dir(output_root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.pkl"
    atomic_write(path, lambda tmp: tmp.write_bytes(pickle.dumps(clf)), suffix=".pkl")
    atomic_write_json(
        d / f"{name}.json",
        {
            "name": name,
            "kind": "events",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "features": list(FEATURE_NAMES),
            "default": int(card.get("kinks", 0)) >= MIN_TRAIN_KINKS,
            **card,
        },
    )
    return path


def collect_kinks(bench, subject, clips, log=print) -> tuple[list[dict], list[str]]:
    """Kink features on each clip's predicted track and their classes from the labels.

    ``bench``: a :class:`swingvision.training.bench_ball.Bench` (it caches detector runs).
    Only kinks inside the labeled frames of clips with labeled events are used.
    """
    from swingvision.ball import events as ev

    feats: list[dict] = []
    labels: list[str] = []
    for clip in clips:
        if not clip.events_labeled:
            continue
        got = bench.candidates(subject, clip, log=log)
        if got is None:
            continue
        cand, frames, _ = got
        track = bench._link(clip, cand, frames)
        _, fs = ev.detect_events(track, bench._event_ctx(clip))
        fs = [f for f in fs if clip.frame0 <= f["frame"] <= clip.frame1]
        gt = [(e.frame, e.kind) for e in clip.events]
        feats += fs
        labels += label_kinks(fs, gt)
    return feats, labels
