"""Net-tape reference ground truth (M7c): the user's review of each session's reference
serves, kept outside the session so it survives reprocessing and deleting the session.

``<output_root>/training/speed_refs/<session_id>.json``::

    {"session_id": "...", "device_key": "...", "air_temp_c": 18.0,
     "refs": [{"t": 1533.474, "status": "accepted", "marked": false,
               "t_racket": 1533.5886, "t_tape": 1533.9883}, ...]}

``t``: the serve's contact time (video clock); ``t_racket`` / ``t_tape``: onsets the user
placed (audio clock, null: the detected one). Written on every review edit from
``edits.json``, so it always matches the session's decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

SPEED_DIR = "speed_refs"


def labels_path(output_root: Path, session_id: str) -> Path:
    return Path(output_root) / "training" / SPEED_DIR / f"{session_id}.json"


def load_labels(output_root: Path, session_id: str) -> dict | None:
    path = labels_path(output_root, session_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_from_edits(output_root: Path, config, edits) -> Path:
    from swingvision.storage.fsutil import atomic_write_json

    path = labels_path(output_root, config.id)
    atomic_write_json(
        path,
        {
            "session_id": config.id,
            "device_key": config.device_key,
            "air_temp_c": config.air_temp_c,
            "refs": [e.model_dump() for e in edits.speed_refs],
        },
    )
    return path
