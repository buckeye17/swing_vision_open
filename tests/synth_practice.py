"""Synthetic practice event streams (events + shots tables) for segmentation tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from swingvision.storage.schemas import EVENTS, SHOTS


@dataclass
class Stream:
    events: list[dict] = field(default_factory=list)
    shots: list[dict] = field(default_factory=list)
    onsets: list[tuple[float, float]] = field(default_factory=list)
    player: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y

    def event(self, kind: str, t: float, x=None, y=None, hitter=None) -> int:
        eid = len(self.events)
        self.events.append(
            {
                "event_id": eid,
                "kind": kind,
                "frame": int(t * 60),
                "t_s": t,
                "court_x": x,
                "court_y": y,
                "conf": 0.8,
                "source": "rules",
                "hitter": hitter,
            }
        )
        return eid

    def hit(
        self,
        t: float,
        x: float,
        y: float,
        *,
        outcome: str = "unknown",
        landing: tuple[float, float] | None = None,
        land_t: float | None = None,
        height: float | None = 1.0,
        speed: float | None = None,
        hitter: str = "me",
        flags: list[str] | None = None,
        sound: float | None = 40.0,
    ) -> int:
        """A hit, its shot row, and (with ``landing``) the bounce event it ends at."""
        eid = self.event("hit", t, x, y, hitter)
        side = -1 if y < 0 else 1
        if landing is not None:
            self.event("bounce", land_t or t + 0.9, *landing)
        if sound:
            self.onsets.append((t, sound))
        self.shots.append(
            {
                "shot_id": len(self.shots),
                "session_id": "s",
                "hitter": hitter,
                "hit_event_id": eid,
                "flight_id": None,
                "frame_contact": int(t * 60),
                "t_contact": t,
                "contact_x": x,
                "contact_y": y,
                "contact_height": height,
                "side": side,
                "end_kind": "bounce" if landing else "lost",
                "landing_x": landing[0] if landing else None,
                "landing_y": landing[1] if landing else None,
                "landing_sigma_m": 0.1 if landing else None,
                "landing_source": "bounce" if landing else None,
                "outcome": outcome,
                "speed_racket_kmh": speed,
                "speed_sigma_kmh": 1.0 if speed else None,
                "flight_time_s": (land_t - t) if land_t else 0.9,
                "quality_flags": flags or [],
            }
        )
        return eid

    def stand(self, t0: float, t1: float, x: float, y: float, hz: float = 15.0) -> None:
        for t in np.arange(t0, t1, 1 / hz):
            self.player.append((float(t), x, y))

    def tables(self) -> tuple[pa.Table, pa.Table]:
        ev = sorted(self.events, key=lambda e: e["t_s"])
        events = pa.table(
            {f.name: pa.array([e.get(f.name) for e in ev], f.type) for f in EVENTS}, schema=EVENTS
        )
        shots = pa.table(
            {f.name: pa.array([s.get(f.name) for s in self.shots], f.type) for f in SHOTS},
            schema=SHOTS,
        )
        return events, shots

    def inputs(self, **kw):
        from swingvision.analysis.segmentation import SegInputs

        events, shots = self.tables()
        on = sorted(self.onsets)
        pl = sorted(self.player)
        return SegInputs(
            events=events,
            shots=shots,
            onsets_t=np.array([o[0] for o in on]),
            onsets_s=np.array([o[1] for o in on]),
            player_t=np.array([p[0] for p in pl]),
            player_x=np.array([p[1] for p in pl]),
            player_y=np.array([p[2] for p in pl]),
            duration_s=max([e["t_s"] for e in self.events] + [0]) + 10,
            camera_xyz=np.array([0.0, -18.0, 3.0]),
            **kw,
        )
