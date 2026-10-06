"""Practice segmentation (PLAN.md §7.8): one segment per practice shot, grouped into blocks.

A practice shot is a feed (a drop, pre-serve dribbles and a toss, or a ball-machine feed),
the player's hit, and where the ball went: a landing, the net, or out of view. The segment
runs from the feed to the landing, padded on both sides, so it plays as one clip.

Shots come from three kinds of evidence, in this order:

* **Seen contacts**: the player's hits (``events``/``shots``) whose ball crosses the net: the
  shot record has a call over the net, or the first bounce after the hit (before anyone's
  next hit) is on the other half, or the ball hits the net or drops back right behind it.
  A ball coming down on the hitter's own half soon after the hit was a feed, a tap or a
  pre-serve dribble. A hit whose flight is lost counts when it was a serve: a fitted
  contact ≥ 2.2 m high, or a **toss** (the tracked ball going up and down above the
  player's head) and not one of a run of taps.
* **Unseen contacts**: a serve's contact can be above the top of the frame and a far
  player's swing can be lost against the background, while the landing is still detected.
  The first bounce of a ball on the half opposite the player (or a net contact, or a drop
  right behind the net on the player's half) with no hit before it becomes a shot when the
  impact sound is there: the loudest onset in a plausible flight time before it, strong,
  or merely present when the ball lands inside the court with the player tracked on the
  other half. Its contact time comes from that sound.
* **Sound and toss only**: at dusk a serve's ball can vanish entirely; a loud impact right
  after a toss, away from every other shot, is still a serve.

The session's audio/video offset (phones record the two with a constant lag, up to ≈0.1 s)
is measured from the seen hits and their racket sounds before any sound is used.

Each shot's **side** is the hitter's half; its **kind** is ``serve`` (serve-practice
sessions; otherwise a toss, a high contact or pre-serve dribbling behind the baseline, or
an unseen contact whose ball comes down fast in a service box), ``groundstroke`` (the ball
bounced on the hitter's half just before the contact: a drop or machine feed), or
``unknown``; unclear shots take their block's kind. A serve's **serve side** (deuce/ad)
comes from where the server stood, else from which box it went to.

**Blocks** group consecutive shots; a new block starts after a long pause (collecting balls),
when the hitter changes ends, or when serves turn into groundstrokes or back.

Measured on the user's two sessions against shots labeled by eye (``sv eval segments``):
docs/m5-practice.md.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow as pa

from swingvision.ball.events import SPEED_OF_SOUND
from swingvision.court import model as court_model
from swingvision.storage.schemas import SEGMENTS

OPEN_OUTCOMES = ("in", "out_long", "out_wide")


@dataclass(frozen=True)
class SegParams:
    pad_before_s: float = 1.0
    pad_after_s: float = 1.0
    #: A pause longer than this between two shots' contacts starts a new block.
    block_gap_s: float = 45.0
    #: A hit's landing is the first bounce/net contact within this, before the next hit.
    landing_window_s: float = 3.0
    #: Feed: activity on the hitter's half (bounces, taps) before the contact, events no
    #: further apart than ``feed_gap_s``, at most ``feed_window_s`` back.
    feed_window_s: float = 6.0
    feed_gap_s: float = 1.5
    #: The last feed event may be this long before the contact (a serve's toss is unseen).
    feed_lead_s: float = 3.0
    #: A bounce on the hitter's half this close before the contact: a groundstroke.
    drop_window_s: float = 0.8
    machine_window_s: float = 3.5
    #: Serve evidence without a serve-practice session: contact height and distance behind
    #: the net of the hitter.
    serve_height_m: float = 2.2
    serve_behind_m: float = 10.5
    #: The server's feet must be this far off the center mark to tell deuce from ad.
    serve_side_min_x: float = 0.15
    #: An unseen contact landing in a service box (± margin) this soon is a serve.
    box_margin_m: float = 0.5
    serve_flight_max_s: float = 0.8
    #: Unseen contacts: no bounce on that half (or claimed landing) for this long before ...
    unseen_quiet_s: float = 2.0
    #: ... and an impact sound this long before the landing (after the A/V offset) ...
    unseen_flight_s: tuple[float, float] = (0.25, 1.8)
    unseen_net_flight_s: tuple[float, float] = (0.12, 1.2)
    #: ... at least this strong (robust z-score), or ``impact_weak_min`` when the ball lands
    #: inside the court with the player tracked on the other half. Without an audio track
    #: the landing alone counts.
    impact_min: float = 15.0
    impact_weak_min: float = 5.0
    #: Landings further out than this (m, |x| and |y|) aren't shots (balls at the fence).
    landing_max_x: float = court_model.HALF_DOUBLES + 3.5
    landing_max_y: float = court_model.HALF_LENGTH + 3.0
    #: Into the net: the ball's first bounce is back on the hitter's half within this of the
    #: net, at least ``net_drop_min_m`` from the hitter (seen contacts) ...
    net_drop_y: float = 4.0
    net_drop_min_m: float = 5.0
    #: ... or, unseen, within this of the net with the player back near their baseline.
    net_drop_unseen_y: float = court_model.SERVICE_LINE_FROM_NET
    net_drop_player_y: float = 8.0
    #: Two of the player's shots can't be closer than this; the better supported one stays.
    min_shot_gap_s: float = 1.5
    #: A bounce deeper on a half where a shot landed this recently is that shot's ball.
    same_ball_s: float = 4.0
    #: Unseen contacts need the player (when tracked) at least this far from the net.
    unseen_player_min_y: float = 4.0
    #: A toss: at least this many tracked ball positions above the player's box (and
    #: within a box width beside it) in the ``toss_window_s`` before the contact, spanning
    #: at least ``toss_min_rise`` box heights up and down and starting no more than
    #: ``toss_max_gap`` box heights above the head (a static false detection, or a ball in
    #: the far court behind a near player, doesn't move that way).
    toss_min_points: int = 4
    toss_window_s: float = 1.6
    toss_min_rise: float = 0.2
    toss_max_gap: float = 0.4
    #: A ball coming down on the hitter's own half this soon after the hit was a tap or a
    #: drop; later, after a toss, it's the next ball being bounced (the serve went unseen).
    own_drop_s: float = 1.2
    #: A hit this soon after another of the player's hits is a tap (bouncing the ball up on
    #: the racket), not a shot, unless its ball is seen landing over the net.
    tap_gap_s: float = 1.0
    #: A fitted contact lower than this isn't a serve, whatever went up before it.
    serve_low_m: float = 1.8
    #: Sound and toss only: a loud impact (z-score) after a toss, with no shot within
    #: ``audio_only_gap_s``, is a serve whose flight was never seen (at dusk).
    audio_only_min: float = 25.0
    audio_only_gap_s: float = 2.5
    #: Audio/video offset search range (s) and the hits needed to trust it. Only hits with
    #: no other event this close count (dribbles alternate hits and bounces every 0.2 s).
    av_offset_max_s: float = 0.2
    av_offset_min_hits: int = 8
    av_offset_isolation_s: float = 0.35
    #: A shot without a swing at its hit takes the nearest swing this close (s), and a
    #: swing stroke this confident decides an unclear shot kind (M6).
    swing_match_s: float = 0.3
    pose_kind_conf: float = 0.75

    def as_config(self) -> dict:
        return asdict(self)


@dataclass
class SegInputs:
    events: pa.Table
    shots: pa.Table
    flights: pa.Table | None = None
    onsets_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onsets_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: The player's smoothed court positions (movement.parquet, player "me") and image
    #: boxes (x0, y0, x1, y1 px; optional).
    player_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    player_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    player_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    player_box: np.ndarray | None = None
    #: The tracked ball (ball/track.parquet: time, image x, y), for tosses.
    ball_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ball_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ball_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    duration_s: float = 0.0
    submode: str = "self_feed"
    has_audio: bool = True
    #: Camera position (court m), for the sound's travel time.
    camera_xyz: np.ndarray | None = None
    #: σ (m) of a landing at (x, y, t); ``None`` leaves it empty.
    landing_sigma: Callable[[float, float, float], float | None] | None = None
    #: The player's swings (``swings``, M6): each shot gets the stroke of its swing, and a
    #: confident pose-based serve or groundstroke decides an unclear shot kind.
    swings: pa.Table | None = None


@dataclass
class _Shot:
    t_contact: float
    contact_source: str
    side: int | None
    hitter: str = "me"
    shot_id: int | None = None
    hit_event_id: int | None = None
    contact_height: float | None = None
    end_reason: str = "none"
    t_end: float | None = None
    landing_event_id: int | None = None
    landing: tuple[float, float] | None = None
    landing_sigma: float | None = None
    landing_source: str | None = None
    conf: float = 0.5
    flags: list[str] = field(default_factory=list)
    t_feed: float | None = None
    feed_kind: str = "none"
    feed_event_id: int | None = None
    kind: str = "unknown"
    serve_side: str | None = None
    swing_id: int | None = None
    stroke: str | None = None
    stroke_conf: float | None = None
    block_id: int = 0
    start_t: float = 0.0
    end_t: float = 0.0


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


def _half(y: float) -> int:
    return 1 if y > 0 else -1


class _Player:
    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, box: np.ndarray | None):
        ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        order = np.argsort(t[ok])
        self.t, self.x, self.y = t[ok][order], x[ok][order], y[ok][order]
        self.box = None if box is None else np.asarray(box, dtype=np.float64)[ok][order]

    def box_at(self, t: float, max_dt: float = 1.0) -> np.ndarray | None:
        if self.box is None or not len(self.t):
            return None
        k = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if k > 0 and abs(self.t[k - 1] - t) < abs(self.t[k] - t):
            k -= 1
        b = self.box[k]
        return b if abs(self.t[k] - t) <= max_dt and np.isfinite(b).all() else None

    def at(self, t: float, max_dt: float = 1.0) -> tuple[float, float] | None:
        if not len(self.t):
            return None
        k = int(np.searchsorted(self.t, t))
        best = None
        for j in (k - 1, k):
            if (
                0 <= j < len(self.t)
                and abs(self.t[j] - t) <= max_dt
                and (best is None or abs(self.t[j] - t) < abs(self.t[best] - t))
            ):
                best = j
        return None if best is None else (float(self.x[best]), float(self.y[best]))

    def median_x(self, t0: float, t1: float) -> float | None:
        a, b = np.searchsorted(self.t, [t0, t1])
        return float(np.median(self.x[a:b])) if b > a else None


def sound_delay(camera_xyz: np.ndarray | None, xy: tuple[float, float] | None) -> float:
    """Seconds for a hit's sound (≈1 m up at ``xy``) to reach the camera."""
    if camera_xyz is None or xy is None:
        return 0.0
    return float(np.linalg.norm(np.array([xy[0], xy[1], 1.0]) - camera_xyz)) / SPEED_OF_SOUND


def av_offset(
    hit_t: np.ndarray,
    hit_delay: np.ndarray,
    onsets_t: np.ndarray,
    onsets_s: np.ndarray,
    max_s: float = 0.2,
    min_hits: int = 8,
    min_strength: float = 10.0,
) -> float:
    """Audio minus video lag (s): the most common gap between a hit's (delay-compensated)
    time and the loudest onset within ``max_s`` of it. 0 with too few hits."""
    gaps = av_gaps(hit_t, hit_delay, onsets_t, onsets_s, -max_s, max_s, min_strength)
    if len(gaps) < min_hits:
        return 0.0
    return offset_from_gaps(gaps, max_s)


def av_gaps(
    t_video: np.ndarray,
    delay: np.ndarray,
    onsets_t: np.ndarray,
    onsets_s: np.ndarray,
    lo_s: float,
    hi_s: float,
    min_strength: float = 10.0,
) -> np.ndarray:
    """For each contact seen at ``t_video`` (plus the sound's travel ``delay``): the gap to the
    loudest onset from ``lo_s`` to ``hi_s`` after it, when that onset is strong enough."""
    gaps = []
    for t, dl in zip(t_video, delay, strict=True):
        lo, hi = np.searchsorted(onsets_t, [t + dl + lo_s, t + dl + hi_s])
        if hi > lo:
            k = lo + int(np.argmax(onsets_s[lo:hi]))
            if onsets_s[k] >= min_strength:
                gaps.append(onsets_t[k] - t - dl)
    return np.asarray(gaps, dtype=np.float64)


def offset_from_gaps(g: np.ndarray, max_s: float) -> float:
    """The mode of audio/video gaps (10 ms bins), refined by the median of those near it."""
    h, edges = np.histogram(g, bins=np.arange(-max_s, max_s + 1e-9, 0.01))
    mode = edges[int(np.argmax(h))] + 0.005
    near = g[np.abs(g - mode) <= 0.03]
    return float(np.median(near)) if len(near) else float(mode)


def _rows(table: pa.Table | None) -> list[dict]:
    return [] if table is None else table.to_pylist()


def segment_practice(inp: SegInputs, p: SegParams | None = None) -> tuple[pa.Table, dict]:
    """Practice shots and blocks → (segments table, summary)."""
    p = p or SegParams()
    events = sorted(_rows(inp.events), key=lambda e: e["t_s"])
    shot_rows = {r["hit_event_id"]: r for r in _rows(inp.shots) if r["hit_event_id"] is not None}
    flight_end = {
        f["flight_id"]: f["end_event_id"] for f in _rows(inp.flights) if f["flight_id"] is not None
    }
    player = _Player(inp.player_t, inp.player_x, inp.player_y, inp.player_box)
    ball_t = np.asarray(inp.ball_t, dtype=np.float64)
    ball_x = np.asarray(inp.ball_x, dtype=np.float64)
    ball_y = np.asarray(inp.ball_y, dtype=np.float64)

    def toss(t: float) -> int:
        """Tracked ball positions above the player's head in the toss window before ``t``."""
        box = player.box_at(t - 0.5)
        if box is None or not len(ball_t):
            return 0
        lo, hi = np.searchsorted(ball_t, [t - p.toss_window_s, t + 0.05])
        bx, by = ball_x[lo:hi], ball_y[lo:hi]
        w, h = box[2] - box[0], box[3] - box[1]
        up = (by < box[1]) & (bx > box[0] - w) & (bx < box[2] + w)
        if up.sum() < 2 or np.ptp(by[up]) < p.toss_min_rise * h:
            return 0
        if box[1] - by[up].max() > p.toss_max_gap * h:
            return 0
        return int(up.sum())

    on_t = np.asarray(inp.onsets_t, dtype=np.float64)
    on_s = np.asarray(inp.onsets_s, dtype=np.float64)
    order = np.argsort(on_t)
    on_t, on_s = on_t[order], on_s[order]
    by_id = {e["event_id"]: e for e in events}
    hits = [e for e in events if e["kind"] == "hit"]
    my_hits = [e for e in hits if e["hitter"] != "machine"]

    def hitter_xy(e: dict) -> tuple[float, float] | None:
        if _finite(e["court_x"]) and _finite(e["court_y"]):
            return float(e["court_x"]), float(e["court_y"])
        return player.at(e["t_s"])

    ev_t = np.array([e["t_s"] for e in events])
    lone = [e for e in my_hits if np.sum(np.abs(ev_t - e["t_s"]) < p.av_offset_isolation_s) == 1]
    offset = av_offset(
        np.array([e["t_s"] for e in lone]),
        np.array([sound_delay(inp.camera_xyz, hitter_xy(e)) for e in lone]),
        on_t,
        on_s,
        p.av_offset_max_s,
        p.av_offset_min_hits,
    )

    def sigma_at(x: float, y: float, t: float) -> float | None:
        return inp.landing_sigma(x, y, t) if inp.landing_sigma else None

    shots: list[_Shot] = []
    claimed: set[int] = set()  # bounce/net events used as a shot's landing

    # -- seen contacts ---------------------------------------------------------------------
    for i, e in enumerate(hits):
        if e["hitter"] == "machine":
            continue
        row = shot_rows.get(e["event_id"])
        if row is None:  # rejected by the 3D fits (not at the hitter)
            continue
        side = row["side"]
        if side is None:
            xy = player.at(e["t_s"])
            side = None if xy is None else _half(xy[1])
        t_next_hit = hits[i + 1]["t_s"] if i + 1 < len(hits) else np.inf
        after = [
            x
            for x in events
            if e["t_s"] < x["t_s"] < min(t_next_hit, e["t_s"] + p.landing_window_s)
            and x["kind"] in ("bounce", "net")
        ]
        first = after[0] if after else None
        s = _Shot(
            t_contact=float(e["t_s"]),
            contact_source="hit",
            side=side,
            hitter=e["hitter"] or "me",
            shot_id=row["shot_id"],
            hit_event_id=e["event_id"],
            contact_height=row["contact_height"],
            flags=[f for f in row["quality_flags"] if f == "contact_not_at_hitter"],
        )
        outcome = row["outcome"]
        hxy = hitter_xy(e)
        if outcome in OPEN_OUTCOMES and _finite(row["landing_x"]):
            s.landing = (float(row["landing_x"]), float(row["landing_y"]))
            s.landing_sigma = row["landing_sigma_m"]
            s.landing_source = row["landing_source"]
            if row["landing_source"] == "bounce":
                s.end_reason = "bounce"
                s.landing_event_id = flight_end.get(row["flight_id"])
                b = by_id.get(s.landing_event_id) if s.landing_event_id is not None else None
                s.t_end = float(b["t_s"]) if b else e["t_s"] + (row["flight_time_s"] or 1.0)
                s.conf = 0.95
            else:
                s.end_reason = "lost"
                s.t_end = e["t_s"] + (row["flight_time_s"] or 1.0)
                s.conf = 0.85
        elif outcome == "net" or (first is not None and first["kind"] == "net"):
            s.end_reason = "net"
            net = first if first is not None and first["kind"] == "net" else None
            s.landing_event_id = net["event_id"] if net else None
            s.t_end = float(net["t_s"]) if net else e["t_s"] + (row["flight_time_s"] or 0.5)
            s.conf = 0.85
        elif (
            first is not None
            and _finite(first["court_y"])
            and side is not None
            and _half(first["court_y"]) == -side
            and ("contact_not_at_hitter" not in s.flags or toss(e["t_s"]) >= p.toss_min_points)
        ):
            s.landing = (float(first["court_x"]), float(first["court_y"]))
            s.landing_sigma = sigma_at(*s.landing, first["t_s"])
            s.landing_source = "bounce"
            s.landing_event_id = first["event_id"]
            s.end_reason = "bounce"
            s.t_end = float(first["t_s"])
            s.flags.append("landing_from_events")
            s.conf = 0.8
        elif first is not None and _net_drop(first, hxy, p):
            # Into the net: the ball dropped back on the hitter's half, next to the net.
            s.end_reason = "net"
            s.landing_event_id = first["event_id"]
            s.t_end = float(first["t_s"])
            s.flags.append("net_from_drop")
            s.conf = 0.8
        elif first is not None and (
            first["t_s"] - e["t_s"] <= p.own_drop_s or toss(e["t_s"]) < p.toss_min_points
        ):
            continue  # came down on the hitter's half: a feed, a tap, a dribble
        elif (
            not any(e["t_s"] - p.tap_gap_s < h["t_s"] < e["t_s"] for h in my_hits)
            and not (_finite(row["contact_height"]) and row["contact_height"] < p.serve_low_m)
            and (
                (_finite(row["contact_height"]) and row["contact_height"] >= p.serve_height_m)
                or toss(e["t_s"]) >= p.toss_min_points
            )
        ):
            # No landing, but a serve's toss or contact height (and not the last of a run of
            # taps bouncing the ball up on the racket).
            s.end_reason = "lost"
            s.t_end = e["t_s"] + (row["flight_time_s"] or 1.0)
            s.flags.append("no_landing")
            s.conf = 0.6
        else:
            continue
        if s.landing is not None and not _plausible_landing(s.landing, p):
            s.flags.append("landing_far_out")
        if toss(e["t_s"]) >= p.toss_min_points:
            s.flags.append("toss")
        if s.landing_event_id is not None:
            claimed.add(s.landing_event_id)
        shots.append(s)

    shots = _dedupe(shots, p)
    claimed = {s.landing_event_id for s in shots if s.landing_event_id is not None}

    # -- unseen contacts -------------------------------------------------------------------
    use_audio = inp.has_audio and len(on_t) > 0
    for e in events:
        if e["kind"] not in ("bounce", "net") or e["event_id"] in claimed:
            continue
        tb = float(e["t_s"])
        if any(abs(tb - s.t_contact) < 0.05 for s in shots):
            continue
        # The ball must not have been in play just before: no hit by the player, no bounce
        # of a shot just claimed, and (bounces) no bounce on the same half.
        if any(tb - p.unseen_flight_s[1] < h["t_s"] < tb for h in my_hits):
            continue
        if any(tb - p.unseen_quiet_s < (s.t_end or s.t_contact) < tb for s in shots if s.t_end):
            continue
        xy = player.at(tb - 0.5)
        if xy is not None and abs(xy[1]) < p.unseen_player_min_y:
            continue  # at the net: collecting balls, not hitting from behind a baseline
        if e["kind"] == "bounce":
            if not (_finite(e["court_x"]) and _finite(e["court_y"])):
                continue
            land = (float(e["court_x"]), float(e["court_y"]))
            if not _plausible_landing(land, p):
                continue
            half = _half(land[1])
            drop = (
                xy is not None
                and _half(xy[1]) == half
                and abs(land[1]) <= p.net_drop_unseen_y
                and abs(xy[1]) >= p.net_drop_player_y
            )
            if xy is not None and _half(xy[1]) == half and not drop:
                continue  # the player is on that half: dribbling, collecting balls
            # The first bounce of the ball: none on that half just before (for a drop at the
            # net, none away from the hitter's baseline, where they dribble).
            if any(
                x["kind"] == "bounce"
                and tb - p.unseen_quiet_s < x["t_s"] < tb
                and _finite(x["court_y"])
                and _half(x["court_y"]) == half
                and (not drop or abs(x["court_y"]) < p.net_drop_player_y)
                for x in events
            ):
                continue
            # A shot that landed on this half shortly before, nearer the net: its ball,
            # still travelling (bouncing on towards the baseline or the fence).
            if any(
                s.landing is not None
                and _half(s.landing[1]) == half
                and 0 < tb - (s.t_end or s.t_contact) < p.same_ball_s
                and abs(land[1]) > abs(s.landing[1])
                for s in shots
            ):
                continue
            side = half if drop else -half
            window = p.unseen_net_flight_s if drop else p.unseen_flight_s
        else:  # net contact
            land = None
            nxt = [
                x
                for x in events
                if tb < x["t_s"] < tb + 1.5 and x["kind"] == "bounce" and _finite(x["court_y"])
            ]
            if xy is not None:
                side = _half(xy[1])
            elif nxt and abs(nxt[0]["court_y"]) < 4.0:
                side = _half(nxt[0]["court_y"])  # a net ball drops on the hitter's side
            else:
                continue
            window = p.unseen_net_flight_s
        hitter_xy_guess = xy if xy is not None else (0.0, side * court_model.HALF_LENGTH)
        delay = sound_delay(inp.camera_xyz, hitter_xy_guess)
        impact_t, impact_s = None, 0.0
        if use_audio:
            lo, hi = np.searchsorted(on_t, [tb - window[1] + offset, tb - window[0] + offset])
            if hi > lo:
                k = lo + int(np.argmax(on_s[lo:hi]))
                impact_t, impact_s = float(on_t[k]), float(on_s[k])
            inside = land is not None and bool(
                court_model.in_court(np.array([land[0]]), np.array([land[1]]), singles=False)[0]
            )
            need = p.impact_weak_min if inside and xy is not None else p.impact_min
            if impact_t is None or impact_s < need:
                continue
        s = _Shot(
            t_contact=(impact_t - offset - delay) if impact_t is not None else tb - 0.8,
            contact_source="audio" if impact_t is not None else "estimate",
            side=side,
            t_end=tb,
            landing_event_id=e["event_id"],
            flags=["contact_unseen"] + ([] if xy is not None else ["player_untracked"]),
            conf=float(np.clip(0.5 + impact_s / 80, 0.5, 0.9)) if impact_t is not None else 0.5,
        )
        if land is not None and side != _half(land[1]):
            s.landing = land
            s.landing_sigma = sigma_at(land[0], land[1], tb)
            s.landing_source = "bounce"
            s.end_reason = "bounce"
        else:
            s.end_reason = "net"
            if land is not None:
                s.flags.append("net_from_drop")
        if toss(s.t_contact) >= p.toss_min_points:
            s.flags.append("toss")
        claimed.add(e["event_id"])
        shots.append(s)

    if use_audio:
        shots += _audio_only(shots, events, my_hits, player, on_t, on_s, offset, toss, inp, p)
    shots.sort(key=lambda s: s.t_contact)
    _attach_swings(shots, _rows(inp.swings), p)
    _feeds_and_kinds(shots, events, player, inp.submode, p)
    blocks = _blocks(shots, p)
    _fill_kinds(blocks, player, p)
    _segment_times(shots, inp.duration_s, p)
    table = _table(shots, blocks)
    summary = {
        "av_offset_s": round(offset, 4),
        "shots": len(shots),
        "seen": sum(s.contact_source == "hit" for s in shots),
        "unseen": sum(s.contact_source != "hit" for s in shots),
        "blocks": len(blocks),
        "serves": sum(s.kind == "serve" for s in shots),
    }
    return table, summary


def _dedupe(shots: list[_Shot], p: SegParams) -> list[_Shot]:
    """Of two shots closer than ``min_shot_gap_s``, keep the better supported (a detected
    landing beats a lost flight; an earlier hit is often the ball still in the toss)."""
    shots = sorted(shots, key=lambda s: s.t_contact)
    out: list[_Shot] = []
    for s in shots:
        if out and s.t_contact - out[-1].t_contact < p.min_shot_gap_s:
            if s.conf > out[-1].conf:
                out[-1] = s
            continue
        out.append(s)
    return out


def _audio_only(
    shots: list[_Shot],
    events: list[dict],
    my_hits: list[dict],
    player: _Player,
    on_t: np.ndarray,
    on_s: np.ndarray,
    offset: float,
    toss: Callable[[float], int],
    inp: SegInputs,
    p: SegParams,
) -> list[_Shot]:
    """Serves known only from a toss and the impact's sound: a loud onset right after the
    ball went up above the player, away from every other shot."""
    found: list[_Shot] = []
    taken = [s.t_contact for s in shots]
    for t_on, strength in zip(on_t, on_s, strict=True):
        if strength < p.audio_only_min:
            continue
        xy = player.at(t_on - offset, max_dt=0.5)
        if xy is None or abs(xy[1]) < p.serve_behind_m:
            continue
        t = float(t_on - offset - sound_delay(inp.camera_xyz, xy))
        if any(abs(t - u) < p.audio_only_gap_s for u in taken):
            continue
        # A bounce at their feet or a tap right there is what made the sound.
        if any(
            abs(e["t_s"] - t) < 0.15
            and _finite(e["court_x"])
            and _finite(e["court_y"])
            and np.hypot(e["court_x"] - xy[0], e["court_y"] - xy[1]) < 3.0
            for e in events
            if e["kind"] == "bounce"
        ) or any(abs(h["t_s"] - t) < 0.15 for h in my_hits):
            continue
        if toss(t) < p.toss_min_points:
            continue
        s = _Shot(
            t_contact=t,
            contact_source="audio",
            side=_half(xy[1]),
            end_reason="lost",
            flags=["contact_unseen", "audio_only", "no_landing", "toss"],
            conf=0.6,
        )
        found.append(s)
        taken.append(t)
    return found


def _net_drop(b: dict, hitter_xy: tuple[float, float] | None, p: SegParams) -> bool:
    """A bounce on the hitter's half right behind the net, well away from the hitter."""
    if b["kind"] != "bounce" or not (_finite(b["court_x"]) and _finite(b["court_y"])):
        return False
    if abs(b["court_y"]) > p.net_drop_y or hitter_xy is None:
        return False
    dist = float(np.hypot(b["court_x"] - hitter_xy[0], b["court_y"] - hitter_xy[1]))
    return dist >= p.net_drop_min_m


def _plausible_landing(xy: tuple[float, float], p: SegParams) -> bool:
    return abs(xy[0]) <= p.landing_max_x and abs(xy[1]) <= p.landing_max_y


def _feeds_and_kinds(
    shots: list[_Shot], events: list[dict], player: _Player, submode: str, p: SegParams
) -> None:
    """Each shot's feed and kind (serve/groundstroke), from what happened on the hitter's half
    before the contact. Serve sides are set later, once blocks fill in unclear kinds."""
    used = {s.hit_event_id for s in shots} | {s.landing_event_id for s in shots}
    prev_end = -np.inf
    for s in shots:
        t = s.t_contact
        lo = max(t - p.feed_window_s, prev_end)
        side = s.side

        def on_my_half(e: dict, side=side) -> bool:
            return side is not None and _finite(e["court_y"]) and _half(e["court_y"]) == side

        feeds = [
            e
            for e in events
            if e["kind"] == "hit"
            and e["hitter"] == "machine"
            and t - p.machine_window_s < e["t_s"] < t
        ]
        # Activity on the hitter's half: bounces and taps there, not other shots' events.
        own = [
            e
            for e in events
            if lo < e["t_s"] < t - 0.03
            and e["event_id"] not in used
            and e["hitter"] != "machine"
            and (e["kind"] == "hit" or (e["kind"] == "bounce" and on_my_half(e)))
        ]
        chain: list[dict] = []
        last = t
        for e in reversed(own):
            if last - e["t_s"] > (p.feed_gap_s if chain else p.feed_lead_s):
                break
            chain.append(e)
            last = e["t_s"]
        # A bounce on the hitter's half just before the contact: a fed groundstroke.
        bounced = any(
            e["kind"] == "bounce" and t - p.drop_window_s < e["t_s"] < t and on_my_half(e)
            for e in events
        )
        n_dribble = sum(e["kind"] == "bounce" for e in chain)
        xy = player.at(t)
        behind = None if xy is None else abs(xy[1]) >= p.serve_behind_m
        high = _finite(s.contact_height) and s.contact_height >= p.serve_height_m  # type: ignore[operator]
        posed = _pose_kind(s, p)
        if submode == "serve":
            s.kind = "serve"
        elif submode == "ball_machine":
            s.kind = "groundstroke" if bounced or posed == "groundstroke" else "unknown"
        elif bounced:
            s.kind = "groundstroke"
        elif behind is not False and (
            high or n_dribble >= 2 or "toss" in s.flags or posed == "serve"
        ):
            s.kind = "serve"  # a toss, a high contact, pre-serve dribbling, a serve's pose
        elif behind is not False and _fast_into_box(s, p):
            s.kind = "serve"  # unseen hitter, but a fast ball into a service box
        elif posed is not None:
            s.kind = posed
        if feeds:
            s.feed_kind = "machine"
            s.feed_event_id = feeds[-1]["event_id"]
            s.t_feed = float(feeds[-1]["t_s"])
        elif chain:
            s.feed_kind = "dribble" if n_dribble >= 2 and not bounced else "drop"
            s.t_feed = float(chain[-1]["t_s"])
        else:
            s.feed_kind = "none"
        prev_end = s.t_end if s.t_end is not None else t + 0.3


def _attach_swings(shots: list[_Shot], swings: list[dict], p: SegParams) -> None:
    """Each shot's swing: the one at its hit, else the nearest within ``swing_match_s``."""
    by_hit = {w["hit_event_id"]: w for w in swings if w["hit_event_id"] is not None}
    for s in shots:
        w = by_hit.get(s.hit_event_id) if s.hit_event_id is not None else None
        if w is None:
            near = [x for x in swings if abs(x["t_contact"] - s.t_contact) <= p.swing_match_s]
            w = min(near, key=lambda x: abs(x["t_contact"] - s.t_contact)) if near else None
        if w is not None:
            s.swing_id, s.stroke, s.stroke_conf = w["swing_id"], w["stroke_type"], w["stroke_conf"]


def _pose_kind(s: _Shot, p: SegParams) -> str | None:
    """``serve`` / ``groundstroke`` from a confident stroke of the shot's swing."""
    if s.stroke is None or (s.stroke_conf or 0.0) < p.pose_kind_conf:
        return None
    if s.stroke == "serve":
        return "serve"
    if s.stroke in ("forehand", "backhand"):
        return "groundstroke"
    return None


def _fill_kinds(blocks: list[list[_Shot]], player: _Player, p: SegParams) -> None:
    """Unclear shots take their block's kind when the clear ones agree; then serve sides."""
    for members in blocks:
        known = {m.kind for m in members if m.kind != "unknown"}
        if len(known) == 1:
            kind = known.pop()
            for m in members:
                m.kind = kind
        for m in members:
            if m.kind == "serve":
                if m.feed_kind == "none":
                    m.feed_kind = "toss"
                m.serve_side = _serve_side(m, player, p)


def _serve_side(s: _Shot, player: _Player, p: SegParams) -> str | None:
    """Deuce when the server stands right of the center mark (in their own frame)."""
    sd = s.side or -1
    x = player.median_x(s.t_contact - 1.0, s.t_contact)
    if x is not None and abs(x) >= p.serve_side_min_x:
        return "deuce" if -sd * x > 0 else "ad"
    if s.landing is not None:
        rx = -sd * s.landing[0]  # landing across the court in the server's frame
        return "deuce" if rx < 0 else "ad"
    return None


def _blocks(shots: list[_Shot], p: SegParams) -> list[list[_Shot]]:
    blocks: list[list[_Shot]] = []
    for s in shots:
        if blocks:
            prev = blocks[-1][-1]
            same = (
                s.t_contact - prev.t_contact <= p.block_gap_s
                and (s.side == prev.side or s.side is None or prev.side is None)
                and (
                    s.kind == prev.kind
                    or "unknown" in (s.kind, prev.kind)
                    or s.kind == _block_kind(blocks[-1])
                )
            )
            if same:
                blocks[-1].append(s)
                s.block_id = len(blocks) - 1
                continue
        blocks.append([s])
        s.block_id = len(blocks) - 1
    return blocks


def _fast_into_box(s: _Shot, p: SegParams) -> bool:
    """An unseen contact whose ball came down in a service box (± margin) soon after the
    impact sound: how a serve looks when the server is out of the picture."""
    if s.contact_source != "audio" or s.landing is None or s.t_end is None:
        return False
    x, y = s.landing
    in_box = (
        abs(x) <= court_model.HALF_SINGLES + p.box_margin_m
        and abs(y) <= court_model.SERVICE_LINE_FROM_NET + p.box_margin_m
    )
    return in_box and s.t_end - s.t_contact <= p.serve_flight_max_s


def _block_kind(members: list[_Shot]) -> str:
    known = {m.kind for m in members if m.kind != "unknown"}
    return known.pop() if len(known) == 1 else "unknown"


def _segment_times(shots: list[_Shot], duration: float, p: SegParams) -> None:
    for s in shots:
        first = s.t_feed if s.t_feed is not None else s.t_contact
        last = s.t_end if s.t_end is not None else s.t_contact + 1.5
        s.start_t = max(0.0, first - p.pad_before_s)
        s.end_t = last + p.pad_after_s
        if duration > 0:
            s.end_t = min(s.end_t, duration)
    # No overlaps: split the gap between one shot's end and the next one's feed.
    for a, b in itertools.pairwise(shots):
        if a.end_t > b.start_t:
            a_last = a.t_end if a.t_end is not None else a.t_contact
            b_first = b.t_feed if b.t_feed is not None else b.t_contact
            cut = (a_last + b_first) / 2 if b_first > a_last else (a.t_contact + b.t_contact) / 2
            a.end_t, b.start_t = cut, cut


def _table(shots: list[_Shot], blocks: list[list[_Shot]]) -> pa.Table:
    rows = []
    for i, s in enumerate(shots):
        rows.append(
            {
                "segment_id": i,
                "kind": "practice_shot",
                "start_t": s.start_t,
                "end_t": s.end_t,
                "block_id": s.block_id,
                "shot_id": s.shot_id,
                "hit_event_id": s.hit_event_id,
                "feed_event_id": s.feed_event_id,
                "feed_kind": s.feed_kind,
                "t_feed": s.t_feed,
                "t_contact": s.t_contact,
                "contact_source": s.contact_source,
                "t_end": s.t_end,
                "end_reason": s.end_reason,
                "landing_event_id": s.landing_event_id,
                "landing_x": s.landing[0] if s.landing else None,
                "landing_y": s.landing[1] if s.landing else None,
                "landing_sigma_m": s.landing_sigma,
                "landing_source": s.landing_source,
                "hitter": s.hitter,
                "side": s.side,
                "shot_kind": s.kind,
                "serve_side": s.serve_side,
                "swing_id": s.swing_id,
                "stroke_type": s.stroke,
                "stroke_conf": s.stroke_conf,
                "conf": s.conf,
                "flags": s.flags,
            }
        )
    n = len(shots)
    for b, members in enumerate(blocks):
        kinds = {m.kind for m in members}
        rows.append(
            {
                "segment_id": n + b,
                "kind": "block",
                "start_t": members[0].start_t,
                "end_t": members[-1].end_t,
                "block_id": b,
                "t_contact": members[0].t_contact,
                "t_end": members[-1].t_end,
                "side": members[0].side,
                "shot_kind": kinds.pop() if len(kinds) == 1 else "mixed",
                "n_shots": len(members),
                "flags": [],
            }
        )
    if not rows:
        return SEGMENTS.empty_table()
    return pa.table(
        {f.name: pa.array([r.get(f.name) for r in rows], f.type) for f in SEGMENTS},
        schema=SEGMENTS,
    )


def practice_shots(segments: pa.Table) -> list[dict]:
    """The practice-shot rows of a segments table, in time order."""
    return sorted(
        (r for r in segments.to_pylist() if r["kind"] == "practice_shot"),
        key=lambda r: r["t_contact"],
    )


def blocks_of(segments: pa.Table) -> list[dict]:
    return sorted(
        (r for r in segments.to_pylist() if r["kind"] == "block"), key=lambda r: r["block_id"]
    )
