"""Flask routes for media. Only files registered to a session are served (no arbitrary paths)."""

from __future__ import annotations

import io
import re

from flask import Flask, abort, request, send_file

from swingvision.app import state
from swingvision.storage import cache

MEDIA = {
    "proxy.mp4": ("proxy_path", "video/mp4"),
    "thumb.jpg": ("thumb_path", "image/jpeg"),
    "court_bg.jpg": ("court_background_path", "image/jpeg"),
}
WINDOW_IMAGE = re.compile(r"^court_w(\d{2})\.jpg$")


def register_routes(server: Flask) -> None:
    @server.route("/media/<session_id>/<name>")
    def media(session_id: str, name: str):
        window = WINDOW_IMAGE.match(name)
        if name not in MEDIA and window is None:
            abort(404)
        found = state.session_for(session_id)
        if found is None:
            abort(404)
        _, _, session = found
        if window is not None:
            path, mimetype = session.court_window_path(int(window.group(1))), "image/jpeg"
        else:
            attr, mimetype = MEDIA[name]
            path = getattr(session, attr)
        if not path.exists():
            abort(404)
        # On a network share: the proxy streams from there until its local copy (made in the
        # background) is complete; images are copied right away.
        path = cache.local_if_ready(path) if name == "proxy.mp4" else cache.local(path)
        # conditional=True → HTTP Range support, which <video> seeking needs.
        return send_file(path, mimetype=mimetype, conditional=True, max_age=0)

    @server.route("/export/<session_id>/<what>.<fmt>")
    def export(session_id: str, what: str, fmt: str):
        """A session table as a CSV or Parquet download (``analysis.export``)."""
        from swingvision.analysis import export as ex

        if what not in ex.EXPORTS or fmt not in ex.FORMATS:
            abort(404)
        found = state.session_for(session_id)
        if found is None:
            abort(404)
        session = found[2]
        try:
            data = ex.export_bytes(session, what, fmt)
        except ex.ExportError as exc:
            return str(exc), 409, {"Content-Type": "text/plain; charset=utf-8"}
        return send_file(
            io.BytesIO(data),
            mimetype="text/csv" if fmt == "csv" else "application/vnd.apache.parquet",
            as_attachment=True,
            download_name=ex.filename(session, what, fmt),
            max_age=0,
        )

    @server.route("/labeling/frame/<session_id>/<clip_id>/<int:frame>.jpg")
    def labeling_frame(session_id: str, clip_id: str, frame: int):
        """A region of a cached labeling frame at full resolution (query: x0, y0, w, h)."""
        import cv2

        from swingvision.training import clipcache as cc
        from swingvision.training.labels import LabelStore

        s = state.settings()
        if s.output_root is None:
            abort(404)
        store = LabelStore(s.output_root)
        try:
            clip = store.get(session_id, clip_id)
        except ValueError:
            abort(404)
        if clip is None:
            abort(404)
        path = cc.frame_path(store, clip, frame)
        if not path.exists():
            abort(404)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        x0 = max(0, min(w - 1, request.args.get("x0", 0, type=int)))
        y0 = max(0, min(h - 1, request.args.get("y0", 0, type=int)))
        cw = max(1, request.args.get("w", w, type=int))
        ch = max(1, request.args.get("h", h, type=int))
        crop = img[y0 : y0 + ch, x0 : x0 + cw]
        if crop.shape[1] > 1920:  # whole-frame views: half size is plenty
            crop = cv2.resize(
                crop, (crop.shape[1] // 2, crop.shape[0] // 2), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            abort(500)
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg", max_age=3600)
