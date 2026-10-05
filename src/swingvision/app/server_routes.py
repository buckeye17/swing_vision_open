"""Flask routes for media. Only files registered to a session are served (no arbitrary paths)."""

from __future__ import annotations

from flask import Flask, abort, send_file

from swingvision.app import state

MEDIA = {
    "proxy.mp4": ("proxy_path", "video/mp4"),
    "thumb.jpg": ("thumb_path", "image/jpeg"),
}


def register_routes(server: Flask) -> None:
    @server.route("/media/<session_id>/<name>")
    def media(session_id: str, name: str):
        if name not in MEDIA:
            abort(404)
        found = state.session_for(session_id)
        if found is None:
            abort(404)
        _, _, session = found
        attr, mimetype = MEDIA[name]
        path = getattr(session, attr)
        if not path.exists():
            abort(404)
        # conditional=True → HTTP Range support, which <video> seeking needs.
        return send_file(path, mimetype=mimetype, conditional=True, max_age=0)
