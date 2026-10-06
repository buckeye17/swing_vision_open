from __future__ import annotations

import dash
import pytest

from swingvision import services
from swingvision.app.components.file_browser import _breadcrumb_paths, list_dir
from swingvision.app.main import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_pages_registered(app):
    paths = {p.get("path_template") or p["path"] for p in dash.page_registry.values()}
    assert {
        "/",
        "/new",
        "/jobs",
        "/settings",
        "/session/<session_id>",
        "/calibrate/<session_id>",
        "/profiles",
        "/labeling",
        "/stats/<session_id>",
    } <= paths


@pytest.mark.parametrize(
    "url",
    ["/", "/new", "/jobs", "/settings", "/profiles", "/labeling", "/session/nope", "/stats/nope"],
)
def test_pages_render(app, settings, url):
    for page in dash.page_registry.values():
        template = page.get("path_template")
        if page["path"] == url or (template and url.startswith(template.split("<")[0])):
            if template:
                page["layout"](session_id="nope")
            else:
                page["layout"]()
            break
    else:
        pytest.fail(f"no page for {url}")


def test_media_route_only_serves_session_files(app, settings):
    client = app.server.test_client()
    assert client.get("/media/unknown/proxy.mp4").status_code == 404
    assert client.get("/media/unknown/session.json").status_code == 404


@pytest.mark.ffmpeg
def test_media_route_supports_range_requests(app, settings, synthetic_video):
    from swingvision.pipeline.runner import run
    from swingvision.pipeline.stages import default_registry

    session = services.create_session(settings, synthetic_video)
    run(default_registry(), session, settings, targets=["proxy"])
    sid = session.load_config().id
    client = app.server.test_client()
    full = client.get(f"/media/{sid}/proxy.mp4")
    assert full.status_code == 200 and full.mimetype == "video/mp4"
    part = client.get(f"/media/{sid}/proxy.mp4", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.data) == 100
    full.close()
    part.close()


def test_list_dir_filters_and_sorts(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "A_dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "clip.MP4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    entries, err = list_dir(str(tmp_path), "file", (".mp4",))
    assert err is None
    assert [e.name for e in entries] == ["A_dir", "b_dir", "clip.MP4"]
    folders, _ = list_dir(str(tmp_path), "folder")
    assert [e.name for e in folders] == ["A_dir", "b_dir"]
    _, err = list_dir(str(tmp_path / "missing"), "file")
    assert err


def test_breadcrumbs(tmp_path):
    crumbs = _breadcrumb_paths(str(tmp_path))
    assert crumbs[-1][1] == tmp_path.name
    assert crumbs[0][0] == tmp_path.anchor
