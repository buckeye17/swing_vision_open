from __future__ import annotations

import dash
import pytest
from pydantic import ValidationError

from swingvision import services


def test_profile_crud(settings):
    p = services.save_profile(settings, " Chris ", "left", "one_handed", 1.83)
    assert p.name == "Chris" and p.handedness == "left" and p.backhand == "one_handed"
    assert p.height_m == pytest.approx(1.83)
    q = services.save_profile(settings, "Chris R", "right", "two_handed", None, profile_id=p.id)
    assert q.id == p.id and q.name == "Chris R" and q.height_m is None
    services.save_profile(settings, "Alex")
    assert [x.name for x in services.list_profiles(settings)] == ["Alex", "Chris R"]
    assert services.get_profile(settings, p.id).handedness == "right"
    assert services.get_profile(settings, None) is None


def test_profile_validation(settings):
    with pytest.raises(ValueError, match="needs a name"):
        services.save_profile(settings, "  ")
    with pytest.raises(ValidationError):
        services.save_profile(settings, "Tall", height_m=3.1)
    with pytest.raises(ValidationError):
        services.save_profile(settings, "Odd", handedness="both")
    with pytest.raises(ValueError, match="Unknown profile"):
        services.save_profile(settings, "Ghost", profile_id="nope")
    assert services.list_profiles(settings) == []


@pytest.mark.ffmpeg
def test_session_profile_assignment_and_delete(settings, synthetic_video):
    p = services.save_profile(settings, "Chris")
    session = services.create_session(settings, synthetic_video, me_profile_id=p.id)
    sid = session.load_config().id
    assert session.load_config().players.me_profile_id == p.id
    services.set_session_player(settings, sid, None)
    assert session.load_config().players.me_profile_id is None
    services.set_session_player(settings, sid, p.id)
    with pytest.raises(ValueError):
        services.set_session_player(settings, sid, "missing")
    assert services.delete_profile(settings, p.id) == 1
    assert session.load_config().players.me_profile_id is None
    assert services.list_profiles(settings) == []


def test_profiles_page_renders(settings):
    from swingvision.app.main import create_app

    create_app()
    services.save_profile(settings, "Chris", height_m=1.8)
    page = next(p for p in dash.page_registry.values() if p["path"] == "/profiles")
    page["layout"]()
    from swingvision.app.pages.profiles import _render

    cards = _render(0)
    assert "Chris" in str(cards) and "180 cm" in str(cards)
