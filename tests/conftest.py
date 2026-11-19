import pytest


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path, monkeypatch):
    """Never touch the real user's config folder / recent-projects list."""
    monkeypatch.setenv("BOXTROT_CONFIG_DIR", str(tmp_path / "_config"))
    from boxtrot.core import project as pj
    monkeypatch.setattr(pj, "RECENT_PATH", None)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def check(cond, msg):
    assert cond, msg
