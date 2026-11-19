import pytest


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXTROT_CONFIG_DIR", str(tmp_path / "_config"))
