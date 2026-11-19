import pytest

from fake_nba import Env


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake NBA.com + fake fit, with priors/applied/log/cache files in a temp folder (season already started)."""
    return Env(tmp_path / "run", monkeypatch)
