import pytest

from fake_nba import Env


@pytest.fixture(autouse=True)
def no_real_betting_lines(tmp_path, monkeypatch):
    """
    Every test: no ODDS_API_KEY, The Odds API can't be reached (lines._download_odds raises), and
    the lines files point at a temp folder, so update.py --update (--lines on by default) never
    spends a request or writes data/. test_lines.py swaps in its own fake download.
    """
    import lines
    monkeypatch.delenv(lines.KEY_ENV, raising=False)
    folder = tmp_path / "betting_lines"
    monkeypatch.setattr(lines, "LINES_PATH", folder / "lines.csv")
    monkeypatch.setattr(lines, "UNMATCHED_PATH", folder / "lines_unmatched.csv")
    monkeypatch.setattr(lines, "RAW_DIR", folder / "raw")

    def refuse(key):
        raise AssertionError("tests must never call The Odds API")

    monkeypatch.setattr(lines, "_download_odds", refuse)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake NBA.com + fake fit, with priors/applied/log/cache files in a temp folder (season already started)."""
    return Env(tmp_path / "run", monkeypatch)
