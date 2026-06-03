import subprocess
import sys
from pathlib import Path

import pytest

BANK = '''def can_withdraw(balance, amount):
    if amount <= 0:
        return False
    return balance >= amount


def fee(amount):
    return amount * 2
'''

EXISTING_TESTS = '''from bank import can_withdraw, fee


def test_withdraw_more_than_balance():
    assert can_withdraw(100, 500) is False


def test_withdraw_less_than_balance():
    assert can_withdraw(100, 50) is True


def test_fee():
    assert fee(10) == 20
'''


@pytest.fixture(scope="session")
def env() -> Path:
    """Reuse this project's own virtualenv. It has pytest, which is all a fixture repo needs."""
    return Path(sys.executable).parent.parent


@pytest.fixture
def sample_repo(tmp_path_factory) -> Path:
    """A tiny git repo whose tests miss the exact balance boundary."""
    repo = tmp_path_factory.mktemp("sample")
    (repo / "bank.py").write_text(BANK)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_bank.py").write_text(EXISTING_TESTS)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "start"],
        cwd=repo, check=True,
    )
    return repo
