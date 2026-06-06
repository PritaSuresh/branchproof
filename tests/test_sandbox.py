import os
import subprocess

import pytest

from branchproof.runner import run_pytest
from branchproof.sandbox import container_command, execution_image, resolve_image


def test_container_has_no_mounts_secrets_or_network():
    command = container_command("test-image", "test-container")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--user=65534:65534" in command
    assert "--pids-limit=128" in command
    assert not any(c in {"-v", "--volume", "--mount", "--env", "--env-file"} for c in command)


def test_image_resolution_uses_explicit_image_then_format(monkeypatch):
    from types import SimpleNamespace

    def inspect(command, **kwargs):
        assert command == ["docker", "image", "inspect", "trusted:tag", "--format", "{{.Id}}"]
        return SimpleNamespace(returncode=0, stdout="sha256:" + "a" * 64 + "\n")

    monkeypatch.setattr("branchproof.sandbox.subprocess.run", inspect)
    assert resolve_image("trusted:tag") == "sha256:" + "a" * 64


def test_broken_sandbox_does_not_fall_back_to_host(tmp_path, env, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Docker unavailable")

    monkeypatch.setattr("branchproof.runner.run_container", unavailable)
    with execution_image("missing-image"):
        result = run_pytest(tmp_path, env)
    assert not result.passed
    assert "Sandbox" in result.report_error


@pytest.mark.skipif(not os.getenv("HARDENER_DOCKER_TESTS"), reason="opt-in real Docker test")
def test_real_container_isolated_and_removed(tmp_path, env, monkeypatch):
    sentinel = tmp_path.parent / "host-only-sentinel.txt"
    sentinel.write_text("must not be visible")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-host-secret")
    (tmp_path / ".env").write_text("DUMMY=private")
    (tmp_path / "test_isolation.py").write_text(
        "import os, pathlib, socket\n"
        "def test_isolated():\n"
        "    assert 'AZURE_OPENAI_API_KEY' not in os.environ\n"
        "    assert not pathlib.Path('.env').exists()\n"
        f"    assert not pathlib.Path({str(sentinel)!r}).exists()\n"
        "    assert os.getuid() != 0\n"
        "    import pytest\n"
        "    with pytest.raises(OSError):\n"
        "        socket.create_connection(('1.1.1.1', 443), timeout=1)\n"
    )
    image = os.getenv("HARDENER_SANDBOX_IMAGE", "hardener-sandbox:py312")
    with execution_image(image):
        result = run_pytest(tmp_path, env, timeout=20)
    assert result.passed, result.tail()
    assert sentinel.read_text() == "must not be visible"
    containers = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=hardener-"],
        capture_output=True, text=True, check=True,
    )
    assert not containers.stdout.strip()


@pytest.mark.skipif(not os.getenv("HARDENER_DOCKER_TESTS"), reason="opt-in real Docker test")
def test_container_timeout_kills_child_processes(tmp_path, env):
    (tmp_path / "test_timeout.py").write_text(
        "import subprocess, time\n"
        "def test_timeout():\n"
        "    subprocess.Popen(['sleep', '120'])\n"
        "    time.sleep(120)\n"
    )
    with execution_image(os.getenv("HARDENER_SANDBOX_IMAGE", "hardener-sandbox:py312")):
        result = run_pytest(tmp_path, env, timeout=1)
    assert result.timed_out
    assert not result.passed
    assert not subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", "name=hardener-"], text=True
    ).strip()
