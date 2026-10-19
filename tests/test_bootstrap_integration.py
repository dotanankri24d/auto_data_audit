"""Full offline source bootstrap, enabled when a prepared wheelhouse is supplied."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import pytest
from data_audit.cli import free_port

ROOT = Path(__file__).parents[1]


def test_offline_first_install_unicode_paths_and_restart(tmp_path):
    wheelhouse = os.environ.get("DATA_AUDIT_TEST_WHEELHOUSE")
    if not wheelhouse:
        pytest.skip("Set DATA_AUDIT_TEST_WHEELHOUSE for full offline bootstrap integration")
    checkout = tmp_path / "פרויקט עם רווחים"
    shutil.copytree(
        ROOT,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git",
            ".audit-runtime",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            "*.egg-info",
        ),
    )
    guard = tmp_path / "network-guard"
    guard.mkdir()
    attempted = tmp_path / "external-attempts.txt"
    (guard / "sitecustomize.py").write_text(
        "import socket, os\n_original=socket.socket.connect\n"
        "def connect(self, address):\n"
        '    if isinstance(address,tuple) and address[0] not in ("127.0.0.1","localhost","::1"):\n'
        '        with open(os.environ["ATTEMPT_LOG"],"a") as log: log.write(str(address)+"\\n")\n'
        '        raise OSError("External network disabled by offline acceptance test")\n'
        "    return _original(self,address)\n"
        "socket.socket.connect=connect\n"
    )
    environment = os.environ.copy()
    environment.update(
        DATA_AUDIT_RUNTIME=str(tmp_path / "סביבת מוצר"),
        DATA_AUDIT_WHEELHOUSE=wheelhouse,
        PYTHONPATH=str(guard),
        ATTEMPT_LOG=str(attempted),
        PIP_NO_INDEX="1",
    )
    first = subprocess.run(
        [sys.executable, str(checkout / "run_ui.py"), "--offline", "--prepare-only"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    # With a valid runtime, wheelhouse is unnecessary and no pip process should run.
    environment.pop("DATA_AUDIT_WHEELHOUSE")
    repeat = subprocess.run(
        [sys.executable, str(checkout / "run_ui.py"), "--offline", "--prepare-only"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert repeat.returncode == 0, repeat.stdout + repeat.stderr
    assert "Installing" not in repeat.stdout and "pip" not in repeat.stdout
    port = free_port()
    log = (tmp_path / "source-ui.log").open("w")
    command = [sys.executable, str(checkout / "run_ui.py")]
    if os.name != "nt":
        command = ["sh", str(checkout / "run_ui.sh")]
    process = subprocess.Popen(
        command + ["--offline", "--no-browser", "--port", str(port)],
        cwd=tmp_path,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    start = time.monotonic()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() - start < 30:
            try:
                with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.3) as response:
                    if response.read() == b"ok":
                        break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("Offline source UI did not become ready")
        print(f"Prepared offline source startup seconds: {time.monotonic() - start:.3f}")
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
            process.wait(timeout=10)
        log.close()
    assert not attempted.exists(), attempted.read_text() if attempted.exists() else ""
