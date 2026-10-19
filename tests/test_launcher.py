import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import pytest
from data_audit.cli import free_port

ROOT = Path(__file__).parents[1]


def wait_health(port, process):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"Launcher exited early: {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.3) as r:
                if r.read() == b"ok":
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise AssertionError("UI never became ready")


def test_installed_cli_offline_from_other_directory(tmp_path):
    port = free_port()
    with (tmp_path / "server.log").open("w+") as log:
        p = subprocess.Popen(
            [sys.executable, "-m", "data_audit.cli", "ui", "--offline", "--no-browser", "--port", str(port)],
            cwd=tmp_path,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            wait_health(port, p)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"http://127.0.0.1:{port}", timeout=1) as response:
                assert response.status == 200 and b"Streamlit" in response.read()
            p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            assert p.wait(timeout=10) == 0
            # The child server must not remain listening after launcher exit.
            with socket.socket() as s:
                assert s.connect_ex(("127.0.0.1", port)) != 0
        finally:
            if p.poll() is None:
                if os.name == "nt":
                    p.terminate()
                else:
                    os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=10)


def test_source_offline_missing_dependency_diagnostic(tmp_path):
    environment = os.environ.copy()
    environment["DATA_AUDIT_RUNTIME"] = str(tmp_path / "סביבה עם רווחים")
    environment.pop("DATA_AUDIT_WHEELHOUSE", None)
    p = subprocess.run(
        [sys.executable, str(ROOT / "run_ui.py"), "--offline", "--prepare-only"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert p.returncode != 0 and "Offline dependencies are missing" in p.stderr
    assert not Path(environment["DATA_AUDIT_RUNTIME"]).exists()


def test_port_collision():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(ValueError, match="unavailable"):
            free_port(s.getsockname()[1])
