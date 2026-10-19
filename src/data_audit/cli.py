"""Local UI server lifecycle with readiness and child-process cleanup."""

from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser


def free_port(port=None):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port or 0))
        except OSError as exc:
            raise ValueError(f"Local port {port} is unavailable; choose another --port") from exc
        return sock.getsockname()[1]


def serve(*, port=None, no_browser=False, environ=None):
    if importlib.util.find_spec("streamlit") is None or importlib.util.find_spec("plotly") is None:
        raise RuntimeError("UI dependencies missing. Run run_ui.py or install data-audit[ui].")
    port = free_port(port)
    root = Path(tempfile.gettempdir()) / "data-audit-logs"
    root.mkdir(exist_ok=True)
    log_path = root / f"ui-{os.getpid()}-{port}.log"
    app = Path(__file__).parent / "ui" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]
    process = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    # SIGTERM/CTRL_BREAK must follow the same cleanup path as Ctrl+C.
    previous = {}
    import threading

    if threading.current_thread() is threading.main_thread():
        for sig in [signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else []):
            previous[sig] = signal.signal(sig, interrupt)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environ, start_new_session=os.name != "nt"
            )
            deadline = time.monotonic() + 30
            url = f"http://127.0.0.1:{port}"
            # Explicit no-proxy opener: localhost health never goes through external proxies.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"UI exited with status {process.returncode}; log: {log_path}")
                try:
                    with opener.open(url + "/_stcore/health", timeout=0.5) as response:
                        if response.status == 200 and response.read() == b"ok":
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError(f"UI readiness timed out; log: {log_path}")
            print(f"Data Audit ready at {url}\nLog: {log_path}", flush=True)
            if not no_browser:
                try:
                    if not webbrowser.open(url):
                        print("Open the address above in your browser.", flush=True)
                except webbrowser.Error:
                    print("Browser unavailable; open the address above.", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="data-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("ui", help="Start the local investigation UI")
    ui.add_argument(
        "--offline", action="store_true", help="Use installed dependencies only; no network installation"
    )
    ui.add_argument("--no-browser", action="store_true")
    ui.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    # Installed CLI never performs installation, including offline mode.
    try:
        return serve(port=args.port, no_browser=args.no_browser)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
