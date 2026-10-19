#!/usr/bin/env python3
"""Source bootstrap. Only the dedicated product venv is installed or repaired."""

from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and launch the local Data Audit UI")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Prepare dependencies without starting the server"
    )
    args = parser.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("Install Python 3.11, 3.12 or 3.13 and run this launcher again.")
    root = Path(__file__).resolve().parent
    runtime = Path(os.environ.get("DATA_AUDIT_RUNTIME", str(root / ".audit-runtime"))).resolve()
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    signature = hashlib.sha256(
        (root / "pyproject.toml").read_bytes()
        + (root / "requirements-ui.lock").read_bytes()
        + f"{sys.version_info[:2]}:{root}".encode()
    ).hexdigest()
    marker = runtime / "prepared-version"
    prepared = python.is_file() and marker.is_file() and marker.read_text() == signature
    if prepared:
        check = subprocess.run(
            [str(python), "-c", "import data_audit, streamlit, plotly, openpyxl, xlrd"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        prepared = check.returncode == 0
    if not prepared:
        wheelhouse = os.environ.get("DATA_AUDIT_WHEELHOUSE")
        if args.offline and not wheelhouse:
            parser.error(
                "Offline dependencies are missing or need repair. Prepare once online, or set DATA_AUDIT_WHEELHOUSE to a local wheel directory."
            )
        runtime.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True).create(runtime)
        pip = [str(python), "-m", "pip", "--disable-pip-version-check"]
        flags = ["--no-index", "--find-links", str(Path(wheelhouse).resolve())] if args.offline else []
        try:
            if args.offline:
                subprocess.run(pip + ["install", *flags, "setuptools>=75", "wheel"], check=True, cwd=root)
            subprocess.run(
                pip
                + [
                    "install",
                    *flags,
                    *(["--no-build-isolation"] if args.offline else []),
                    "-c",
                    str(root / "requirements-ui.lock"),
                    "-e",
                    str(root) + "[ui]",
                ],
                check=True,
                cwd=root,
            )
        except subprocess.CalledProcessError:
            print(
                "UI setup failed. Check package access or supply the missing verified wheels. No other Python environment was modified.",
                file=sys.stderr,
            )
            return 1
        marker.write_text(signature)
    if args.prepare_only:
        print(f"Prepared product environment: {runtime}")
        return 0
    command = [str(python), "-m", "data_audit.cli", "ui"]
    for enabled, flag in [(args.offline, "--offline"), (args.no_browser, "--no-browser")]:
        if enabled:
            command.append(flag)
    if args.port is not None:
        command += ["--port", str(args.port)]
    # exec preserves signals, exit status and ownership of the UI child process.
    os.execv(str(python), command)


if __name__ == "__main__":
    raise SystemExit(main())
