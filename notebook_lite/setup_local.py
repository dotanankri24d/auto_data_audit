"""Install dependencies into this folder only. Python 3.11–3.13 required."""

from pathlib import Path
import argparse
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use only the wheels in this folder")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise SystemExit("Use Python 3.11–3.13")
    folder = Path(__file__).resolve().parent
    environment = folder / ".venv"
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    wheels = folder / "wheels"
    wheels.mkdir(exist_ok=True)
    pip = [str(python), "-m", "pip", "--cache-dir", str(folder / ".pip-cache")]
    lock = folder / "requirements.lock.txt"
    requirements = ["-r", str(lock if lock.exists() else folder / "requirements.txt")]
    if not args.offline:
        subprocess.run(pip + ["download", "--dest", str(wheels)] + requirements, check=True)
    subprocess.run(pip + ["install", "--no-index", "--find-links", str(wheels)] + requirements, check=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(environment),
            "--name",
            "python3",
            "--display-name",
            "Python (notebook_lite)",
        ],
        check=True,
    )
    print(f'Open the notebook: "{python}" -m jupyterlab "{folder / "analysis.ipynb"}"')


if __name__ == "__main__":
    main()
