"""Install the locked documentation dependencies, then build or serve the site."""

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dev", "build", "preview"), nargs="?", default="dev")
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parents[1]
    pnpm = ["uv", "run", "--locked", "--group", "docs", "corepack", "pnpm"]
    try:
        subprocess.run([*pnpm, "install", "--frozen-lockfile"], cwd=directory, check=True)
        subprocess.run([*pnpm, arguments.command], cwd=directory, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
