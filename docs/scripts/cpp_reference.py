"""Generate the C++ API reference from litepgo's public headers."""

from pathlib import Path
import subprocess


def main() -> int:
    docs = Path(__file__).resolve().parents[1]
    (docs / "public/reference/cpp").mkdir(parents=True, exist_ok=True)
    return subprocess.run(["doxygen", "Doxyfile"], cwd=docs).returncode


if __name__ == "__main__":
    raise SystemExit(main())
