"""Minimal FEM, energy, and Newton solver core for fit-micro."""

from litepgo import _core


def build_info() -> dict:
    """Versions this build was compiled against and its available sparse backends."""
    return _core.build_info()


__all__ = ["build_info"]
