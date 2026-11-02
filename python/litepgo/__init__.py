"""Geometry processing, finite elements, contact, and optimization.

Import public submodules explicitly, for example ``from litepgo import energy,
fem, solver``. Energies represent scalar objectives; responses represent vector
measurements or residuals. Native implementations provide analytic derivatives
and sparse assembly through NumPy/SciPy-compatible Python interfaces.

Start with the [user manual](/) for data conventions, formulations, and complete
examples. This reference lists the API present in the selected compiled build.
``build_info()`` reports that build's versions and available sparse backends.
"""

from litepgo import _core


def build_info() -> dict:
    """Versions this build was compiled against and its available sparse backends."""
    return _core.build_info()


__all__ = ["build_info"]
