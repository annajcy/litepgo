"""Geometry processing, finite elements, contact, and optimization.

Import public submodules explicitly, for example ``from litepgo import energy,
fem, solver``. Energies represent scalar objectives; responses represent vector
measurements or residuals. Native implementations provide analytic derivatives
and sparse assembly through NumPy/SciPy-compatible Python interfaces.

Start with the [user manual](../../index.html) for data conventions, formulations, and complete
examples. This reference lists the API present in the selected compiled build.
``build_info()`` reports that build's versions and available sparse backends.
"""

from contextlib import contextmanager

from litepgo import _core


def build_info() -> dict:
    """Versions this build was compiled against and its available sparse backends."""
    return _core.build_info()


@contextmanager
def tbb_thread_limit(threads: int):
    """Temporarily limit parallelism in the shared TBB runtime.

    ``threads`` must be a positive integer (not a bool). Overlapping limits
    take the minimum, including limits installed by other libraries. Exiting
    the context removes this limit, even on an exception. The effect is shared
    across threads, not local to this Python thread.

    Covers litepgo's TBB work and its MKL/PARDISO backend, but not Accelerate,
    OpenMP or independent NumPy/SciPy thread pools. Without a limit, TBB uses
    its default concurrency or the host application's existing controls.
    """
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError("threads must be a positive integer")
    if threads <= 0:
        raise ValueError("threads must be positive")
    limit = _core._TbbThreadLimit(threads)
    try:
        yield
    finally:
        limit.close()


__all__ = ["build_info", "tbb_thread_limit"]
