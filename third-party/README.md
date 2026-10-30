# Third-party attribution

Tinyobjloader and optional Gmsh are fetched by CMake with pinned archive hashes;
see `cmake/third-party/`. No local SDK is required.

IPC Toolkit 1.6.1 (MIT) is fetched with a pinned archive hash; see
`cmake/third-party/ipc_toolkit.cmake`. Its own dependencies (libigl,
Tight-Inclusion, scalable-ccd, spdlog, TinyAD, abseil, robin-map, filib) are
pinned by tag or commit in its CPM recipes, without archive hashes. Planes are
handled by litepgo itself: IPC Toolkit 1.6.1 gathers plane-vertex candidates
within half the barrier's reach.

On x86-64 Linux, Intel oneMKL 2026.1 is fetched as the pinned `mkl` and
`mkl-include` wheels from PyPI, published by Intel under the Intel Simplified
Software License; see `cmake/third-party/mkl.cmake`.

VEG parsing is adapted from libpgo commit
`6aedf568c78b06b375c3ef3a8f21a09c6dbce3c4`; its license is retained as
`libpgo-LICENSE`. The Gmsh CMake recipe is adapted from the same checkout.
