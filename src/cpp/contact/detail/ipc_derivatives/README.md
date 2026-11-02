# IPC Toolkit derivative subset

Source: https://github.com/ipc-sim/ipc-toolkit/tree/v1.6.1/src/ipc/distance
License: MIT, Copyright (c) 2020–2025 Zachary Ferguson; full text in LICENSE.

Copied the 3D gradient/Hessian function bodies from point_line.cpp,
point_plane.cpp, line_line.cpp and edge_edge_mollifier.cpp (cross squared norm
only). Generated arithmetic is unchanged; namespace/includes are adapted.
point_point.cpp is copied with Eigen fixed-size 3D/6-DOF types replacing Toolkit
bounded-dynamic types (explicit dimension casts and a release-build unused-parameter annotation). No Toolkit headers, linking, autograd or other dependencies
are required beyond the project's existing Eigen dependency. Unneeded 2D,
closest-feature dispatch, rest-threshold derivatives and geometry helpers were omitted.

Native feature gathering/scattering and the barrier/mollifier chain/product rules
are implemented separately in distance_derivatives.cpp and ipc_contact_energy.cpp;
those adapters are new code, not copied generated upstream routines.

Original source SHA256:
- point_line.cpp: `9260fc8918bee89e6158a30158848f94f1a20ebc3d95cc4df818f6c7fc98ecac`
- point_plane.cpp: `3fd91b1730d11e3a4e18edae04ab7e79767474d3c93026e3b7068c5fb63e9f67`
- line_line.cpp: `079efbd652f16eb28dcffc463551c499395e24a1bacde42193d97b6926d04158`
- edge_edge_mollifier.cpp: `a8b17ebced68a8c7807ff2838fd65fc654c2c875dfb48293e20d74f02bf48bd9`
