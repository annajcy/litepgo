#pragma once
#include "litepgo/fem/element_locator.hpp"
#include "litepgo/mesh/mesh_data.hpp"

namespace litepgo::fem {
struct SampledBoundary {
    mesh::TriangleMesh mesh;
    PointEmbedding embedding;
};
// Uniform reference-face sampling; geometry must be conforming and positively oriented.
// Shared points are welded by topology, with the first (lowest-ID) boundary host retained.
// subdivisions=1 preserves extract_surface's vertex order, winding and diagonals.
template <class GeometryBasis>
SampledBoundary sample_boundary(const mesh::HexTopology& topology,
                                const std::vector<InterpolatedField3D<GeometryBasis>>& X_fields, int subdivisions);

template <>
SampledBoundary sample_boundary<HexQ1Basis>(const mesh::HexTopology& topology,
                                            const std::vector<InterpolatedField3D<HexQ1Basis>>& X_fields,
                                            int subdivisions);
} // namespace litepgo::fem
