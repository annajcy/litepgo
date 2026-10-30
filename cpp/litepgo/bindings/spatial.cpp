#include "litepgo/bindings/bindings.hpp"
#include "litepgo/spatial/spatial_hash.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::spatial::binding {
void bind(nb::module_& m) {
    nb::class_<AABB>(m, "AABB")
        .def(
            "__init__",
            [](AABB* self, const std::array<double, 3>& lo, const std::array<double, 3>& hi) {
                new (self) AABB{Eigen::Vector3d(lo[0], lo[1], lo[2]), Eigen::Vector3d(hi[0], hi[1], hi[2])};
            },
            "lo"_a, "hi"_a)
        .def_ro("lo", &AABB::lo)
        .def_ro("hi", &AABB::hi);
    nb::class_<SpatialHash>(m, "SpatialHash")
        .def(
            "__init__",
            [](SpatialHash* self, const std::vector<AABB>& boxes, std::optional<double> size) {
                new (self) SpatialHash(boxes, size);
            },
            "boxes"_a, "cell_size"_a = nb::none())
        .def("query", &SpatialHash::query, "region"_a, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "overlapping_pairs",
            [](const SpatialHash& index, nb::object filter) {
                if (!filter.is_none() && !PyCallable_Check(filter.ptr()))
                    throw nb::type_error("filter must be callable or None");
                std::vector<BoxPair> pairs;
                {
                    nb::gil_scoped_release release;
                    pairs = index.overlapping_pairs([](std::size_t, std::size_t) { return true; });
                }
                // Python callbacks run serially with the GIL, after native parallel detection.
                if (!filter.is_none())
                    std::erase_if(pairs, [&](const BoxPair& p) { return !nb::cast<bool>(filter(p[0], p[1])); });
                return pairs;
            },
            "filter"_a = nb::none(), "Unique overlapping pairs. Optional Python filter runs serially after detection.");
}
} // namespace litepgo::spatial::binding
