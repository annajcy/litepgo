#include "litepgo/fem/element_locator.hpp"
#include "bindings.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <vector>

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
using bindings::register_instance;

template <class B> void bind_element_locator(nb::module_& m, const char* name) {
    using Locator = ElementLocator<B>;
    nb::class_<Locator>(m, name)
        .def(nb::init<std::vector<std::shared_ptr<energy::Response>>>(), "X_fields"_a)
        .def("locate", &Locator::locate, "points"_a);
    register_instance(m, "ElementLocator", nb::make_tuple(nb::type<B>()), nb::type<Locator>());
}
} // namespace

void bind_fem_element_locator(nb::module_& m) {
    nb::class_<PointEmbedding>(m, "PointEmbedding")
        .def(
            "__init__",
            [](PointEmbedding* self, const Eigen::VectorXi& ids, const Eigen::MatrixXd& xi) {
                if (xi.cols() != 3 || ids.size() != xi.rows() || !xi.allFinite() || (ids.array() < 0).any())
                    throw std::invalid_argument("PointEmbedding requires nonnegative element_ids and finite (n, 3) xi");
                new (self) PointEmbedding{ids, xi};
            },
            "element_ids"_a, "xi"_a)
        .def_ro("element_ids", &PointEmbedding::element_ids)
        .def_ro("xi", &PointEmbedding::xi);
    bind_element_locator<TetP1Basis>(m, "ElementLocatorTetP1");
    bind_element_locator<HexQ1Basis>(m, "ElementLocatorHexQ1");
}
} // namespace litepgo::fem::binding
