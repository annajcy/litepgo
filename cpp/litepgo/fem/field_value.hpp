#pragma once

#include <Eigen/Core>
#include <concepts>
#include <type_traits>

namespace litepgo {

// Owning fixed-size double matrices/vectors. Empty parameter vectors are valid
// local values; global field containers additionally require a nonempty value.
template <class T>
concept FixedFieldValue = requires {
    typename T::Scalar;
    typename T::PlainObject;
    requires std::same_as<typename T::Scalar, double>;
    requires std::same_as<T, typename T::PlainObject>;
    requires std::is_base_of_v<Eigen::MatrixBase<T>, T>;
    requires(T::RowsAtCompileTime >= 0 && T::ColsAtCompileTime > 0);
};

} // namespace litepgo
