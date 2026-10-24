#pragma once

#include <concepts>

namespace litepgo {

// Point contains all arguments. Each derivative object selects one argument;
// its gradient/Hessian hold all other arguments fixed. These concepts check
// signatures, not mathematical correctness, layout, or thread safety.
template <class E>
concept PointValue = requires(const E& energy, const typename E::Point& point) {
    { energy.value(point) } -> std::same_as<double>;
};

template <class E>
concept PointFirstOrderEnergy = PointValue<E> && requires(const E& energy, const typename E::Point& point) {
    { energy.gradient(point) } -> std::same_as<typename E::Gradient>;
};

// Independent of first-order capability: Hessian-only objects are allowed.
template <class E>
concept PointSecondOrderEnergy = PointValue<E> && requires(const E& energy, const typename E::Point& point) {
    { energy.hessian(point) } -> std::same_as<typename E::Hessian>;
};

} // namespace litepgo
