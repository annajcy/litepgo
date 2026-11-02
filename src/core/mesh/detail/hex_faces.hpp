#pragma once
#include <array>
namespace litepgo::mesh::detail {
// Outward winding for HexQ1 corner order.
inline constexpr std::array<std::array<int, 4>, 6> hex_faces{
    {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}}};
} // namespace litepgo::mesh::detail
