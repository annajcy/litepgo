#pragma once
#include "litepgo/mesh/mesh_data.hpp"
#include <filesystem>

namespace litepgo::io {
VolumeMesh read_veg(const std::filesystem::path& path);
void write_veg(const std::filesystem::path& path, const VolumeMesh& mesh);
} // namespace litepgo::io
