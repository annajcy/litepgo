#pragma once
#include "litepgo/mesh/mesh_data.hpp"
#include <filesystem>

namespace litepgo::io {
VolumeMesh read_msh(const std::filesystem::path& path);
void write_msh(const std::filesystem::path& path, const VolumeMesh& mesh);
} // namespace litepgo::io
