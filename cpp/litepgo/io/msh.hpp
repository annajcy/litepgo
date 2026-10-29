#pragma once
#include "litepgo/mesh/mesh_data.hpp"
#include <filesystem>

namespace litepgo::io {
mesh::VolumeMesh read_msh(const std::filesystem::path& path);
void write_msh(const std::filesystem::path& path, const mesh::VolumeMesh& mesh);
} // namespace litepgo::io
