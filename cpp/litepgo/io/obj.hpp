#pragma once
#include "litepgo/mesh/mesh_data.hpp"
#include <filesystem>

namespace litepgo::io {
TriangleMesh read_obj(const std::filesystem::path& path);
void write_obj(const std::filesystem::path& path, const TriangleMesh& mesh);
} // namespace litepgo::io
