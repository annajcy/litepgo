#pragma once

#include <filesystem>
#include <vector>

namespace litepgo::mesh {
void write_abc_from_objs(const std::filesystem::path& path, const std::vector<std::filesystem::path>& frame_paths,
                         double fps);
} // namespace litepgo::mesh
