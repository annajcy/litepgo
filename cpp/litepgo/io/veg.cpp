// VEG parsing adapted from libpgo; see third-party/libpgo-LICENSE.
#include "litepgo/io/veg.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace litepgo::io {
namespace {
std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

bool startsWith(const std::string& text, const char* prefix) {
    return text.rfind(prefix, 0) == 0;
}

// includeChain holds the files being read around this one, to catch *INCLUDE cycles.
std::vector<std::string> readAsciiLines(const std::filesystem::path& path,
                                        std::vector<std::filesystem::path> includeChain = {}) {
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path);
    if (std::find(includeChain.begin(), includeChain.end(), canonical) != includeChain.end())
        throw std::runtime_error("veg *INCLUDE cycle through " + path.string());
    includeChain.push_back(canonical);

    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Failed to open veg file: " + path.string());

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        if (startsWith(line, "*INCLUDE ")) {
            const auto includePath = path.parent_path() / trim(line.substr(9));
            auto includeLines = readAsciiLines(includePath, includeChain);
            lines.insert(lines.end(), std::make_move_iterator(includeLines.begin()),
                         std::make_move_iterator(includeLines.end()));
            continue;
        }

        lines.push_back(std::move(line));
    }
    if (in.bad())
        throw std::runtime_error("Failed reading VEG: " + path.string());
    return lines;
}

std::vector<double> parseDoubles(std::string text) {
    for (char& ch : text) {
        if (ch == ',' || ch == '\t')
            ch = ' ';
    }
    std::istringstream in(text);
    std::vector<double> values;
    double value;
    while (in >> value)
        values.push_back(value);
    if (!in.eof())
        throw std::runtime_error("Malformed VEG numeric record: " + text);
    return values;
}

std::vector<int> parseInts(std::string text) {
    for (char& ch : text) {
        if (ch == ',' || ch == '\t')
            ch = ' ';
    }
    std::istringstream in(text);
    std::vector<int> values;
    int value;
    while (in >> value)
        values.push_back(value);
    if (!in.eof())
        throw std::runtime_error("Malformed VEG numeric record: " + text);
    return values;
}

std::string compact(std::string text) {
    text.erase(std::remove_if(text.begin(), text.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }),
               text.end());
    return text;
}

template <int K> MeshData<K> makeMesh(const std::vector<Eigen::Vector3d>& vertices, const std::vector<int>& elements) {
    if (vertices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("VEG vertex count exceeds int range");
    Vec positions(static_cast<Eigen::Index>(vertices.size()) * 3);
    for (std::size_t i = 0; i < vertices.size(); ++i)
        positions.segment<3>(static_cast<Eigen::Index>(i) * 3) = vertices[i];
    std::vector<typename MeshTopology<K>::Cell> cells(elements.size() / K);
    for (std::size_t e = 0; e < cells.size(); ++e)
        std::copy_n(elements.begin() + static_cast<std::ptrdiff_t>(e * K), K, cells[e].begin());
    MeshData<K> mesh{std::move(positions), MeshTopology<K>(static_cast<int>(vertices.size()), std::move(cells))};
    mesh.validate();
    return mesh;
}

VolumeMesh readGeometry(const std::filesystem::path& path) {
    const std::vector<std::string> lines = readAsciiLines(path);
    std::vector<Eigen::Vector3d> vertices;
    std::vector<int> elements;
    int elementWidth = 0;
    int oneIndexedVertices = 1;
    bool hasVertices = false;

    size_t i = 0;
    while (i < lines.size()) {
        const std::string line = trim(lines[i]);

        if (startsWith(line, "*VERTICES")) {
            if (hasVertices)
                throw std::runtime_error("Duplicate *VERTICES section in " + path.string());
            hasVertices = true;
            if (++i >= lines.size())
                throw std::runtime_error("Missing *VERTICES header in " + path.string());
            const std::vector<int> header = parseInts(lines[i++]);
            if (header.size() < 2 || header[0] < 0 || header[1] != 3)
                throw std::runtime_error("Invalid 3D vertex header in " + path.string());
            if (static_cast<size_t>(header[0]) > lines.size() - i)
                throw std::runtime_error("Truncated vertex data in " + path.string());
            vertices.reserve(static_cast<size_t>(header[0]));
            for (int row = 0; row < header[0] && i < lines.size(); ++row, ++i) {
                const std::vector<double> values = parseDoubles(lines[i]);
                if (values.size() < 4)
                    throw std::runtime_error("Malformed vertex line in " + path.string() + ": " + lines[i]);
                if (row == 0 && values[0] == 0)
                    oneIndexedVertices = 0;
                if (values[0] != row + oneIndexedVertices)
                    throw std::runtime_error("Vertex IDs must be consecutive in " + path.string());
                vertices.emplace_back(values[1], values[2], values[3]);
            }
            continue;
        }

        if (startsWith(line, "*ELEMENTS")) {
            if (!hasVertices || elementWidth)
                throw std::runtime_error("Expected one *VERTICES section followed by one *ELEMENTS section in " +
                                         path.string());
            if (++i >= lines.size())
                throw std::runtime_error("Missing element type in " + path.string());
            const std::string type = compact(lines[i++]);
            if (type == "TET")
                elementWidth = 4;
            else if (type == "CUBIC")
                elementWidth = 8;
            else
                throw std::runtime_error("Unsupported element type in " + path.string() + ": " + type);

            if (i >= lines.size())
                throw std::runtime_error("Missing element count in " + path.string());
            const std::vector<int> header = parseInts(lines[i++]);
            if (header.empty() || header[0] < 0)
                throw std::runtime_error("Invalid element count in " + path.string());
            if (static_cast<size_t>(header[0]) > lines.size() - i)
                throw std::runtime_error("Truncated element data in " + path.string());
            if (header.size() < 2 || header[1] != elementWidth)
                throw std::runtime_error("Element width does not match its type in " + path.string());
            elements.reserve(static_cast<size_t>(header[0]) * elementWidth);
            for (int row = 0; row < header[0] && i < lines.size(); ++row, ++i) {
                const std::vector<int> values = parseInts(lines[i]);
                if (static_cast<int>(values.size()) < elementWidth + 1)
                    throw std::runtime_error("Malformed element line in " + path.string() + ": " + lines[i]);
                for (int local = 0; local < elementWidth; ++local) {
                    const int id = values[static_cast<size_t>(local + 1)];
                    if (id < oneIndexedVertices || static_cast<size_t>(id - oneIndexedVertices) >= vertices.size())
                        throw std::runtime_error("Element vertex index out of range in " + path.string());
                    elements.push_back(id - oneIndexedVertices);
                }
            }
            continue;
        }

        ++i; // Ignore legacy material, set, region, and other non-geometry records.
    }

    if (elementWidth == 4)
        return makeMesh<4>(vertices, elements);
    else if (elementWidth == 8)
        return makeMesh<8>(vertices, elements);
    else
        throw std::runtime_error("No *ELEMENTS section found in " + path.string());
}
} // namespace

VolumeMesh read_veg(const std::filesystem::path& path) {
    return readGeometry(path);
}

void write_veg(const std::filesystem::path& path, const VolumeMesh& mesh) {
    std::visit([](const auto& data) { data.validate(); }, mesh);
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Cannot write VEG: " + path.string());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::visit(
        [&](const auto& data) {
            constexpr int K = std::decay_t<decltype(data.topology)>::vertices_per_cell;
            out << "*VERTICES\n" << data.topology.num_vertices() << " 3 0 0\n";
            for (int v = 0; v < data.topology.num_vertices(); ++v) {
                out << v + 1;
                for (int c = 0; c < 3; ++c)
                    out << ' ' << data.positions[3 * Eigen::Index(v) + c];
                out << '\n';
            }
            out << "*ELEMENTS\n"
                << (K == 4 ? "TET" : "CUBIC") << '\n'
                << data.topology.num_cells() << ' ' << K << " 0\n";
            for (int e = 0; e < data.topology.num_cells(); ++e) {
                out << e + 1;
                for (int v : data.topology.cell_vertices(e))
                    out << ' ' << v + 1;
                out << '\n';
            }
        },
        mesh);
    out.close();
    if (!out)
        throw std::runtime_error("Failed writing VEG: " + path.string());
}
} // namespace litepgo::io
