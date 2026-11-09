#include "litepgo/mesh/io/alembic.hpp"
#include "litepgo/mesh/io/obj.hpp"

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <gtest/gtest.h>

#include <random>

namespace {
namespace abc = Alembic::Abc;
namespace geom = Alembic::AbcGeom;
namespace fs = std::filesystem;
using namespace litepgo::mesh;

class AlembicExport : public testing::Test {
  protected:
    const fs::path directory =
        fs::path(testing::TempDir()) / ("litepgo-abc-test-" + std::to_string(std::random_device{}()));

    void SetUp() override {
        ASSERT_TRUE(fs::create_directory(directory));
    }

    void TearDown() override {
        fs::remove_all(directory);
    }
};

TEST_F(AlembicExport, ReadsBackFramesInInputOrder) {
    const std::vector<fs::path> frames{directory / "frame_1.obj", directory / "frame_2.obj",
                                       directory / "frame_10.obj"};
    const std::vector<float> heights{0.25f, -0.5f, 1.5f};
    const TriangleTopology topology(4, {{0, 1, 2}, {1, 3, 2}});
    for (std::size_t i = 0; i < frames.size(); ++i) {
        litepgo::Vec xyz(12);
        xyz << 0., 0., heights[i], 1., 0., heights[i], 0., 1., heights[i], 1., 1., heights[i];
        write_obj(frames[i], {xyz, topology});
    }
    const auto path = directory / "mesh.abc";
    write_abc_from_objs(path, frames, 24.0);

    const abc::IArchive archive(Alembic::AbcCoreOgawa::ReadArchive(), path.string());
    const geom::IPolyMesh object(archive.getTop(), "mesh");
    const auto& schema = object.getSchema();
    ASSERT_EQ(schema.getNumSamples(), frames.size());
    EXPECT_EQ(schema.getTopologyVariance(), geom::kHomogenousTopology);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        EXPECT_DOUBLE_EQ(schema.getTimeSampling()->getSampleTime(i), static_cast<double>(i) / 24.0);
        const auto sample = schema.getValue(abc::ISampleSelector(static_cast<abc::index_t>(i)));
        ASSERT_EQ(sample.getPositions()->size(), 4);
        EXPECT_EQ((*sample.getPositions())[0], abc::V3f(0., 0., heights[i]));
        EXPECT_EQ((*sample.getPositions())[3], abc::V3f(1., 1., heights[i]));
        ASSERT_EQ(sample.getFaceCounts()->size(), 2);
        EXPECT_EQ((*sample.getFaceCounts())[0], 3);
        EXPECT_EQ((*sample.getFaceCounts())[1], 3);
        ASSERT_EQ(sample.getFaceIndices()->size(), 6);
        const auto* indices = sample.getFaceIndices()->get();
        EXPECT_EQ((std::vector<abc::int32_t>(indices, indices + 6)), (std::vector<abc::int32_t>{2, 1, 0, 2, 3, 1}));
    }
}
} // namespace
