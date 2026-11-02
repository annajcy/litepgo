"""Point, region and filtered pair queries; mirrors spatial_query.cpp.

Run from the repository root after building litepgo:
    .venv/bin/python examples/spatial/spatial_query.py
"""

from litepgo import spatial


def main():
    boxes = [
        spatial.AABB([0.2, 0.2, 0], [1.8, 0.8, 1]),
        spatial.AABB([1.7, 0.2, 0], [1.9, 0.8, 1]),
    ]
    index = spatial.SpatialHash(boxes, cell_size=1)
    hits = index.query(spatial.AABB([0.9, 0.3, 0.2], [1.2, 0.6, 0.8]))
    point = index.query(spatial.AABB([1.75, 0.5, 0.5], [1.75, 0.5, 0.5]))
    pairs = index.overlapping_pairs(lambda i, j: i != j)
    assert hits == [0] and sorted(point) == [0, 1] and pairs == [[0, 1]]
    print(f"region_hits={len(hits)}, point_hits={len(point)}, pairs={len(pairs)}")


if __name__ == "__main__":
    main()
