"""Reusable AABB spatial queries, independent of contact primitive types.

SpatialHash owns copies of its input boxes. IDs are input indices. Query results
are unique and unordered. Python pair filters run serially after native detection.
The cell size is a starting size: grids exceeding the estimated memory budget
are coarsened. This changes search cost, not the returned overlaps.
"""
from litepgo._core.spatial import AABB, SpatialHash

__all__ = ["AABB", "SpatialHash"]
