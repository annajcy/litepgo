"""Plot the complete active set and a separate guide to reading individual contact records."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from litepgo import mesh

COLORS = {"VV": "#ae3378", "VE": "#cc7900", "VF": "#197b91", "EE": "#5054b0"}
TITLES = ["Vertex–vertex", "Vertex–edge", "Vertex–face", "Near-parallel edges", "Parallel edges"]


def closest_points(kind, p):
    def on_edge(point, a, b):
        edge = b - a
        return a + np.clip(np.dot(point - a, edge) / np.dot(edge, edge), 0, 1) * edge

    if kind == "VV":
        return p[:2]
    if kind == "VE":
        return np.array([p[0], on_edge(p[0], p[1], p[2])])
    if kind == "VF":
        normal = np.cross(p[2] - p[1], p[3] - p[1])
        return np.array([p[0], p[0] - normal * np.dot(p[0] - p[1], normal) / np.dot(normal, normal)])
    # Mollified EE can have an endpoint closest feature, including parallel edges.
    candidates = [(point, on_edge(point, p[2], p[3])) for point in p[:2]]
    candidates += [(on_edge(point, p[0], p[1]), point) for point in p[2:]]
    parameters, _, rank, _ = np.linalg.lstsq(np.column_stack((p[1] - p[0], p[2] - p[3])),
                                            p[2] - p[0], rcond=None)
    if rank == 2 and np.all((parameters >= 0) & (parameters <= 1)):
        candidates.append((p[0] + parameters[0] * (p[1] - p[0]),
                           p[2] + parameters[1] * (p[3] - p[2])))
    return np.array(min(candidates, key=lambda pair: np.linalg.norm(pair[1] - pair[0])))


def draw_record(axis, kind, p, mollifier, detail=False):
    color = COLORS[kind]
    alpha = .4 if mollifier == 0 and not detail else .9
    if kind == "VE":
        axis.plot(*p[1:].T, color=color, alpha=alpha, linewidth=3)
    elif kind == "VF":
        axis.add_collection3d(Poly3DCollection([p[1:]], facecolors=color, edgecolors=color, alpha=.25))
    elif kind == "EE":
        for edge in (p[:2], p[2:]):
            axis.plot(*edge.T, color=color, alpha=alpha, linewidth=3)
    axis.scatter(*p.T, color=color, alpha=alpha, s=30 if detail else 16)
    ends = closest_points(kind, p)
    axis.plot(*ends.T, color="black" if detail else color, alpha=alpha, linestyle="--", linewidth=2)
    if detail:
        axis.scatter(*ends.T, color="#e9b700", edgecolors="black", s=35, depthshade=False)


def frame(axis, points):
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    span = np.maximum(high - low, .45 * (high - low).max())
    for set_limit, midpoint, width in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), center, span):
        set_limit(midpoint - .65 * width, midpoint + .65 * width)
    axis.set_box_aspect(span)
    for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate.set_major_locator(MaxNLocator(3))
    axis.view_init(elev=24, azim=-62)
    axis.tick_params(labelsize=9, pad=1)
    axis.set_xlabel("x (local)", labelpad=0, fontsize=9)
    axis.set_ylabel("y", labelpad=0, fontsize=9)
    axis.set_zlabel("z", labelpad=0, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="output directory from contact_snapshot")
    parser.add_argument("-o", "--output", type=Path, help="PNG path (default: input/contacts.png)")
    args = parser.parse_args()
    surface = mesh.read_obj(args.input / "mesh.obj")
    with (args.input / "contacts.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["type", "v0", "v1", "v2", "v3", "distance", "weight", "mollifier"]:
            parser.error("expected contact_snapshot CSV columns")
        records = list(reader)
    if surface.positions.shape != (30, 3) or surface.cells.shape != (10, 3):
        parser.error("expected the example's five triangle pairs")
    groups = [[] for _ in range(5)]
    for row, record in enumerate(records):
        kind = record["type"]
        ids = np.array([int(record[f"v{k}"]) for k in range(4)])
        n = {"VV": 2, "VE": 3, "VF": 4, "EE": 4}.get(kind, 0)
        numbers = np.array([float(record[key]) for key in ("distance", "weight", "mollifier")])
        if (not n or np.any(ids[:n] < 0) or np.any(ids[:n] >= 30) or np.any(ids[n:] != -1)
                or not np.isfinite(numbers).all() or numbers[0] < 0 or numbers[1] <= 0
                or not 0 <= numbers[2] <= 1 or len(set(ids[:n] // 6)) != 1):
            parser.error("invalid contact record")
        ends = closest_points(kind, surface.positions[ids[:n]])
        if not np.isclose(np.linalg.norm(ends[1] - ends[0]), numbers[0], rtol=1e-9, atol=1e-12):
            parser.error("plotted closest distance disagrees with the snapshot")
        groups[ids[0] // 6].append((kind, ids[:n], numbers, row))
    if not all(groups):
        parser.error("expected active records in all five example scenes")

    figure = plt.figure(figsize=(12, 8), layout="constrained")
    figure.suptitle(f"Complete active contact snapshot · {len(records)} merged records\n"
                   "One mesh, five separated triangle pairs · area weighting on · dhat=0.7 · dmin=0.05")
    for group, title in enumerate(TITLES):
        axis = figure.add_subplot(2, 3, group + 1, projection="3d")
        positions = surface.positions - np.array([10 * group, 0, 0])
        axis.add_collection3d(Poly3DCollection(positions[surface.cells[2 * group:2 * group + 2]],
                                             facecolors="#b7c2cc", edgecolors="#71808e", alpha=.2))
        for kind, ids, numbers, _ in groups[group]:
            draw_record(axis, kind, positions[ids], numbers[2])
        frame(axis, positions[6 * group:6 * group + 6])
        distance = min(row[2][0] for row in groups[group])
        mollifiers = [row[2][2] for row in groups[group] if row[0] == "EE"]
        detail = f" · EE m={min(mollifiers):.3g}" if mollifiers else ""
        axis.set_title(f"{title}\n{len(groups[group])} records · min d={distance:.3g}{detail}", fontsize=10)
    legend = figure.add_subplot(2, 3, 6)
    legend.axis("off")
    handles = [Line2D([], [], color=color, marker="o", label=f"{kind}: {sum(r['type'] == kind for r in records)} records")
               for kind, color in COLORS.items()]
    legend.legend(handles=handles, loc="upper left", frameon=False, fontsize=12)
    legend.text(.02, .52,
                "Every CSV row is drawn.\n\n"
                "Colors identify assembly stencils.\n"
                "Dashed links show shortest distances.\nThey are not force arrows.\n\n"
                "Faint EE: mollifier = 0;\nall four support vertices are retained.\n\n"
                "Weights include reference areas\nand merged multiplicities; no kappa.",
                transform=legend.transAxes, va="top", fontsize=10, linespacing=1.2)
    output = args.output or args.input / "contacts.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved {output}")

    figure = plt.figure(figsize=(12, 8), layout="constrained")
    figure.suptitle("How to read ONE contact record\nFive selected records; the companion overview contains all 20")
    for group, expected in enumerate(("VV", "VE", "VF", "EE", "EE")):
        kind, ids, numbers, row = min((r for r in groups[group] if r[0] == expected),
                                      key=lambda r: (r[2][2], r[2][0]))
        p = surface.positions[ids] - np.array([10 * group, 0, 0])
        axis = figure.add_subplot(2, 3, group + 1, projection="3d")
        draw_record(axis, kind, p, numbers[2], detail=True)
        frame(axis, p)
        axis.set_axis_off()
        offset = .07 * np.ptp(p, axis=0).max()
        for i, (vertex, vertex_id) in enumerate(zip(p, ids)):
            below = (kind == "EE" and i < 2) or (kind in ("VE", "VF") and i > 0)
            axis.text(*(vertex + [0, 0, -offset if below else offset]), f"v{vertex_id}", fontsize=10,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1))
        label = TITLES[group] if kind == "EE" else {"VV": "vertex + vertex", "VE": "vertex + edge",
                                                   "VF": "vertex + triangle"}[kind]
        axis.set_title(f"{kind}: {label} · CSV row #{row}\n"
                       f"d={numbers[0]:.3g} · weight={numbers[1]:.3g} · m={numbers[2]:.3g}", fontsize=10)
    guide = figure.add_subplot(2, 3, 6)
    guide.axis("off")
    guide.text(.02, .95,
               "Read this figure first\n\n"
               "1. One panel = one CSV record.\n"
               "2. vN labels are mesh vertex IDs.\n"
               "3. Yellow dots = closest points,\n"
               "   possibly inside an edge or face.\n"
               "4. The black dashed segment has\n"
               "   length d (the reported distance).\n\n"
               "Only this record is shown.\n"
               "Other geometry is hidden.\n\n"
               "m multiplies this barrier term.\n"
               "Even with m=0, EE keeps four IDs.\n\n"
               "All 20 records remain in the CSV\n"
               "and the companion overview plot.",
               transform=guide.transAxes, va="top", fontsize=10, linespacing=1.2)
    detail_output = output.with_name(output.stem + "_details.png")
    figure.savefig(detail_output, dpi=160)
    plt.close(figure)
    print(f"Saved {detail_output}")


if __name__ == "__main__":
    main()
