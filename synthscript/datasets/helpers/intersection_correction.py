import numpy as np
import shapely
from shapely import Polygon
from shapely.affinity import translate
from shapely.prepared import prep

from synthscript.datasets.helpers.polygon_separation import separate_polygons
from synthscript.transforms.helpers.line_group_info import LineGroupInfo


def avoid_line_intersections(
    polygons: list[Polygon],
    *,
    delta: float = 0.5,
    binary_iterations: int = 32,
    contact_epsilon: float = 1e-8,
) -> list[Polygon]:
    n = len(polygons)
    if n <= 1:
        return polygons

    adjacent_distances = shapely.distance(polygons[:-1], polygons[1:])
    if np.all(adjacent_distances >= delta):
        return polygons

    lg_info = LineGroupInfo.from_polygons(polygons)
    original_center = lg_info.center
    reading_dir = lg_info.reading_direction

    left_extent = np.empty(n, dtype=float)
    right_extent = np.empty(n, dtype=float)

    for i, polygon in enumerate(polygons):
        coordinates = np.asarray(polygon.exterior.coords, dtype=float)[:, :2]
        projections = coordinates @ reading_dir
        left_extent[i] = projections.min()
        right_extent[i] = projections.max()

    pairwise_displacements = np.zeros(n, dtype=float)
    for i in range(1, n):
        true_gap = adjacent_distances[i - 1]
        if true_gap < delta:
            overlap = right_extent[i - 1] - left_extent[i] + delta
            if overlap > 0:
                pairwise_displacements[i] = overlap

    raw_forward_push = np.cumsum(pairwise_displacements)
    displacement_along_line = raw_forward_push - np.mean(raw_forward_push)

    for k in range(n):
        scalar_shift = displacement_along_line[k]
        if abs(scalar_shift) < 1e-12:
            continue
        movement = scalar_shift * reading_dir
        polygons[k] = translate(
            polygons[k],
            xoff=movement[0],
            yoff=movement[1],
        )

    poly_bounds = np.array([p.bounds for p in polygons], dtype=float)

    for k in range(1, n):
        max_reversal = float(raw_forward_push[k])
        if max_reversal <= 1e-12:
            continue

        polygon = polygons[k]
        vx = -reading_dir[0] * max_reversal
        vy = -reading_dir[1] * max_reversal

        bx0, by0, bx1, by1 = poly_bounds[k]
        swept_box = (
            min(bx0, bx0 + vx),
            min(by0, by0 + vy),
            max(bx1, bx1 + vx),
            max(by1, by1 + vy),
        )

        prev_bounds = poly_bounds[:k]
        overlaps = (
            (swept_box[0] <= prev_bounds[:, 2])
            & (swept_box[2] >= prev_bounds[:, 0])
            & (swept_box[1] <= prev_bounds[:, 3])
            & (swept_box[3] >= prev_bounds[:, 1])
        )
        candidate_indices = np.flatnonzero(overlaps)

        if len(candidate_indices) == 0:
            displacement = max_reversal

        else:
            active_candidates = [
                (prep(polygons[idx]), poly_bounds[idx])
                for idx in reversed(candidate_indices)
            ]
            base_coords = np.asarray(polygon.exterior.coords, dtype=float)[:, :2]

            def intersects_previous(distance: float) -> bool:
                off_x = -reading_dir[0] * distance
                off_y = -reading_dir[1] * distance
                cbx0 = bx0 + off_x
                cby0 = by0 + off_y
                cbx1 = bx1 + off_x
                cby1 = by1 + off_y

                candidate = None
                for prepared_prev, (pbx0, pby0, pbx1, pby1) in active_candidates:
                    if cbx0 <= pbx1 and cbx1 >= pbx0 and cby0 <= pby1 and cby1 >= pby0:
                        if candidate is None:
                            candidate = shapely.polygons(base_coords + (off_x, off_y))
                        if prepared_prev.intersects(candidate):
                            return True
                return False

            if not intersects_previous(max_reversal):
                displacement = max_reversal
            else:
                lo, hi = 0.0, max_reversal
                for _ in range(binary_iterations):
                    if hi - lo <= 1e-3:
                        break
                    mid = 0.5 * (lo + hi)
                    if intersects_previous(mid):
                        hi = mid
                    else:
                        lo = mid

                displacement = max(0.0, lo - contact_epsilon)

        if displacement <= 0:
            continue

        movement = reading_dir * (-displacement)
        polygons[k] = translate(
            polygon,
            xoff=movement[0],
            yoff=movement[1],
        )
        poly_bounds[k] = [
            bx0 + movement[0],
            by0 + movement[1],
            bx1 + movement[0],
            by1 + movement[1],
        ]

    bounds = shapely.bounds(polygons)
    new_center = (bounds[:, :2].min(axis=0) + bounds[:, 2:].max(axis=0)) / 2
    center_shift_x = original_center[0] - new_center[0]
    center_shift_y = original_center[1] - new_center[1]

    if abs(center_shift_x) > 1e-12 or abs(center_shift_y) > 1e-12:
        for i in range(n):
            polygons[i] = translate(
                polygons[i],
                xoff=center_shift_x,
                yoff=center_shift_y,
            )

    return polygons


def avoid_paragraph_intersections(
    polygon_groups: list[list[Polygon]],
    delta: float = 5,
    max_iterations: int = 100,
    damping: float = 0.5,
) -> list[list[Polygon]]:
    # we use convex hulls: we dont want intersections to be so fine-grained that a far
    # line  from a paragraph could be interleaved in the space between the lines of
    # another.
    _, shifts = separate_polygons(
        [shapely.GeometryCollection(group).convex_hull for group in polygon_groups],
        delta=delta,
        max_iterations=max_iterations,
        damping=damping,
    )

    for polygon_group, shift in zip(polygon_groups, shifts, strict=True):
        if not np.any(shift):
            continue
        for i in range(len(polygon_group)):
            polygon_group[i] = translate(polygon_group[i], shift[0], shift[1])

    return polygon_groups
