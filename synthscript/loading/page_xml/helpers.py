import math
import re
import warnings

from shapely import make_valid
from shapely.geometry import Polygon

_INVALID_XML_ID_CHARACTER = re.compile(r"[^A-Za-z0-9_.-]+")
_VALID_XML_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def make_page_xml_id(
    original: str,
    *,
    prefix: str,
    used_ids: set[str],
) -> str:
    """Return a deterministic, document-unique XML ID."""
    original = str(original)
    prefix = _INVALID_XML_ID_CHARACTER.sub("_", prefix)
    if not prefix or not re.match(r"^[A-Za-z_]", prefix):
        prefix = f"id_{prefix}"

    if original.startswith(prefix) and _VALID_XML_ID.fullmatch(original):
        base = original
    else:
        cleaned = _INVALID_XML_ID_CHARACTER.sub("_", original).strip("_")
        base = f"{prefix}{cleaned or 'item'}"

    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def normalize_page_orientation(angle: float) -> float:
    """Normalize an angle to PAGE's documented (-180, 180] range."""
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError(f"Orientation must be finite, got {angle!r}.")
    normalized = (angle + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0):
        return 180.0
    if math.isclose(normalized, 0.0):
        return 0.0
    return normalized


def ocr_rotation_to_page_orientation(rotation: float) -> float:
    return normalize_page_orientation(-float(rotation))


def page_orientation_to_ocr_rotation(orientation: float) -> float:
    return normalize_page_orientation(-float(orientation))


def polygon_from_page_points(points: list[tuple[float, float]]) -> Polygon:
    """Validate PAGE points and safely repair a polygon when possible."""
    if len(points) < 3 or len(set(points)) < 3:
        raise ValueError("PAGE Coords needs at least three distinct points.")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("PAGE coordinates must be finite.")
    if any(value < 0 for point in points for value in point):
        raise ValueError("PAGE coordinates cannot be negative.")

    polygon = Polygon(points)
    if not polygon.is_valid:
        warnings.warn(
            "Repairing invalid PAGE Coords with Shapely before loading.",
            stacklevel=2,
        )
        repaired = make_valid(polygon)
        polygon = repaired if isinstance(repaired, Polygon) else repaired.convex_hull
    if polygon.is_empty or not isinstance(polygon, Polygon) or polygon.area <= 0:
        raise ValueError("PAGE Coords does not describe a usable polygon.")
    return polygon


def parse_page_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in value.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid PAGE point {token!r}.")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(f"Invalid PAGE point {token!r}.") from exc
    return points


def polygon_to_page_points(polygon: Polygon) -> str:
    """Serialize a polygon with one consistent round-to-nearest policy."""
    if polygon.is_empty:
        raise ValueError("Cannot serialize an empty polygon.")
    if not polygon.is_valid:
        warnings.warn(
            "Repairing an invalid polygon with Shapely before PAGE serialization.",
            stacklevel=2,
        )
        repaired = make_valid(polygon)
        polygon = repaired if isinstance(repaired, Polygon) else repaired.convex_hull
    if not isinstance(polygon, Polygon) or polygon.area <= 0:
        raise ValueError("Cannot serialize an unusable polygon.")

    rounded: list[tuple[int, int]] = []
    for x, y in list(polygon.exterior.coords)[:-1]:
        point = (round(x), round(y))
        if point[0] < 0 or point[1] < 0:
            raise ValueError("PAGE coordinates cannot be negative.")
        if not rounded or rounded[-1] != point:
            rounded.append(point)
    if len(rounded) > 1 and rounded[0] == rounded[-1]:
        rounded.pop()
    if len(rounded) < 3 or Polygon(rounded).area <= 0:
        raise ValueError("Rounding collapsed a polygon below three usable points.")
    return " ".join(f"{x},{y}" for x, y in rounded)
