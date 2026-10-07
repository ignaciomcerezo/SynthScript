from abc import abstractmethod
from collections.abc import Sequence

import cv2
import numpy as np
import shapely
from shapely import Polygon

from synthscript.shared.parameters import Parameter
from synthscript.transforms.helpers.line_group_info import LineGroupInfo
from synthscript.transforms.transforms import (
    ParagraphTransform,
    line_group_equivalent_type,
)


class _DirectionalArchWarp(ParagraphTransform):
    """
    Shared function for warping a group of lines along the two
    paragraph-relevant axis.
    Each subclasses chooses two axes:
        1. The domain, where distance is measured.
        2. The orthogonal, where displacement occurs.
    In VerticalWarp and HorizontalWarp, this directions are
    derived from the paragraph's reading direction.

    Positive amplitude bows thhe domain range towards the displacement
    direction.
    """

    def __init__(
        self,
        amplitude: Parameter | float,
        *,
        segmentation_thinness: int = 10,
        probability: float = 1,
    ):
        self.amplitude = Parameter(amplitude)
        self.segmentation_thinness = segmentation_thinness
        self.probability = probability
        self.may_cause_intersections = True

    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)

        if not polygons:
            return list(images), list(polygons)

        reading_dir = LineGroupInfo.from_polygons(polygons).reading_direction
        orthogonal_dir = LineGroupInfo.compute_orthogonal_direction(
            [LineGroupInfo.poly_center(polygon) for polygon in polygons], reading_dir
        )

        domain_dir, disp_dir = self._axes(reading_dir, orthogonal_dir)

        domain_min, domain_max = self._domain_bounds(polygons, domain_dir)

        new_polygons = []
        new_images = []

        amplitude = self.amplitude()
        for image, polygon in zip(images, polygons, strict=True):
            orig_bounds = polygon.bounds

            new_polygon = self._apply_arch_poly(
                polygon, amplitude, domain_dir, disp_dir, domain_min, domain_max
            )

            new_images.append(
                self._apply_arch_img(
                    image,
                    amplitude,
                    domain_dir,
                    disp_dir,
                    domain_min,
                    domain_max,
                    orig_bounds,
                    new_polygon.bounds,
                )
            )
            new_polygons.append(new_polygon)

        return new_images, new_polygons

    @abstractmethod
    def _axes(
        self, reading_dir: np.ndarray, orthogonal_dir: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (domain_dir, displacement_dir). Overridden per subclass."""
        raise NotImplementedError

    @staticmethod
    def _domain_bounds(
        polygons: Sequence[Polygon], domain_dir: np.ndarray
    ) -> tuple[float, float]:
        proj_min = np.inf
        proj_max = -np.inf

        for polygon in polygons:
            coords = np.asarray(polygon.exterior.coords, dtype=float)[:, :2]
            projections = coords @ domain_dir
            proj_min = min(proj_min, float(projections.min()))
            proj_max = max(proj_max, float(projections.max()))

        return proj_min, proj_max

    @staticmethod
    def _displacement(x_norm: np.ndarray, amplitude: float) -> np.ndarray:
        abs_amp = abs(amplitude)

        if amplitude >= 0:
            # bow toward +displacement_dir: middle of the domain range
            # moves furthest, the ends dont move.
            return abs_amp * (1.0 - x_norm**2)
        else:
            # bow toward -displacement_dir: ends move furthest, the
            # middle of the domain does not move.
            return abs_amp * (x_norm**2)

    def _apply_arch_poly(
        self,
        poly: Polygon,
        amplitude: float,
        domain_dir: np.ndarray,
        disp_dir: np.ndarray,
        domain_min: float,
        domain_max: float,
    ) -> Polygon:
        densified_poly = shapely.segmentize(poly, self.segmentation_thinness)
        domain_width = domain_max - domain_min

        def vectorized_mapping(coords: np.ndarray) -> np.ndarray:
            out = np.empty_like(coords, dtype=np.float64)
            xy = coords[:, :2]

            u = xy @ domain_dir
            v = xy @ disp_dir

            if domain_width == 0:
                x_norm = np.zeros_like(u)
            else:
                x_norm = (2.0 * u - (domain_max + domain_min)) / domain_width

            displacement = self._displacement(x_norm, amplitude)
            v_new = v + displacement

            out[:, 0] = u * domain_dir[0] + v_new * disp_dir[0]
            out[:, 1] = u * domain_dir[1] + v_new * disp_dir[1]

            if coords.shape[1] == 3:
                out[:, 2] = coords[:, 2]

            return out

        return shapely.transform(densified_poly, vectorized_mapping)

    def _apply_arch_img(
        self,
        image: np.ndarray,
        amplitude: float,
        domain_dir: np.ndarray,
        disp_dir: np.ndarray,
        domain_min: float,
        domain_max: float,
        orig_bounds: tuple,
        new_bounds: tuple,
    ) -> np.ndarray:
        """
        Applies the same warp to the line's raster image via cv2.remap.
        """

        orig_box_x0, orig_box_y0, _, _ = orig_bounds
        new_box_x0, new_box_y0, new_box_x2, new_box_y2 = new_bounds

        new_width = max(1, int(np.ceil(new_box_x2 - new_box_x0)))
        new_height = max(1, int(np.ceil(new_box_y2 - new_box_y0)))

        x_d, y_d = np.meshgrid(np.arange(new_width), np.arange(new_height))

        x_global = new_box_x0 + x_d
        y_global = new_box_y0 + y_d

        u = x_global * domain_dir[0] + y_global * domain_dir[1]
        v = x_global * disp_dir[0] + y_global * disp_dir[1]

        domain_width = domain_max - domain_min
        if domain_width == 0:
            x_norm = np.zeros_like(u, dtype=np.float32)
        else:
            x_norm = (2.0 * u - (domain_max + domain_min)) / domain_width

        displacement = self._displacement(x_norm, amplitude)

        # Inverse mapping: u is unchanged by the warp, v is shifted by
        # +displacement going forward, so subtract it going backward.
        u_s = u
        v_s = v - displacement

        x_s_global = u_s * domain_dir[0] + v_s * disp_dir[0]
        y_s_global = u_s * domain_dir[1] + v_s * disp_dir[1]

        x_s = (x_s_global - orig_box_x0).astype(np.float32)
        y_s = (y_s_global - orig_box_y0).astype(np.float32)

        return cv2.remap(
            image,
            x_s,
            y_s,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
