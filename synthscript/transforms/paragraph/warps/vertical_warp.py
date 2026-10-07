import numpy as np

from synthscript.transforms.paragraph.warps._directional_warp import (
    _DirectionalArchWarp,
)


class VerticalWarp(_DirectionalArchWarp):
    """
    Arches lines in the "up the paragraph" axis. If 'amplitude' is positive,
    arches like a mountain, otherwise like a valley.
    """

    def _axes(
        self, reading_dir: np.ndarray, orthogonal_dir: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        domain_dir = orthogonal_dir
        disp_dir = reading_dir
        return domain_dir, disp_dir
