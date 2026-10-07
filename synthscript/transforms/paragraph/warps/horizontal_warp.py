import numpy as np

from synthscript.transforms.paragraph.warps._directional_warp import (
    _DirectionalArchWarp,
)


class HorizontalWarp(_DirectionalArchWarp):
    """
    Arches lines in the perpendicular direction to the reading axis.
    If 'amplitude' is positive, arches like a closing parenthesis, ),
    otherwise like an opening one.
    Negative amplitude: bow pointing left, like an opening parenthesis, (.
    """

    def _axes(
        self, reading_dir: np.ndarray, orthogonal_dir: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        domain_dir = reading_dir
        disp_dir = -orthogonal_dir
        return domain_dir, disp_dir
