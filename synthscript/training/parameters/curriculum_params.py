from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class CurriculumParams:
    """Parámetros para el aprendizaje por currículum"""

    orders_per_epoch: list[list[int | str]]
