import pytest

from synthscript.metric.ast.parser import LatexParser
from synthscript.metric.calibration import (
    CalibrationObjective,
    OptimizerConfig,
    calibrate,
)
from synthscript.metric.calibration.samples import (
    CALIBRATION_DATASET,
    CANONICAL_EQUIVALENCE_PAIRS,
    CalibrationExample,
)
from synthscript.metric.node_metrics import ConfiguredNodeMetrics


def test_calibration_loss_uses_normalized_asymmetric_squared_error() -> None:
    objective = CalibrationObjective.from_pairs(
        [("a", "b", 2.0)],
        underestimation_penalty=4,
        regularization_strength=0,
    )

    # Predicted distance is 1 and total baseline mass is 2:
    # 4 * (2 * (1 - 2) / 2) ** 2 == 4.
    assert objective.loss(ConfiguredNodeMetrics()) == 4


def test_character_weight_is_a_fixed_baseline() -> None:
    objective = CalibrationObjective.from_pairs(
        [("a", "b", 1.0)], regularization_strength=0
    )

    with pytest.raises(ValueError, match="character_weight"):
        objective.loss(ConfiguredNodeMetrics(character_weight=2))
    with pytest.raises(ValueError, match="character_weight"):
        OptimizerConfig(parameter_names=("character_weight",))


def test_calibration_keeps_best_bounded_parameters() -> None:
    objective = CalibrationObjective.from_pairs(
        [(r"$x$", "x", 2.0)], regularization_strength=0
    )
    result = calibrate(
        objective,
        config=OptimizerConfig(
            iterations=3,
            patience=3,
            parameter_names=("math_weight",),
        ),
    )

    assert result.loss <= result.initial_loss
    assert result.parameters.character_weight == 1
    assert 0.25 <= result.parameters.math_weight <= 3


def test_calibration_loss_balances_families_not_example_counts() -> None:
    common = dict(category="test", template="template", split="train")
    examples = [
        CalibrationExample(
            "a",
            "b",
            0,
            family="large",
            **common,  # ty: ignore[invalid-argument-type]
        )
        for _ in range(10)
    ]
    examples.append(
        CalibrationExample(
            "a",
            "b",
            3,
            family="small",
            **common,  # ty: ignore[invalid-argument-type]
        )
    )
    objective = CalibrationObjective.from_examples(
        examples,
        underestimation_penalty=1,
        regularization_strength=0,
    )

    assert objective.family_losses(ConfiguredNodeMetrics()) == {
        "large": 1,
        "small": 4,
    }
    assert objective.data_loss(ConfiguredNodeMetrics()) == 2.5


def test_generated_splits_hold_out_whole_templates() -> None:
    template_sets = [
        {example.template for example in split}
        for split in (
            CALIBRATION_DATASET.train,
            CALIBRATION_DATASET.validation,
            CALIBRATION_DATASET.test,
        )
    ]

    assert all(template_sets)
    assert template_sets[0].isdisjoint(template_sets[1])
    assert template_sets[0].isdisjoint(template_sets[2])
    assert template_sets[1].isdisjoint(template_sets[2])
    calibration_pairs = {example.as_pair() for example in CALIBRATION_DATASET.all}
    assert calibration_pairs.isdisjoint(CANONICAL_EQUIVALENCE_PAIRS)


@pytest.mark.parametrize(("source", "target", "expected"), CANONICAL_EQUIVALENCE_PAIRS)
def test_calibration_canonical_equivalences_are_unit_tests(
    source: str, target: str, expected: float
) -> None:
    assert expected == 0
    parser = LatexParser()
    assert parser.parse(source) == parser.parse(target)
