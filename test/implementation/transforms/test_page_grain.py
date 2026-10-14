import numpy as np
import pytest

from synthscript.shared.parameters import UniformDistribution
from synthscript.transforms import BackgroundTransform
from synthscript.transforms.image import PageGrain
from synthscript.transforms.image_transform_pack import ImageTransformPack


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_grain_preserves_image_and_has_configured_strength(dtype):
    scale = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    image = np.full((256, 256), scale / 2, dtype=dtype)
    original = image.copy()
    transform = PageGrain(0.02)
    transform.rng = 123

    result = transform(image)

    assert isinstance(transform, BackgroundTransform)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(image, original)
    noise = result.astype(np.float64) - image
    assert abs(noise.mean()) < 0.003 * scale
    assert noise.std() == pytest.approx(0.02 * scale, rel=0.03)


@pytest.mark.parametrize("value", [0, 255])
def test_grain_clips_instead_of_wrapping(value):
    image = np.full((128, 128), value, dtype=np.uint8)
    transform = PageGrain(0.1)
    transform.rng = 123

    result = transform(image)

    assert np.any(result == value)
    assert np.any(result != value)
    if value == 0:
        assert result.max() < 128
    else:
        assert result.min() > 128


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_grain_is_achromatic_and_preserves_alpha(channels):
    image = np.full((32, 32, channels), 128, dtype=np.uint8)
    if channels == 4:
        image[..., 3] = np.arange(32, dtype=np.uint8)
    transform = PageGrain()
    transform.rng = 123

    result = transform(image)

    assert result.shape == image.shape
    for channel in range(min(channels, 3)):
        np.testing.assert_array_equal(result[..., 0], result[..., channel])
    if channels == 4:
        np.testing.assert_array_equal(result[..., 3], image[..., 3])


def test_zero_intensity_is_an_identity_without_drawing_noise():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    transform = PageGrain(0)
    transform.rng = 123

    result = transform(image)

    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)
    assert (
        transform.rng.bit_generator.state
        == np.random.default_rng(123).bit_generator.state
    )


def test_float_grain_stays_in_normalized_range():
    image = np.ones((64, 64), dtype=np.float32)
    transform = PageGrain(0.5)
    transform.rng = 123

    result = transform(image)

    assert result.min() == 0
    assert result.max() == 1
    assert np.any((result > 0) & (result < 1))


def test_pack_propagates_rng_to_random_intensity_and_only_changes_background():
    image = np.full((32, 32), 128, dtype=np.uint8)
    packs = [ImageTransformPack(rng=123) for _ in range(2)]
    for pack in packs:
        pack.add_transform(PageGrain(UniformDistribution(0.01, 0.04)))

    first = packs[0].transform_background(image)
    np.testing.assert_array_equal(first, packs[1].transform_background(image))
    assert np.any(first != image)
    assert packs[0].transform_strokes([image])[0] is image
    assert packs[0].transform_global_image(image) is image


def test_disabled_grain_is_skipped_by_pack():
    image = np.full((32, 32), 128, dtype=np.uint8)
    pack = ImageTransformPack(rng=123)
    pack.add_transform(PageGrain(probability=0))

    assert pack.is_identity
    assert pack.transform_background(image) is image
