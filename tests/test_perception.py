import numpy as np
import pytest

from warehouse.contracts import Level
from warehouse.perception import estimate_stock


@pytest.mark.parametrize("fraction,expected", [(0.0, Level.EMPTY), (0.15, Level.LOW), (0.6, Level.OK)])
def test_controlled_color_fixtures(fraction, expected):
    image = np.full((100, 100, 3), (30, 158, 166), dtype=np.uint8)
    image[:int(100 * fraction)] = (244, 82, 6)
    observation = estimate_stock(image, (0, 0, 100, 100), "fixture", 1.0)
    assert observation.level == expected
    assert observation.count_estimate is None
    assert observation.source == "VISION_AREA"


@pytest.mark.parametrize("image", [None, np.zeros((100, 100, 3), dtype=np.uint8)])
def test_obstructed_or_missing_image_is_unknown(image):
    assert estimate_stock(image, (0, 0, 100, 100), "fixture", 1.0).level == Level.UNKNOWN
