import math

import pytest

from src.food11.promote import promotion_gate


def test_gate_boundary():
    threshold = 0.8 + 2 * math.sqrt(0.8 * 0.2 / 1000)
    assert promotion_gate(0.8, threshold, 1000)[0]
    assert not promotion_gate(0.8, threshold - 0.00001, 1000)[0]
    assert not promotion_gate(0.8, 0.8, 1000)[0]


@pytest.mark.parametrize("p,c,n", [(0.8, 0.9, 0), (-0.1, 0.9, 100), (0.8, float("nan"), 100)])
def test_gate_rejects_invalid_statistics(p, c, n):
    with pytest.raises(ValueError):
        promotion_gate(p, c, n)
