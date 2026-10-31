import math

import numpy as np
import pytest

from magnet_sim.expr import ExprError, Expression, evaluate_number, normalize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("sqrt(3)", math.sqrt(3)),
        ("√3", math.sqrt(3)),
        ("2√3", 2 * math.sqrt(3)),
        ("pi/6", math.pi / 6),
        ("π/2", math.pi / 2),
        ("2^3", 8.0),
        ("-1.5", -1.5),
        ("1e-3", 1e-3),
        ("2pi", 2 * math.pi),
        ("3（1+1）", 6.0),
        ("atan2(1, 1)", math.pi / 4),
    ],
)
def test_evaluate_number(text, expected):
    assert evaluate_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "x", "1/0", "1 >", "__import__('os')", "(1).real", "x > 0"])
def test_evaluate_number_rejects(text):
    with pytest.raises(ExprError):
        evaluate_number(text)


def region(text):
    e = Expression(text, variables=("x", "y", "r"))
    assert e.is_boolean
    return lambda x, y: bool(e(x=x, y=y, r=math.hypot(x, y)))


def test_region_basic_and_chinese_syntax():
    f = region("x>0 and x^2+y^2<4")
    assert f(1, 0) and not f(-1, 0) and not f(3, 0)
    g = region("x＞0 且 x²+y²＜4")
    assert g(1, 0) and not g(-1, 0)
    h = region("-1 < x < 1 || y ≥ 2")
    assert h(0, -5) and h(5, 2) and not h(5, 0)
    k = region("1 < r < 2")
    assert k(1.5, 0) and not k(0.5, 0)
    n = region("非 (x > 0)")
    assert n(-1, 0) and not n(1, 0)
    m = region("y > 2x + 1")
    assert m(0, 2) and not m(0, 0)


def test_region_vectorized():
    e = Expression("x > 0 and y < x", variables=("x", "y"))
    X, Y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    result = e(x=X, y=Y)
    assert result.shape == X.shape
    assert result.dtype == bool
    assert np.array_equal(result, (X > 0) & (Y < X))


def test_normalize_keeps_function_names():
    assert "atan2(" in normalize("atan2(y, x) > 0")
    assert normalize("x>0 and y>0") == "x>0 and y>0"


@pytest.mark.parametrize(
    "text",
    ["import os", "x.__class__", "open('a')", "lambda: 1", "[x for x in y]", "z > 0", "sin > 0", "x > 0 if y else 1"],
)
def test_rejects_unsafe_or_unknown(text):
    with pytest.raises(ExprError):
        Expression(text, variables=("x", "y"))
