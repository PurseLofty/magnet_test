"""磁场区域的几何形状。

每种形状都实现 ``contains(x, y)``：x、y 可以是标量或 numpy 数组，
返回同形状的布尔值。形状通过 :func:`make_region` 从用户输入的文字创建，
原始输入保存在 ``region.inputs`` 中，便于再次编辑时回填。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .expr import ExprError, Expression, evaluate_number


@dataclass
class Param:
    """形状参数的界面描述。choices 不为空时显示为下拉框。"""

    key: str
    label: str
    default: str
    choices: Optional[List[str]] = None
    wide: bool = False


@dataclass
class ShapeSpec:
    kind: str
    label: str
    params: List[Param]


HALF_CHOICES = ["x >", "x <", "y >", "y <"]
STRIP_CHOICES = ["a < x < b（竖直条带）", "a < y < b（水平条带）"]

SHAPES: Dict[str, ShapeSpec] = {
    s.kind: s
    for s in [
        ShapeSpec("plane", "整个平面", []),
        ShapeSpec(
            "half",
            "半平面",
            [Param("cond", "条件", "x >", HALF_CHOICES), Param("value", "边界值", "0")],
        ),
        ShapeSpec(
            "strip",
            "条形区域",
            [
                Param("axis", "方向", STRIP_CHOICES[0], STRIP_CHOICES),
                Param("a", "a", "-1"),
                Param("b", "b", "1"),
            ],
        ),
        ShapeSpec(
            "rect",
            "矩形",
            [
                Param("x1", "x 从", "-1"),
                Param("x2", "x 到", "1"),
                Param("y1", "y 从", "-1"),
                Param("y2", "y 到", "1"),
            ],
        ),
        ShapeSpec(
            "circle",
            "圆形",
            [Param("cx", "圆心 x", "0"), Param("cy", "圆心 y", "0"), Param("r", "半径", "1")],
        ),
        ShapeSpec(
            "annulus",
            "圆环",
            [
                Param("cx", "圆心 x", "0"),
                Param("cy", "圆心 y", "0"),
                Param("r1", "内半径", "1"),
                Param("r2", "外半径", "2"),
            ],
        ),
        ShapeSpec(
            "expr",
            "自定义不等式",
            [Param("text", "不等式", "x > 0 and x^2 + y^2 < 4", wide=True)],
        ),
    ]
}


def _fmt(v: float) -> str:
    """简洁地显示数字：2.0 → 2，0.333333 → 0.333。"""
    return f"{v:.3g}"


class Region:
    kind = ""
    inputs: Dict[str, str]

    def contains(self, x, y):  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - 抽象方法
        raise NotImplementedError


@dataclass
class WholePlane(Region):
    inputs: Dict[str, str] = field(default_factory=dict)
    kind = "plane"

    def contains(self, x, y):
        return np.ones(np.shape(x), dtype=bool) if np.ndim(x) else True

    def describe(self) -> str:
        return "整个平面"


@dataclass
class HalfPlane(Region):
    axis: str = "x"  # "x" 或 "y"
    greater: bool = True
    value: float = 0.0
    inputs: Dict[str, str] = field(default_factory=dict)
    kind = "half"

    def contains(self, x, y):
        c = x if self.axis == "x" else y
        return c > self.value if self.greater else c < self.value

    def describe(self) -> str:
        return f"{self.axis} {'>' if self.greater else '<'} {_fmt(self.value)}"


@dataclass
class Strip(Region):
    axis: str = "x"
    lo: float = -1.0
    hi: float = 1.0
    inputs: Dict[str, str] = field(default_factory=dict)
    kind = "strip"

    def contains(self, x, y):
        c = x if self.axis == "x" else y
        return (c > self.lo) & (c < self.hi)

    def describe(self) -> str:
        return f"{_fmt(self.lo)} < {self.axis} < {_fmt(self.hi)}"


@dataclass
class Rectangle(Region):
    x1: float = -1.0
    x2: float = 1.0
    y1: float = -1.0
    y2: float = 1.0
    inputs: Dict[str, str] = field(default_factory=dict)
    kind = "rect"

    def contains(self, x, y):
        return (x > self.x1) & (x < self.x2) & (y > self.y1) & (y < self.y2)

    def describe(self) -> str:
        return (
            f"矩形 {_fmt(self.x1)}<x<{_fmt(self.x2)}, "
            f"{_fmt(self.y1)}<y<{_fmt(self.y2)}"
        )


@dataclass
class Circle(Region):
    cx: float = 0.0
    cy: float = 0.0
    r: float = 1.0
    inputs: Dict[str, str] = field(default_factory=dict)
    kind = "circle"

    def contains(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 < self.r**2

    def describe(self) -> str:
        return f"圆 圆心({_fmt(self.cx)}, {_fmt(self.cy)}) 半径{_fmt(self.r)}"


@dataclass
class Annulus(Region):
    cx: float = 0.0
    cy: float = 0.0
    r1: float = 1.0
    r2: float = 2.0
    inputs: Dict[str, str] = field(default_factory=dict)
    kind = "annulus"

    def contains(self, x, y):
        d2 = (x - self.cx) ** 2 + (y - self.cy) ** 2
        return (d2 > self.r1**2) & (d2 < self.r2**2)

    def describe(self) -> str:
        return (
            f"圆环 圆心({_fmt(self.cx)}, {_fmt(self.cy)}) "
            f"半径{_fmt(self.r1)}~{_fmt(self.r2)}"
        )


class Inequality(Region):
    """用户自定义不等式区域，变量 x、y、r（r = √(x²+y²)）。"""

    kind = "expr"

    def __init__(self, text: str, inputs: Optional[Dict[str, str]] = None):
        expr = Expression(text, variables=("x", "y", "r"))
        if not expr.is_boolean:
            raise ExprError("区域表达式需要是条件（含 > < 等比较符号），例如 x^2+y^2 < 4")
        self.expr = expr
        self.text = expr.text
        self.inputs = inputs if inputs is not None else {"text": text}
        # 试算一次，尽早发现运行期错误（如函数参数个数不对）
        self.contains(np.array([[0.0, 1.0], [-1.0, 0.5]]), np.array([[0.0, 1.0], [2.0, -3.0]]))
        self.contains(0.5, -0.5)

    def contains(self, x, y):
        r = np.hypot(x, y)
        result = self.expr(x=x, y=y, r=r)
        if np.ndim(x) and np.ndim(result) == 0:
            # 常量条件（如 True），扩展到数组形状
            result = np.full(np.shape(x), bool(result))
        return np.asarray(result, dtype=bool) if np.ndim(x) else bool(result)

    def describe(self) -> str:
        return self.text


def make_region(kind: str, inputs: Dict[str, str]) -> Region:
    """根据形状类型和用户输入的文字创建区域；输入有误时抛出 ExprError。"""
    spec = SHAPES.get(kind)
    if spec is None:
        raise ExprError(f"未知的区域类型：{kind}")
    labels = {p.key: p.label for p in spec.params}
    inputs = {k: str(v).strip() for k, v in inputs.items()}

    def num(key: str) -> float:
        return evaluate_number(inputs.get(key, ""), labels[key])

    if kind == "plane":
        return WholePlane(inputs=inputs)
    if kind == "half":
        cond = inputs.get("cond", "x >")
        if cond not in HALF_CHOICES:
            raise ExprError("半平面条件无效")
        return HalfPlane(axis=cond[0], greater=cond.endswith(">"), value=num("value"), inputs=inputs)
    if kind == "strip":
        axis = "y" if inputs.get("axis", STRIP_CHOICES[0]) == STRIP_CHOICES[1] else "x"
        a, b = num("a"), num("b")
        if a == b:
            raise ExprError("条形区域的 a 和 b 不能相等")
        return Strip(axis=axis, lo=min(a, b), hi=max(a, b), inputs=inputs)
    if kind == "rect":
        x1, x2, y1, y2 = num("x1"), num("x2"), num("y1"), num("y2")
        if x1 == x2 or y1 == y2:
            raise ExprError("矩形的宽和高都不能为 0")
        return Rectangle(min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2), inputs=inputs)
    if kind == "circle":
        r = num("r")
        if r <= 0:
            raise ExprError("半径必须大于 0")
        return Circle(num("cx"), num("cy"), r, inputs=inputs)
    if kind == "annulus":
        r1, r2 = num("r1"), num("r2")
        if r1 < 0 or r2 <= 0:
            raise ExprError("半径必须大于 0")
        if r1 >= r2:
            raise ExprError("内半径必须小于外半径")
        return Annulus(num("cx"), num("cy"), r1, r2, inputs=inputs)
    if kind == "expr":
        return Inequality(inputs.get("text", ""), inputs=inputs)
    raise ExprError(f"未知的区域类型：{kind}")  # pragma: no cover


def default_inputs(kind: str) -> Dict[str, str]:
    return {p.key: p.default for p in SHAPES[kind].params}

