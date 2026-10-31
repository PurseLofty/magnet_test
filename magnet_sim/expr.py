"""安全的数学表达式解析器。

用于两种场合：
1. 区域不等式，如 ``x>0 and x^2+y^2<4``，对 numpy 数组逐点求值，返回布尔数组；
2. 数值输入框，如 ``sqrt(3)``、``pi/2``，求出一个浮点数。

只允许白名单内的语法结构（四则运算、乘方、比较、与或非、少量数学函数），
不会执行任意 Python 代码。
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable, Dict, Iterable

import numpy as np


class ExprError(ValueError):
    """表达式有误（错误信息为中文，可直接展示给用户）。"""


_FUNCS: Dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "atan2": np.arctan2,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "lg": np.log10,
    "hypot": np.hypot,
    "min": np.minimum,
    "max": np.maximum,
}

_CONSTS: Dict[str, float] = {"pi": math.pi, "e": math.e}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: np.true_divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}

_CMPOPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# 文本替换：全角符号、数学符号、中文逻辑词 → Python 语法
_REPLACEMENTS = [
    ("（", "("),
    ("）", ")"),
    ("，", ","),
    ("＜", "<"),
    ("＞", ">"),
    ("≤", "<="),
    ("≦", "<="),
    ("⩽", "<="),
    ("≥", ">="),
    ("≧", ">="),
    ("⩾", ">="),
    ("≠", "!="),
    ("＝", "="),
    ("＋", "+"),
    ("－", "-"),
    ("−", "-"),
    ("×", "*"),
    ("·", "*"),
    ("÷", "/"),
    ("π", "pi"),
    ("²", "**2"),
    ("³", "**3"),
    ("^", "**"),
    ("&&", " and "),
    ("||", " or "),
    ("&", " and "),
    ("|", " or "),
    ("并且", " and "),
    ("且", " and "),
    ("或者", " or "),
    ("或", " or "),
    ("非", " not "),
]

# 2x → 2*x，3(x+1) → 3*(x+1)；不影响 1e5、atan2(...) 以及 "0 and"
_IMPLICIT_MUL = re.compile(
    r"(?<![A-Za-z_\d.])(\d+(?:\.\d+)?)(?=(?!and\b|or\b|not\b)(?![eE][+-]?\d)[A-Za-z_(])"
)
# 单独的 "=" 视为 "=="
_LONE_EQ = re.compile(r"(?<![<>=!])=(?!=)")
# √3 → sqrt(3)，√x → sqrt(x)
_SQRT_ATOM = re.compile(r"√\s*(\d+(?:\.\d+)?|[A-Za-z_]\w*)")


def normalize(text: str) -> str:
    """把用户输入的常见写法统一成 Python 表达式语法。"""
    s = text.strip()
    s = _SQRT_ATOM.sub(r"sqrt(\1)", s)
    s = s.replace("√", "sqrt")
    for old, new in _REPLACEMENTS:
        s = s.replace(old, new)
    s = _LONE_EQ.sub("==", s)
    s = _IMPLICIT_MUL.sub(r"\1*", s)
    return " ".join(s.split())


def _compile(node: ast.AST, names: frozenset) -> Callable[[dict], object]:
    """把经过白名单校验的 AST 编译成嵌套闭包，求值时只需传入变量字典。"""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return lambda env: value
        if isinstance(value, (int, float)):
            value = float(value)
            return lambda env: value
        raise ExprError(f"不支持的常量：{value!r}")

    if isinstance(node, ast.Name):
        name = node.id
        if name in names:
            return lambda env: env[name]
        if name in _CONSTS:
            value = _CONSTS[name]
            return lambda env: value
        if name in ("true", "True"):
            return lambda env: True
        if name in ("false", "False"):
            return lambda env: False
        if name in _FUNCS:
            raise ExprError(f"函数 {name} 后面需要加括号，如 {name}(x)")
        allowed = "、".join(sorted(names)) if names else "（无）"
        raise ExprError(f"不认识的名称“{name}”。可用变量：{allowed}；常数：pi、e")

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExprError("不支持的运算符（只支持 + - * / ^ %）")
        left = _compile(node.left, names)
        right = _compile(node.right, names)
        return lambda env: op(left(env), right(env))

    if isinstance(node, ast.UnaryOp):
        operand = _compile(node.operand, names)
        if isinstance(node.op, ast.USub):
            return lambda env: -operand(env)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return lambda env: np.logical_not(operand(env))
        raise ExprError("不支持的一元运算符")

    if isinstance(node, ast.BoolOp):
        parts = [_compile(v, names) for v in node.values]
        if isinstance(node.op, ast.And):
            def _and(env):
                result = parts[0](env)
                for p in parts[1:]:
                    result = np.logical_and(result, p(env))
                return result
            return _and

        def _or(env):
            result = parts[0](env)
            for p in parts[1:]:
                result = np.logical_or(result, p(env))
            return result
        return _or

    if isinstance(node, ast.Compare):
        operands = [_compile(node.left, names)] + [_compile(c, names) for c in node.comparators]
        ops = []
        for op_node in node.ops:
            op = _CMPOPS.get(type(op_node))
            if op is None:
                raise ExprError("不支持的比较运算（只支持 < <= > >= == !=）")
            ops.append(op)

        def _cmp(env):
            values = [f(env) for f in operands]
            result = ops[0](values[0], values[1])
            for i in range(1, len(ops)):
                result = np.logical_and(result, ops[i](values[i], values[i + 1]))
            return result
        return _cmp

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            fname = getattr(node.func, "id", "?")
            raise ExprError(
                f"不支持的函数“{fname}”。可用函数：" + "、".join(sorted(_FUNCS))
            )
        if node.keywords:
            raise ExprError("函数参数不能使用 名称=值 的写法")
        func = _FUNCS[node.func.id]
        args = [_compile(a, names) for a in node.args]
        expected = 2 if node.func.id in ("atan2", "hypot", "min", "max") else 1
        if len(args) != expected:
            raise ExprError(f"函数 {node.func.id} 需要 {expected} 个参数")
        return lambda env: func(*[a(env) for a in args])

    raise ExprError("表达式中含有不支持的写法")


def _is_boolean(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.BoolOp):
        return all(_is_boolean(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_boolean(node.operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return True
    if isinstance(node, ast.Name) and node.id in ("true", "True", "false", "False"):
        return True
    return False


class Expression:
    """编译后的表达式，可对标量或 numpy 数组反复求值。"""

    def __init__(self, text: str, variables: Iterable[str] = ()):
        if not text or not text.strip():
            raise ExprError("表达式不能为空")
        self.text = text.strip()
        self.normalized = normalize(text)
        self.variables = frozenset(variables)
        try:
            tree = ast.parse(self.normalized, mode="eval")
        except SyntaxError:
            raise ExprError(
                f"无法解析“{self.text}”，请检查括号是否配对、运算符是否完整"
            ) from None
        self._fn = _compile(tree.body, self.variables)
        self.is_boolean = _is_boolean(tree.body)

    def __call__(self, **env):
        try:
            with np.errstate(all="ignore"):
                return self._fn(env)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ExprError(f"计算“{self.text}”时出错：{exc}") from None


def evaluate_number(text: str, what: str = "数值") -> float:
    """把数值输入框里的文字（允许 sqrt(3)、pi/6、2√3 等）转为浮点数。"""
    if text is None or not str(text).strip():
        raise ExprError(f"{what}不能为空")
    try:
        expr = Expression(str(text))
    except ExprError as exc:
        raise ExprError(f"{what}：{exc}") from None
    if expr.is_boolean:
        raise ExprError(f"{what}应该是一个数，而不是条件")
    value = expr()
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ExprError(f"{what}“{text}”不是一个数") from None
    if not math.isfinite(value):
        raise ExprError(f"{what}“{text}”的结果不是有限的数")
    return value
