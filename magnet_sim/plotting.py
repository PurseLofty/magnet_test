"""在 matplotlib 坐标轴上绘制磁场区域与粒子初始状态（与界面库无关）。"""

from __future__ import annotations

import math
from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import font_manager

from .field import OUT, MagneticField
from .physics import Particle

OUT_RGB = (1.0, 0.55, 0.10)  # 穿出纸面：橙色
IN_RGB = (0.15, 0.45, 1.00)  # 穿入纸面：蓝色
POSITIVE_COLOR = "#d62728"
NEGATIVE_COLOR = "#1f5fbf"
BOUNDARY_COLOR = "#333333"
SYMBOL_COLOR = "#444444"

_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

_CJK_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]


def setup_fonts() -> None:
    """让 matplotlib 能显示中文和负号。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in _CJK_FONTS if name in available]
    matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


def region_tag(i: int) -> str:
    return _CIRCLED[i] if i < len(_CIRCLED) else f"({i + 1})"


def fmt_num(v: float) -> str:
    return f"{v:.3g}"


def particle_color(q: float) -> str:
    return POSITIVE_COLOR if q >= 0 else NEGATIVE_COLOR


def _label_point(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """在区域内部找一个离边界尽量远的点，用来放区域标签。"""
    if not mask.any():
        return None
    m = mask.copy()
    for _ in range(max(mask.shape)):
        er = m.copy()
        er[1:, :] &= m[:-1, :]
        er[:-1, :] &= m[1:, :]
        er[:, 1:] &= m[:, :-1]
        er[:, :-1] &= m[:, 1:]
        er[0, :] = er[-1, :] = False
        er[:, 0] = er[:, -1] = False
        if not er.any():
            break
        m = er
    iy, ix = np.nonzero(m)
    k = int(np.argmin((iy - iy.mean()) ** 2 + (ix - ix.mean()) ** 2))
    return xs[ix[k]], ys[iy[k]]


def draw_field(ax, field: MagneticField, view: Tuple[float, float, float, float], resolution: int = 600) -> None:
    xmin, xmax, ymin, ymax = view
    w, h = xmax - xmin, ymax - ymin
    span = max(w, h)
    nx = max(50, int(resolution * w / span))
    ny = max(50, int(resolution * h / span))
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    idx = field.index_grid(X, Y)

    regions = field.regions
    bmax = max([fr.B for fr in regions] + [0.0]) or 1.0

    # 1. 区域着色：方向决定颜色，强度决定深浅
    rgba = np.zeros((ny, nx, 4))
    for i, fr in enumerate(regions):
        if fr.B <= 0:
            continue
        mask = idx == i
        color = OUT_RGB if fr.direction == OUT else IN_RGB
        rgba[mask] = (*color, 0.15 + 0.30 * fr.B / bmax)
    ax.imshow(rgba, origin="lower", extent=(xmin, xmax, ymin, ymax), interpolation="nearest", zorder=0)

    # 2. 区域边界
    for i in range(len(regions)):
        mask = idx == i
        if mask.any() and not mask.all():
            ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=BOUNDARY_COLOR, linewidths=1.3, zorder=2)

    # 3. 磁场符号：· 表示穿出纸面，× 表示穿入纸面
    step = span / 16
    sx = np.arange(xmin + step / 2, xmax, step)
    sy = np.arange(ymin + step / 2, ymax, step)
    SX, SY = np.meshgrid(sx, sy)
    sidx = field.index_grid(SX, SY)
    out_pts = np.zeros(SX.shape, dtype=bool)
    in_pts = np.zeros(SX.shape, dtype=bool)
    for i, fr in enumerate(regions):
        if fr.B <= 0:
            continue
        if fr.direction == OUT:
            out_pts |= sidx == i
        else:
            in_pts |= sidx == i
    if out_pts.any():
        ax.plot(SX[out_pts], SY[out_pts], "o", ms=3.5, color=SYMBOL_COLOR, zorder=3)
    if in_pts.any():
        ax.plot(SX[in_pts], SY[in_pts], "x", ms=6, mew=1.3, color=SYMBOL_COLOR, zorder=3)

    # 4. 区域编号与 B 的大小
    k = max(1, resolution // 150)
    sub = idx[::k, ::k]
    for i, fr in enumerate(regions):
        pt = _label_point(sub == i, xs[::k], ys[::k])
        if pt is None:
            continue
        ax.text(
            pt[0],
            pt[1],
            f"{region_tag(i)} B={fmt_num(fr.B)}",
            ha="center",
            va="center",
            fontsize=11,
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", alpha=0.85),
        )


def draw_particle_start(ax, particle: Particle, view) -> None:
    xmin, xmax, ymin, ymax = view
    span = max(xmax - xmin, ymax - ymin)
    color = particle_color(particle.q)
    if particle.speed > 0:
        a = math.radians(particle.angle_deg)
        length = 0.1 * span
        ax.annotate(
            "",
            xy=(particle.x0 + length * math.cos(a), particle.y0 + length * math.sin(a)),
            xytext=(particle.x0, particle.y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=18),
            zorder=5,
        )
        ax.text(
            particle.x0 + 1.25 * length * math.cos(a),
            particle.y0 + 1.25 * length * math.sin(a),
            "$v_0$",
            color=color,
            ha="center",
            va="center",
            fontsize=12,
            zorder=5,
        )
    ax.plot([particle.x0], [particle.y0], "o", ms=7, mfc="white", mec=color, mew=1.5, zorder=5)


def draw_scene(ax, field: MagneticField, particle: Particle, view) -> None:
    """清空坐标轴并画出磁场、坐标网格和粒子初始位置。"""
    xmin, xmax, ymin, ymax = view
    ax.clear()
    draw_field(ax, field, view)
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.axvline(0, color="#999999", lw=0.8, zorder=1)
    ax.grid(True, ls=":", color="#bbbbbb", zorder=1)
    draw_particle_start(ax, particle, view)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0, labelpad=10)
    ax.set_title(r"$\bullet$  磁场垂直纸面向外        $\times$  磁场垂直纸面向里", fontsize=11, color="#555555")
