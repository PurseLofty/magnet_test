"""带电粒子在垂直纸面的磁场中运动的数值计算。

洛伦兹力 F = q v × B，B = Bz·ẑ 时：
    ax = (q/m)·vy·Bz,  ay = -(q/m)·vx·Bz
即速度以角速度 ω = -q·Bz/m 旋转（ω > 0 为逆时针）。

在同一区域内磁场均匀，粒子做匀速圆周运动，因此每一步都用圆弧的
解析解推进（没有积分误差，轨道不会"越转越大"）。当一步跨越了区域边界时，
用二分法找到越界时刻，使粒子准确地在边界处换用新的磁场。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .field import MagneticField

STOP_LEFT_VIEW = "left_view"
STOP_MAX_PATH = "max_path"
STOP_MAX_STEPS = "max_steps"
STOP_AT_REST = "at_rest"

STOP_MESSAGES = {
    STOP_LEFT_VIEW: "粒子已离开显示区域",
    STOP_MAX_PATH: "已达到最长演示路程",
    STOP_MAX_STEPS: "已达到最长演示时间",
    STOP_AT_REST: "速度为 0，粒子静止",
}


@dataclass
class Particle:
    q: float  # 电荷量（带符号）
    m: float  # 质量，> 0
    x0: float
    y0: float
    speed: float  # 速度大小，>= 0
    angle_deg: float  # 速度方向：与 x 轴正方向的夹角，逆时针为正（角度制）

    @property
    def v0(self) -> Tuple[float, float]:
        a = math.radians(self.angle_deg)
        return self.speed * math.cos(a), self.speed * math.sin(a)


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    region: np.ndarray  # 每个采样点所在的区域序号（-1 为无场）
    stop_reason: str

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def state_at(self, t: float) -> Tuple[int, float, float]:
        """返回 (已走过的采样点个数, x, y)，在相邻采样点之间线性插值。"""
        t = min(max(t, 0.0), self.duration)
        i = int(np.searchsorted(self.t, t, side="right"))
        if i >= len(self.t):
            return len(self.t), float(self.x[-1]), float(self.y[-1])
        t0, t1 = self.t[i - 1], self.t[i]
        f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        x = self.x[i - 1] + f * (self.x[i] - self.x[i - 1])
        y = self.y[i - 1] + f * (self.y[i] - self.y[i - 1])
        return i, float(x), float(y)

    def velocity_at(self, t: float) -> Tuple[float, float]:
        i, _, _ = self.state_at(t)
        i = min(max(i - 1, 0), len(self.t) - 1)
        return float(self.vx[i]), float(self.vy[i])


def advance(x, y, vx, vy, omega, h):
    """在角速度 omega 下沿圆弧（omega = 0 时沿直线）运动时间 h 后的状态。"""
    if omega == 0.0:
        return x + vx * h, y + vy * h, vx, vy
    phi = omega * h
    c, s = math.cos(phi), math.sin(phi)
    # 1 - cos(phi) 在 phi 很小时用 2sin²(phi/2) 计算，避免精度损失
    one_minus_c = 2.0 * math.sin(phi / 2.0) ** 2
    nx = x + (vx * s - vy * one_minus_c) / omega
    ny = y + (vy * s + vx * one_minus_c) / omega
    return nx, ny, vx * c - vy * s, vx * s + vy * c


def cyclotron_radius(m: float, speed: float, q: float, B: float) -> float:
    """r = mv / (|q|B)，B = 0 时返回 inf。"""
    return math.inf if q == 0 or B == 0 else m * speed / (abs(q) * abs(B))


def cyclotron_period(m: float, q: float, B: float) -> float:
    """T = 2πm / (|q|B)，B = 0 时返回 inf。"""
    return math.inf if q == 0 or B == 0 else 2 * math.pi * m / (abs(q) * abs(B))


def simulate(
    particle: Particle,
    field: MagneticField,
    view: Tuple[float, float, float, float],
    max_path_factor: float = 40.0,
    max_steps: int = 150_000,
    step_fraction: float = 1.0 / 800.0,
    max_angle_deg: float = 3.0,
) -> Trajectory:
    """计算粒子轨迹，直到离开视图、走过 max_path_factor 倍视图尺寸或达到步数上限。

    view = (xmin, xmax, ymin, ymax)。
    """
    if particle.m <= 0:
        raise ValueError("质量必须大于 0")
    xmin, xmax, ymin, ymax = view
    span = max(xmax - xmin, ymax - ymin)
    margin = 0.03 * span
    lo_x, hi_x, lo_y, hi_y = xmin - margin, xmax + margin, ymin - margin, ymax + margin

    x, y = float(particle.x0), float(particle.y0)
    vx, vy = particle.v0
    speed = particle.speed
    qm = particle.q / particle.m
    t = 0.0
    idx, bz = field.lookup(x, y)

    ts, xs, ys, vxs, vys, regs = [t], [x], [y], [vx], [vy], [idx]

    if speed <= 0:
        return Trajectory(*(np.array(a) for a in (ts, xs, ys, vxs, vys)), np.array(regs), STOP_AT_REST)

    dt_line = span * step_fraction / speed
    max_angle = math.radians(max_angle_deg)
    max_time = max_path_factor * span / speed
    reason = STOP_MAX_STEPS

    for _ in range(max_steps):
        omega = -qm * bz
        dt = dt_line if omega == 0.0 else min(dt_line, max_angle / abs(omega))
        nx, ny, nvx, nvy = advance(x, y, vx, vy, omega, dt)
        nidx, nbz = field.lookup(nx, ny)
        if nidx != idx:
            # 二分查找越过边界的时刻 h ∈ (lo, hi]
            lo, hi = 0.0, dt
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                mx, my, _, _ = advance(x, y, vx, vy, omega, mid)
                if field.region_index(mx, my) == idx:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= dt * 1e-12:
                    break
            dt = hi
            nx, ny, nvx, nvy = advance(x, y, vx, vy, omega, dt)
            nidx, nbz = field.lookup(nx, ny)
        x, y, vx, vy = nx, ny, nvx, nvy
        idx, bz = nidx, nbz
        t += dt
        ts.append(t)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        regs.append(idx)
        if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
            reason = STOP_LEFT_VIEW
            break
        if t >= max_time:
            reason = STOP_MAX_PATH
            break

    return Trajectory(
        np.array(ts), np.array(xs), np.array(ys), np.array(vxs), np.array(vys), np.array(regs), reason
    )
