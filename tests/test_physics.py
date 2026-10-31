import math

import numpy as np
import pytest

from magnet_sim.expr import ExprError
from magnet_sim.field import IN, OUT, FieldRegion, MagneticField
from magnet_sim.physics import (
    STOP_LEFT_VIEW,
    Particle,
    cyclotron_period,
    cyclotron_radius,
    simulate,
)
from magnet_sim.regions import make_region

VIEW = (-10.0, 10.0, -10.0, 10.0)


def field_of(*items):
    return MagneticField([FieldRegion(make_region(k, inp), B, d) for k, inp, B, d in items])


def test_uniform_field_radius_period_and_sense():
    # 正电荷在向外的磁场中顺时针转：q=1, m=2, v=3, B=1.5 → r = 4
    f = field_of(("plane", {}, 1.5, OUT))
    p = Particle(q=1, m=2, x0=0, y0=0, speed=3, angle_deg=0)
    traj = simulate(p, f, VIEW, max_path_factor=3)
    r = cyclotron_radius(2, 3, 1, 1.5)
    assert r == pytest.approx(4.0)
    # 圆心在 (0, -r)
    d = np.hypot(traj.x - 0, traj.y + r)
    assert np.allclose(d, r, atol=1e-9)
    # 一开始向下偏（顺时针）
    assert traj.y[5] < 0
    # 周期：回到出发点的时间
    T = cyclotron_period(2, 1, 1.5)
    i = np.argmin(np.abs(traj.t - T))
    assert math.hypot(traj.x[i], traj.y[i]) < 0.05


def test_negative_charge_turns_the_other_way():
    f = field_of(("plane", {}, 1.0, OUT))
    p = Particle(q=-1, m=1, x0=0, y0=0, speed=1, angle_deg=0)
    traj = simulate(p, f, VIEW, max_path_factor=1)
    assert traj.y[5] > 0
    f_in = field_of(("plane", {}, 1.0, IN))
    traj2 = simulate(Particle(q=1, m=1, x0=0, y0=0, speed=1, angle_deg=0), f_in, VIEW, max_path_factor=1)
    assert traj2.y[5] > 0


def test_half_plane_entry_exits_at_two_r():
    # 垂直边界射入 x>0 区域，转半圈后从 (0, -2r) 射出，速度反向
    f = field_of(("half", {"cond": "x >", "value": "0"}, 1.0, OUT))
    p = Particle(q=1, m=1, x0=-3, y0=0, speed=2, angle_deg=0)
    traj = simulate(p, f, VIEW)
    assert traj.stop_reason == STOP_LEFT_VIEW
    inside = np.where(traj.region == 0)[0]
    exit_i = inside[-1] + 1
    assert traj.x[exit_i] == pytest.approx(0.0, abs=1e-9)
    assert traj.y[exit_i] == pytest.approx(-4.0, abs=1e-6)
    assert traj.vx[exit_i] == pytest.approx(-2.0, abs=1e-6)
    assert traj.vy[exit_i] == pytest.approx(0.0, abs=1e-6)


def test_circular_field_radial_in_radial_out():
    # 圆形磁场 R=2，沿半径射入（r=R），偏转 90° 后从 (0,-2) 沿半径射出
    f = field_of(("circle", {"cx": "0", "cy": "0", "r": "2"}, 1.0, OUT))
    p = Particle(q=1, m=1, x0=-4, y0=0, speed=2, angle_deg=0)
    traj = simulate(p, f, VIEW)
    inside = np.where(traj.region == 0)[0]
    exit_i = inside[-1] + 1
    assert traj.x[exit_i] == pytest.approx(0.0, abs=1e-6)
    assert traj.y[exit_i] == pytest.approx(-2.0, abs=1e-6)
    assert traj.vx[exit_i] == pytest.approx(0.0, abs=1e-6)
    assert traj.vy[exit_i] == pytest.approx(-2.0, abs=1e-6)


def test_priority_and_zero_field_hole():
    # 列表中靠前的区域优先：B=0 的小圆在大圆中"挖洞"
    f = field_of(
        ("circle", {"cx": "0", "cy": "0", "r": "1"}, 0.0, OUT),
        ("circle", {"cx": "0", "cy": "0", "r": "3"}, 2.0, IN),
    )
    assert f.lookup(0, 0) == (0, 0.0)
    assert f.lookup(2, 0) == (1, -2.0)
    assert f.lookup(5, 0) == (-1, 0.0)
    X, Y = np.meshgrid([0.0, 2.0, 5.0], [0.0])
    assert f.index_grid(X, Y).tolist() == [[0, 1, -1]]


def test_two_opposite_regions_speed_conserved():
    f = field_of(
        ("half", {"cond": "x <", "value": "0"}, 1.0, OUT),
        ("expr", {"text": "x >= 0"}, 2.0, IN),
    )
    p = Particle(q=1, m=1, x0=0, y0=0, speed=1.5, angle_deg=30)
    traj = simulate(p, f, VIEW, max_path_factor=10)
    assert np.allclose(np.hypot(traj.vx, traj.vy), 1.5)
    assert set(np.unique(traj.region)) <= {0, 1}


def test_invalid_regions():
    with pytest.raises(ExprError):
        make_region("circle", {"cx": "0", "cy": "0", "r": "-1"})
    with pytest.raises(ExprError):
        make_region("annulus", {"cx": "0", "cy": "0", "r1": "2", "r2": "1"})
    with pytest.raises(ExprError):
        make_region("expr", {"text": "x + y"})
    with pytest.raises(ExprError):
        make_region("expr", {"text": "sqrt(x, y) > 1"})
