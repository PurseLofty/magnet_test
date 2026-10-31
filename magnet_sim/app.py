"""主窗口：左侧为演示画面，右侧为粒子、磁场区域和显示范围的设置。"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .expr import ExprError, evaluate_number  # noqa: E402
from .field import DIRECTION_LABELS, OUT, FieldRegion, MagneticField  # noqa: E402
from .physics import STOP_MESSAGES, Particle, Trajectory, simulate  # noqa: E402
from .plotting import draw_scene, fmt_num, particle_color, region_tag, setup_fonts  # noqa: E402
from .region_dialog import RegionDialog  # noqa: E402
from .regions import make_region  # noqa: E402

FRAME_MS = 15  # 动画刷新间隔（毫秒）
SECONDS_TO_CROSS_VIEW = 6.0  # 1× 速度下，粒子走过一个视图宽度所需的真实时间（秒）

IDLE, PLAYING, PAUSED, FINISHED = "idle", "playing", "paused", "finished"


def default_field() -> MagneticField:
    return MagneticField(
        [FieldRegion(make_region("circle", {"cx": "0", "cy": "0", "r": "2"}), 1.0, OUT, B_text="1")]
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("带电粒子在磁场中的运动 · 演示")
        self._setup_tk_fonts()
        setup_fonts()

        self.field = default_field()
        self.trajectory: Optional[Trajectory] = None
        self.state = IDLE
        self.sim_t = 0.0
        self._last_tick = 0.0
        self._after_id: Optional[str] = None
        self._bg = None
        self._anim_artists: List = []
        self._applied_inputs: Optional[Tuple] = None
        self._particle: Optional[Particle] = None
        self._view: Optional[Tuple[float, float, float, float]] = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<space>", self._on_space)

        w, h = min(1400, self.winfo_screenwidth() - 60), min(860, self.winfo_screenheight() - 100)
        self.geometry(f"{w}x{h}+20+20")
        self.after(50, lambda: self.apply_inputs(force=True))

    # ------------------------------------------------------------------ 界面
    def _setup_tk_fonts(self) -> None:
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(size=11)
            except tk.TclError:
                pass
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=26)
        style.configure("Big.TButton", padding=(10, 4))

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # 左侧：画布 + 播放控制
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.95)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("draw_event", self._on_draw)

        bar = ttk.Frame(left, padding=(8, 6))
        bar.grid(row=1, column=0, sticky="ew")
        self.play_btn = ttk.Button(bar, text="▶ 开始", width=11, style="Big.TButton", command=self.on_play)
        self.play_btn.pack(side="left", padx=3)
        ttk.Button(bar, text="⟲ 重置", style="Big.TButton", command=self.on_reset).pack(side="left", padx=3)
        ttk.Button(bar, text="显示完整轨迹", style="Big.TButton", command=self.on_show_full).pack(side="left", padx=3)
        ttk.Label(bar, text="   播放速度").pack(side="left")
        self.speed_var = tk.DoubleVar(value=0.0)  # log10(倍数)
        ttk.Scale(bar, from_=-1.0, to=1.0, variable=self.speed_var, length=180, command=self._on_speed).pack(
            side="left", padx=4
        )
        self.speed_label = ttk.Label(bar, text="1.0×", width=6)
        self.speed_label.pack(side="left")
        self.status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status_var, padding=(10, 0, 10, 6), foreground="#333").grid(
            row=2, column=0, sticky="w"
        )

        # 右侧：设置面板
        right = ttk.Frame(self, padding=(6, 8, 10, 8))
        right.grid(row=0, column=1, sticky="ns")
        self._build_particle_panel(right)
        self._build_region_panel(right)
        self._build_view_panel(right)

    def _entry(self, parent, var, row, col, width=8):
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=col, sticky="w", padx=4, pady=3)
        e.bind("<Return>", lambda ev: self.apply_inputs())
        e.bind("<FocusOut>", lambda ev: self.apply_inputs())
        return e

    def _build_particle_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="带电粒子", padding=8)
        box.pack(fill="x", pady=(0, 8))

        self.sign_var = tk.StringVar(value="+")
        ttk.Label(box, text="电性").grid(row=0, column=0, sticky="w")
        sign = ttk.Frame(box)
        sign.grid(row=0, column=1, columnspan=3, sticky="w")
        for text, val in (("正电荷 +", "+"), ("负电荷 −", "-")):
            ttk.Radiobutton(sign, text=text, value=val, variable=self.sign_var, command=self.apply_inputs).pack(
                side="left", padx=4
            )

        self.q_var = tk.StringVar(value="1")
        self.m_var = tk.StringVar(value="1")
        ttk.Label(box, text="电荷量 |q|").grid(row=1, column=0, sticky="w")
        self._entry(box, self.q_var, 1, 1)
        ttk.Label(box, text="质量 m").grid(row=1, column=2, sticky="w")
        self._entry(box, self.m_var, 1, 3)

        self.x0_var = tk.StringVar(value="-4")
        self.y0_var = tk.StringVar(value="0")
        ttk.Label(box, text="初始 x₀").grid(row=2, column=0, sticky="w")
        self._entry(box, self.x0_var, 2, 1)
        ttk.Label(box, text="初始 y₀").grid(row=2, column=2, sticky="w")
        self._entry(box, self.y0_var, 2, 3)

        self.v_var = tk.StringVar(value="2")
        self.angle_var = tk.StringVar(value="0")
        ttk.Label(box, text="速度大小 v").grid(row=3, column=0, sticky="w")
        self._entry(box, self.v_var, 3, 1)
        ttk.Label(box, text="方向 θ (°)").grid(row=3, column=2, sticky="w")
        self._entry(box, self.angle_var, 3, 3)
        ttk.Label(
            box, text="θ：与 x 轴正方向的夹角，逆时针为正\n数值处可写算式，如 sqrt(3)、2√3", foreground="#666"
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _build_region_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="磁场区域", padding=8)
        box.pack(fill="both", expand=True, pady=(0, 8))
        cols = ("no", "shape", "B", "dir")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=7, selectmode="browse")
        for col, text, width, anchor in (
            ("no", "编号", 44, "center"),
            ("shape", "区域", 230, "w"),
            ("B", "B", 52, "center"),
            ("dir", "方向", 64, "center"),
        ):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "shape"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<Double-1>", lambda e: self.on_edit_region())
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)

        btns = ttk.Frame(box)
        btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for text, cmd in (
            ("添加", self.on_add_region),
            ("编辑", self.on_edit_region),
            ("删除", self.on_delete_region),
            ("上移", lambda: self.on_move_region(-1)),
            ("下移", lambda: self.on_move_region(1)),
        ):
            ttk.Button(btns, text=text, width=5, command=cmd).pack(side="left", padx=2)
        ttk.Label(
            box,
            text="区域重叠时，列表中靠前的优先；\n不在任何区域内的地方没有磁场；\n【不可见】表示在显示范围外或被前面的区域盖住。",
            foreground="#666",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._refresh_tree()

    def _build_view_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="显示范围", padding=8)
        box.pack(fill="x")
        self.xmin_var, self.xmax_var = tk.StringVar(value="-5"), tk.StringVar(value="5")
        self.ymin_var, self.ymax_var = tk.StringVar(value="-5"), tk.StringVar(value="5")
        ttk.Label(box, text="x 从").grid(row=0, column=0, sticky="w")
        self._entry(box, self.xmin_var, 0, 1, 7)
        ttk.Label(box, text="到").grid(row=0, column=2)
        self._entry(box, self.xmax_var, 0, 3, 7)
        ttk.Label(box, text="y 从").grid(row=1, column=0, sticky="w")
        self._entry(box, self.ymin_var, 1, 1, 7)
        ttk.Label(box, text="到").grid(row=1, column=2)
        self._entry(box, self.ymax_var, 1, 3, 7)

    # ------------------------------------------------------------------ 输入
    def _input_tuple(self) -> Tuple:
        return tuple(
            v.get().strip()
            for v in (
                self.sign_var, self.q_var, self.m_var, self.x0_var, self.y0_var, self.v_var, self.angle_var,
                self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var,
            )
        )

    def _read_particle(self) -> Particle:
        q = evaluate_number(self.q_var.get(), "电荷量 |q|")
        m = evaluate_number(self.m_var.get(), "质量 m")
        v = evaluate_number(self.v_var.get(), "速度大小 v")
        if q < 0:
            raise ExprError("电荷量请填大小（≥ 0），正负请用“电性”选择")
        if m <= 0:
            raise ExprError("质量 m 必须大于 0")
        if v < 0:
            raise ExprError("速度大小 v 不能为负数，方向请用 θ 表示")
        return Particle(
            q=q if self.sign_var.get() == "+" else -q,
            m=m,
            x0=evaluate_number(self.x0_var.get(), "初始 x₀"),
            y0=evaluate_number(self.y0_var.get(), "初始 y₀"),
            speed=v,
            angle_deg=evaluate_number(self.angle_var.get(), "速度方向 θ"),
        )

    def _read_view(self) -> Tuple[float, float, float, float]:
        xmin = evaluate_number(self.xmin_var.get(), "x 范围")
        xmax = evaluate_number(self.xmax_var.get(), "x 范围")
        ymin = evaluate_number(self.ymin_var.get(), "y 范围")
        ymax = evaluate_number(self.ymax_var.get(), "y 范围")
        if xmin >= xmax or ymin >= ymax:
            raise ExprError("显示范围的“从”必须小于“到”")
        return xmin, xmax, ymin, ymax

    def apply_inputs(self, force: bool = False) -> bool:
        """读取输入；若有变化则停止动画并重画场景。输入有误时提示并返回 False。"""
        current = self._input_tuple()
        if not force and current == self._applied_inputs:
            return True
        try:
            particle = self._read_particle()
            view = self._read_view()
        except ExprError as exc:
            self._applied_inputs = current  # 避免同一错误在失去焦点时反复弹窗
            messagebox.showerror("输入有误", str(exc), parent=self)
            return False
        self._applied_inputs = current
        self._particle, self._view = particle, view
        self._stop_timer()
        self.trajectory = None
        self.state = IDLE
        self._redraw_scene()
        self._update_buttons()
        self.status_var.set("设置好粒子和磁场后，点击“开始”（或按空格键）")
        return True

    def _inputs_ok(self) -> bool:
        """在开始播放前确认当前输入有效（包括之前出错尚未改正的情况）。"""
        try:
            self._read_particle()
            self._read_view()
        except ExprError as exc:
            messagebox.showerror("输入有误", str(exc), parent=self)
            return False
        return self.apply_inputs()

    # ------------------------------------------------------------------ 绘图
    def _redraw_scene(self) -> None:
        if self._particle is None or self._view is None:
            return
        draw_scene(self.ax, self.field, self._particle, self._view)
        self._refresh_tree()
        color = particle_color(self._particle.q)
        (trail,) = self.ax.plot([], [], "-", color=color, lw=2.2, animated=True, zorder=6)
        (dot,) = self.ax.plot([], [], "o", ms=13, color=color, animated=True, zorder=7)
        (sign,) = self.ax.plot(
            [], [], "+" if self._particle.q >= 0 else "_", ms=10, mew=2.2, color="white", animated=True, zorder=8
        )
        self.trail, self.dot, self.sign_marker = trail, dot, sign
        self._anim_artists = [trail, dot, sign]
        self.canvas.draw()

    def _on_draw(self, event) -> None:
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_animated()

    def _draw_animated(self) -> None:
        for artist in self._anim_artists:
            self.ax.draw_artist(artist)

    def _blit(self) -> None:
        if self._bg is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self._bg)
        self._draw_animated()
        self.canvas.blit(self.fig.bbox)

    def _set_progress(self, t: float) -> None:
        traj = self.trajectory
        if traj is None:
            return
        i, x, y = traj.state_at(t)
        self.trail.set_data(np.append(traj.x[:i], x), np.append(traj.y[:i], y))
        self.dot.set_data([x], [y])
        self.sign_marker.set_data([x], [y])
        self._blit()

        idx = int(traj.region[max(i - 1, 0)])
        if idx >= 0 and self.field.regions[idx].B > 0:
            fr = self.field.regions[idx]
            where = f"区域 {region_tag(idx)}（B={fmt_num(fr.B)}，{DIRECTION_LABELS[fr.direction]}）"
        else:
            where = "无磁场区域"
        text = f"t = {t:.3f}    位置 ({x:.3f}, {y:.3f})    {where}"
        if self.state == FINISHED:
            text += f"    —— {STOP_MESSAGES[traj.stop_reason]}"
        self.status_var.set(text)

    # ------------------------------------------------------------------ 播放控制
    def _compute(self) -> bool:
        if not self._inputs_ok():
            return False
        p, view = self._particle, self._view
        xmin, xmax, ymin, ymax = view
        if not (xmin <= p.x0 <= xmax and ymin <= p.y0 <= ymax):
            messagebox.showwarning("提示", "粒子的初始位置不在显示范围内，请调整初始位置或显示范围。", parent=self)
            return False
        self.status_var.set("正在计算轨迹……")
        self.update_idletasks()
        self.trajectory = simulate(p, self.field, view)
        self.sim_t = 0.0
        return True

    def on_play(self) -> None:
        if self._input_tuple() != self._applied_inputs and not self.apply_inputs():
            return
        if self.state == PLAYING:
            self.state = PAUSED
            self._stop_timer()
        elif self.state == PAUSED:
            self.state = PLAYING
            self._start_timer()
        else:  # IDLE / FINISHED：从头开始
            if not self._compute():
                return
            if self.trajectory.duration <= 0:
                self.state = FINISHED
                self._set_progress(0.0)
            else:
                self.state = PLAYING
                self._start_timer()
        self._update_buttons()

    def on_reset(self) -> None:
        self.apply_inputs(force=True)

    def on_show_full(self) -> None:
        if self.state in (IDLE, FINISHED) or self.trajectory is None:
            if not self._compute():
                return
        self._stop_timer()
        self.state = FINISHED
        self.sim_t = self.trajectory.duration
        self._set_progress(self.sim_t)
        self._update_buttons()

    def _on_space(self, event) -> None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, tk.Button, ttk.Button, ttk.Radiobutton)):
            return
        if isinstance(event.widget, (tk.Toplevel,)) or event.widget.winfo_toplevel() is not self:
            return
        self.on_play()

    def _on_speed(self, _value=None) -> None:
        self.speed_label.config(text=f"{self._speed_factor():.2g}×")

    def _speed_factor(self) -> float:
        return 10 ** self.speed_var.get()

    def _start_timer(self) -> None:
        self._stop_timer()
        self._last_tick = time.perf_counter()
        self._after_id = self.after(FRAME_MS, self._tick)

    def _stop_timer(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self) -> None:
        self._after_id = None
        if self.state != PLAYING or self.trajectory is None:
            return
        now = time.perf_counter()
        elapsed, self._last_tick = now - self._last_tick, now
        xmin, xmax, ymin, ymax = self._view
        rate = max(xmax - xmin, ymax - ymin) / (SECONDS_TO_CROSS_VIEW * self._particle.speed)
        self.sim_t += elapsed * rate * self._speed_factor()
        if self.sim_t >= self.trajectory.duration:
            self.sim_t = self.trajectory.duration
            self.state = FINISHED
            self._update_buttons()
        self._set_progress(self.sim_t)
        if self.state == PLAYING:
            self._after_id = self.after(FRAME_MS, self._tick)

    def _update_buttons(self) -> None:
        self.play_btn.config(
            text={IDLE: "▶ 开始", PLAYING: "⏸ 暂停", PAUSED: "▶ 继续", FINISHED: "▶ 重新播放"}[self.state]
        )

    # ------------------------------------------------------------------ 区域管理
    def _refresh_tree(self, select: Optional[int] = None) -> None:
        if select is None:
            select = self._selected_index()
        shown = None
        if self._view is not None:
            xmin, xmax, ymin, ymax = self._view
            X, Y = np.meshgrid(np.linspace(xmin, xmax, 240), np.linspace(ymin, ymax, 240))
            shown = set(np.unique(self.field.index_grid(X, Y)).tolist())
        self.tree.delete(*self.tree.get_children())
        for i, fr in enumerate(self.field.regions):
            direction = "—" if fr.B == 0 else ("· 出" if fr.direction == OUT else "× 入")
            desc = fr.region.describe()
            if shown is not None and i not in shown:
                desc = "【不可见】" + desc
            self.tree.insert("", "end", iid=str(i), values=(region_tag(i), desc, fmt_num(fr.B), direction))
        if select is not None and 0 <= select < len(self.field.regions):
            self.tree.selection_set(str(select))
            self.tree.see(str(select))

    def _selected_index(self) -> Optional[int]:
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _region_visible(self, region) -> bool:
        if self._view is None:
            return True
        xmin, xmax, ymin, ymax = self._view
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
        return bool(np.any(region.contains(X, Y)))

    def _field_changed(self, select: Optional[int]) -> None:
        self._refresh_tree(select)
        self.apply_inputs(force=True)  # 会重画场景并刷新列表

    def on_add_region(self) -> None:
        dlg = RegionDialog(self, "添加磁场区域", check_area=self._region_visible)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.field.regions.append(dlg.result)
            self._field_changed(len(self.field.regions) - 1)

    def on_edit_region(self) -> None:
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("提示", "请先在列表中选中一个区域。", parent=self)
            return
        dlg = RegionDialog(self, f"编辑磁场区域 {region_tag(i)}", self.field.regions[i], check_area=self._region_visible)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.field.regions[i] = dlg.result
            self._field_changed(i)

    def on_delete_region(self) -> None:
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("提示", "请先在列表中选中一个区域。", parent=self)
            return
        if messagebox.askyesno("确认", f"删除区域 {region_tag(i)}？", parent=self):
            del self.field.regions[i]
            self._field_changed(min(i, len(self.field.regions) - 1))

    def on_move_region(self, delta: int) -> None:
        i = self._selected_index()
        if i is None:
            return
        j = i + delta
        if not 0 <= j < len(self.field.regions):
            return
        regs = self.field.regions
        regs[i], regs[j] = regs[j], regs[i]
        self._field_changed(j)

    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._stop_timer()
        self.quit()
        self.destroy()


def _enable_windows_dpi_awareness() -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main() -> None:
    _enable_windows_dpi_awareness()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
