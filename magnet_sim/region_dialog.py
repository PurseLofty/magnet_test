"""添加 / 编辑磁场区域的对话框。"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict, Optional

from .expr import ExprError, evaluate_number
from .field import IN, OUT, FieldRegion
from .regions import SHAPES, default_inputs, make_region

EXPR_HELP = (
    "可用变量：x、y、r（r 为到原点的距离，r = √(x²+y²)）\n"
    "比较：>  <  >=  <=（也可以写 ≥ ≤），支持连写，如 -1 < x < 1\n"
    "组合：and（且）、or（或）、not（非）\n"
    "运算：+ - * / ^（乘方），2x 可省略乘号\n"
    "函数：sqrt abs sin cos tan atan2 exp ln；常数：pi\n"
    "例：x > 0 and x^2 + y^2 < 4      y > x      1 < r < 2      abs(x) + abs(y) < 2"
)


class RegionDialog(tk.Toplevel):
    """模态对话框；确定后 ``self.result`` 为新的 FieldRegion，取消则为 None。

    check_area(region) 用于检查区域在当前视图内是否有面积，返回 False 时会提示用户。
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        initial: Optional[FieldRegion] = None,
        check_area: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result: Optional[FieldRegion] = None
        self._check_area = check_area

        self._labels = [s.label for s in SHAPES.values()]
        self._kinds = list(SHAPES.keys())
        kind = initial.region.kind if initial else "circle"
        self._inputs: Dict[str, Dict[str, str]] = {k: default_inputs(k) for k in self._kinds}
        if initial:
            self._inputs[kind].update(initial.region.inputs)

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="区域形状").grid(row=0, column=0, sticky="w", pady=4)
        self.shape_var = tk.StringVar(value=SHAPES[kind].label)
        shape_box = ttk.Combobox(body, textvariable=self.shape_var, values=self._labels, state="readonly", width=16)
        shape_box.grid(row=0, column=1, sticky="w", pady=4)
        shape_box.bind("<<ComboboxSelected>>", lambda e: self._on_shape_change())

        self.param_frame = ttk.LabelFrame(body, text="形状参数", padding=8)
        self.param_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        self._param_vars: Dict[str, tk.StringVar] = {}
        self._current_kind = kind

        field_frame = ttk.LabelFrame(body, text="磁场", padding=8)
        field_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(field_frame, text="磁感应强度 B").grid(row=0, column=0, sticky="w")
        b_text = initial.B_text if initial and initial.B_text else (fmt(initial.B) if initial else "1")
        self.b_var = tk.StringVar(value=b_text)
        ttk.Entry(field_frame, textvariable=self.b_var, width=12).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(field_frame, text="（≥ 0；填 0 表示无磁场，可用来挖空其他区域）", foreground="#666").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Label(field_frame, text="方向").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dir_var = tk.StringVar(value=initial.direction if initial else OUT)
        dir_frame = ttk.Frame(field_frame)
        dir_frame.grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Radiobutton(dir_frame, text="穿出纸面（·）", variable=self.dir_var, value=OUT).pack(side="left", padx=(6, 12))
        ttk.Radiobutton(dir_frame, text="穿入纸面（×）", variable=self.dir_var, value=IN).pack(side="left")

        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="确定", command=self._on_ok).pack(side="left", padx=4)
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="left", padx=4)

        self._build_params()
        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self.destroy())

        self.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        self.geometry(f"+{px + max(0, (pw - self.winfo_width()) // 2)}+{py + max(0, (ph - self.winfo_height()) // 3)}")
        self.grab_set()
        self.focus_set()

    # ------------------------------------------------------------------
    def _save_param_inputs(self) -> None:
        self._inputs[self._current_kind] = {k: v.get() for k, v in self._param_vars.items()}

    def _on_shape_change(self) -> None:
        self._save_param_inputs()
        self._current_kind = self._kinds[self._labels.index(self.shape_var.get())]
        self._build_params()

    def _build_params(self) -> None:
        for child in self.param_frame.winfo_children():
            child.destroy()
        self._param_vars = {}
        spec = SHAPES[self._current_kind]
        values = self._inputs[self._current_kind]
        if not spec.params:
            ttk.Label(self.param_frame, text="整个平面都有磁场，不需要参数。").grid(row=0, column=0, sticky="w")
        for row, p in enumerate(spec.params):
            ttk.Label(self.param_frame, text=p.label).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=values.get(p.key, p.default))
            self._param_vars[p.key] = var
            if p.choices:
                w = ttk.Combobox(self.param_frame, textvariable=var, values=p.choices, state="readonly", width=22)
            else:
                w = ttk.Entry(self.param_frame, textvariable=var, width=46 if p.wide else 12)
            w.grid(row=row, column=1, sticky="w", padx=6, pady=3)
        if self._current_kind == "expr":
            ttk.Label(
                self.param_frame, text=EXPR_HELP, foreground="#555", justify="left"
            ).grid(row=len(spec.params), column=0, columnspan=2, sticky="w", pady=(6, 0))
        elif spec.params:
            ttk.Label(
                self.param_frame, text="数值处可以写算式，如 sqrt(3)、2√3、pi/2", foreground="#666"
            ).grid(row=len(spec.params), column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _on_ok(self) -> None:
        self._save_param_inputs()
        try:
            region = make_region(self._current_kind, self._inputs[self._current_kind])
            B = evaluate_number(self.b_var.get(), "磁感应强度 B")
            if B < 0:
                raise ExprError("磁感应强度 B 不能为负数；方向请用“穿出/穿入”选择")
        except ExprError as exc:
            messagebox.showerror("输入有误", str(exc), parent=self)
            return
        if self._check_area is not None and not self._check_area(region):
            if not messagebox.askyesno("提示", "这个区域在当前显示范围内看不到（没有面积）。\n仍然保存吗？", parent=self):
                return
        self.result = FieldRegion(region, B, self.dir_var.get(), B_text=self.b_var.get().strip())
        self.destroy()


def fmt(v: float) -> str:
    return f"{v:.6g}"
