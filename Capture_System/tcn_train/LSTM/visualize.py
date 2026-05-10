"""
visualize.py — CNN-GRU 流式推理可视化

超声瀑布图（单通道）上叠加：
  红色：预测力矩 (Nm)
  蓝色：真实力矩 (Nm)
  绿色：acc_z (m/s²)

直接运行，交互式选择数据段：
    python visualize.py
    python visualize.py --ckpt checkpoints/best.pt --fps 30 --channel 1
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.font_manager as fm
import matplotlib

def _setup_font():
    prefer = ["Microsoft YaHei", "SimHei", "SimSun", "FangSong",
              "STHeiti", "Noto Sans CJK SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in prefer:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_font()

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
except ImportError:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="k", foreground="w")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from data_loader import _load_segment, _nearest_align, _preprocess_us_channel, KIN_DIM
from dataset import DualChannelDataset
from model import TorqueCNNGRU

US_S0, US_S1 = 150, 850


# ── checkpoint 加载 ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, device):
    ck  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["args"]
    model = TorqueCNNGRU(
        kin_dim        = KIN_DIM,
        us_feat        = cfg.get("us_feat",    64),
        gru_hidden     = cfg.get("gru_hidden", 128),
        gru_layers     = cfg.get("gru_layers", 2),
        dropout        = 0.0,
        use_ultrasound = not cfg.get("no_ultrasound", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return (model,
            ck["kin_mean"], ck["kin_std"],
            float(ck["lbl_mean"]), float(ck["lbl_std"]),
            cfg.get("window", 50),
            cfg.get("ref_frames", 30))


# ── 超声预处理（全通道）──────────────────────────────────────────────────────

def load_us_all(seg_dir: Path, ref_ts: np.ndarray, ref_frames: int):
    """返回 us_cur {ch: (N,1000)}, us_ref {ch: (700,)}"""
    us_df = pd.read_csv(seg_dir / "input" / "ultrasound.csv")
    dcols = sorted([c for c in us_df.columns if c.startswith("d") and c[1:].isdigit()],
                   key=lambda x: int(x[1:]))
    N = len(ref_ts)
    us_env = {}
    us_ref = {}
    for ch in sorted(us_df["channel"].unique()):
        ch = int(ch)
        sub   = us_df[us_df["channel"] == ch].reset_index(drop=True)
        ch_ts = sub["timestamp"].values.astype(np.float64)
        idx   = _nearest_align(ch_ts, ref_ts)
        raw   = sub[dcols].values[idx].astype(np.float32)
        env   = _preprocess_us_channel(raw)                    # (N, 1000)
        us_env[ch] = env
        us_ref[ch] = env[:ref_frames, US_S0:US_S1].mean(axis=0)  # (700,)
    return us_env, us_ref


# ── LUT ──────────────────────────────────────────────────────────────────────

def gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


# ── 主窗口 ────────────────────────────────────────────────────────────────────

class InferViewer(QWidget):
    HISTORY = 400

    def __init__(self, seg_dir, model, kin, kin_n, us_env, us_ref,
                 lbl, lbl_mean, lbl_std, W, device, fps, channel):
        super().__init__()
        self.setWindowTitle(f"CNN-GRU 推理 — {seg_dir.name}")
        self.resize(1100, 700)

        self.model    = model
        self.kin      = kin
        self.kin_n    = kin_n
        self.us_env   = us_env    # {ch: (N, 1000)}
        self.us_ref   = us_ref    # {ch: (700,)}
        self.lbl      = lbl
        self.lbl_mean = lbl_mean
        self.lbl_std  = lbl_std
        self.W        = W
        self.device   = device
        self.N        = len(lbl)
        self.channel  = channel
        self.cur      = 0

        self.kin_buf = deque(maxlen=W)
        self.us_buf  = deque(maxlen=W)   # 每帧存 (2, 4, 700)

        H, D = self.HISTORY, US_S1 - US_S0
        self.us_disp   = np.zeros((H, D), dtype=np.float32)
        self.truth_buf = np.zeros(H, dtype=np.float32)
        self.pred_buf  = np.full(H, np.nan, dtype=np.float32)
        self.acc_buf   = np.zeros(H, dtype=np.float32)

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(max(1, int(1000 / fps)))

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(4)

        top = QHBoxLayout()
        top.addWidget(QLabel("通道:"))
        self.combo = QComboBox()
        for ch in sorted(self.us_env.keys()):
            self.combo.addItem(f"Channel {ch}", ch)
        self.combo.setCurrentIndex(self.channel - 1)
        self.combo.currentIndexChanged.connect(
            lambda i: setattr(self, "channel", self.combo.itemData(i)) or
                      self.us_disp.__setitem__(slice(None), 0))
        top.addWidget(self.combo); top.addStretch()
        self.stat_lbl = QLabel("推理中...")
        self.stat_lbl.setStyleSheet("color:#aaa;font-size:11px")
        top.addWidget(self.stat_lbl)
        root.addLayout(top)

        self.pw = pg.PlotWidget(title=f"Channel {self.channel} — 超声 + 推理叠加")
        self.pw.invertY(True)
        self.pw.setLabel("left", "Depth"); self.pw.setLabel("bottom", "Frame")
        self.img = pg.ImageItem()
        self.img.setLookupTable(gray_lut())
        self.pw.addItem(self.img)

        self.c_truth = self.pw.plot(pen=pg.mkPen("#4FC3F7", width=2))
        self.c_pred  = self.pw.plot(pen=pg.mkPen("#EF5350", width=2))
        self.c_acc   = self.pw.plot(pen=pg.mkPen("#66BB6A", width=1.5))
        for item in (self.c_truth, self.c_pred, self.c_acc):
            item.setZValue(10)

        for text, color, yo in [("Ground Truth","#4FC3F7",0),
                                  ("Prediction","#EF5350",20),
                                  ("acc_z","#66BB6A",40)]:
            t = pg.TextItem(text, color=color, anchor=(0,0))
            t.setPos(5, US_S0 + yo); t.setZValue(20)
            self.pw.addItem(t)

        root.addWidget(self.pw, stretch=1)

    def _map(self, arr, vmin, vmax):
        span = vmax - vmin if vmax != vmin else 1.0
        return US_S0 + (arr - vmin) / span * (US_S1 - US_S0)

    def _step(self):
        if self.cur >= self.N:
            self.timer.stop(); return
        i = self.cur

        # 构造双通道超声帧 (2, 4, 700)
        cur_frame = np.stack([
            self.us_env[ch][i, US_S0:US_S1] for ch in sorted(self.us_env)
        ], axis=0)   # (4, 700)
        ref_frame = np.stack([
            self.us_ref[ch] for ch in sorted(self.us_ref)
        ], axis=0)   # (4, 700)
        dual = np.stack([cur_frame, ref_frame], axis=0)  # (2, 4, 700)

        self.kin_buf.append(self.kin_n[i])
        self.us_buf.append(dual)

        pred_val = np.nan
        if len(self.kin_buf) == self.W:
            k = torch.from_numpy(np.stack(self.kin_buf)).unsqueeze(0).to(self.device)
            u = torch.from_numpy(np.stack(self.us_buf)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_val = self.model(k, u).item() * self.lbl_std + self.lbl_mean

        self.truth_buf = np.roll(self.truth_buf, -1); self.truth_buf[-1] = self.lbl[i]
        self.pred_buf  = np.roll(self.pred_buf,  -1); self.pred_buf[-1]  = pred_val
        self.acc_buf   = np.roll(self.acc_buf,   -1); self.acc_buf[-1]   = self.kin[i, 2]
        self.us_disp   = np.roll(self.us_disp,   -1, axis=0)
        self.us_disp[-1] = self.us_env[self.channel][i, US_S0:US_S1]

        img_data = (self.us_disp * 255).astype(np.uint8)
        self.img.setImage(img_data, autoLevels=False, levels=(0, 255))
        self.img.setRect(pg.QtCore.QRectF(0, US_S0, self.HISTORY, US_S1 - US_S0))

        x = np.arange(self.HISTORY)
        self.c_truth.setData(x, self._map(self.truth_buf,
                                           self.truth_buf.min(), self.truth_buf.max()))
        valid = ~np.isnan(self.pred_buf)
        if valid.any():
            self.c_pred.setData(x[valid], self._map(self.pred_buf[valid],
                                                     self.truth_buf.min(), self.truth_buf.max()))
        self.c_acc.setData(x, self._map(self.acc_buf,
                                         self.acc_buf.min(), self.acc_buf.max()))

        if valid.any():
            err = self.pred_buf[valid] - self.truth_buf[valid]
            self.stat_lbl.setText(
                f"帧 {i+1}/{self.N}  MAE={np.mean(np.abs(err)):.3f}  "
                f"RMSE={np.sqrt(np.mean(err**2)):.3f} Nm")
        self.cur += 1


# ── 交互式选择 ────────────────────────────────────────────────────────────────

def pick_segment() -> Path:
    candidates = [
        Path(__file__).parent.parent.parent / "超声采集数据",
        Path(__file__).parent.parent.parent.parent / "超声采集数据",
    ]
    data_root = next((c for c in candidates if c.exists()), None)
    if data_root is None:
        print("[ERROR] 找不到超声采集数据目录"); sys.exit(1)
    segs = sorted(data_root.rglob("handle_data_*"))
    if not segs:
        print("[ERROR] 没有找到 handle_data_* 目录"); sys.exit(1)
    print(f"\n找到 {len(segs)} 个数据段：")
    for i, s in enumerate(segs):
        print(f"  [{i:2d}] {s.relative_to(data_root)}")
    while True:
        try:
            idx = int(input(f"\n请输入编号 (0~{len(segs)-1}): ").strip())
            if 0 <= idx < len(segs):
                return segs[idx]
        except (ValueError, EOFError):
            pass
        print("输入无效")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg",     default=None)
    p.add_argument("--ckpt",    default="checkpoints/best.pt")
    p.add_argument("--fps",     type=float, default=30.0)
    p.add_argument("--channel", type=int,   default=1)
    args = p.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent / ckpt_path
    if not ckpt_path.exists():
        print(f"[ERROR] 权重文件不存在: {ckpt_path}"); sys.exit(1)

    seg_dir = Path(args.seg) if args.seg else pick_segment()

    print("加载模型...")
    model, kin_mean, kin_std, lbl_mean, lbl_std, W, ref_frames = \
        load_checkpoint(ckpt_path, device)

    print("加载运动学 + 标签...")
    kin, _, lbl = _load_segment(seg_dir)
    kin_n = (kin - kin_mean) / kin_std

    print("加载并预处理超声（4通道）...")
    torque_df = pd.read_csv(seg_dir / "label" / "torque.csv")
    ref_ts    = torque_df["timestamp"].values.astype(np.float64)
    us_env, us_ref = load_us_all(seg_dir, ref_ts, ref_frames)

    print(f"总帧数: {len(lbl)}  窗口: {W}  fps: {args.fps}")

    app = QApplication(sys.argv)
    win = InferViewer(seg_dir, model, kin, kin_n, us_env, us_ref,
                      lbl, lbl_mean, lbl_std, W, device, args.fps, args.channel)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
