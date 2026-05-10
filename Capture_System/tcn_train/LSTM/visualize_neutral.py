"""
visualize_neutral.py — 中性参考（Neutral Referencing）效果可视化

直接运行，交互式选择数据段：
    python visualize_neutral.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter, median_filter

# 中文字体：找系统里第一个可用的中文字体
def _setup_font():
    prefer = ["Microsoft YaHei", "SimHei", "SimSun", "FangSong",
              "STHeiti", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in prefer:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_font()

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from data_loader import _nearest_align

S0, S1 = 150, 850


# ── 预处理 ────────────────────────────────────────────────────────────────────

def hilbert_envelope(raw: np.ndarray) -> np.ndarray:
    rf = raw.astype(np.float32)
    rf -= rf.mean(axis=1, keepdims=True)
    env = np.abs(hilbert(rf, axis=1))
    log = 20.0 * np.log10(np.clip(env, 1e-6, None))
    out = np.zeros_like(log)
    gain = np.arange(log.shape[1]) * 0.025
    for i in range(len(log)):
        f = log[i] + gain
        vmin, vmax = np.percentile(f, 25), np.percentile(f, 99)
        out[i] = np.clip((f - vmin) / (vmax - vmin + 1e-5), 0, 1)
    out = median_filter(out, size=(3, 3))
    out = gaussian_filter(out, sigma=[0.5, 1.2])
    return out.astype(np.float32)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_seg(seg_dir: Path):
    torque_df = pd.read_csv(seg_dir / "label" / "torque.csv")
    ref_ts    = torque_df["timestamp"].values.astype(np.float64)
    label     = torque_df["hip_torque_Nm"].values.astype(np.float32)

    us_df = pd.read_csv(seg_dir / "input" / "ultrasound.csv")
    dcols = sorted([c for c in us_df.columns if c.startswith("d") and c[1:].isdigit()],
                   key=lambda x: int(x[1:]))
    us_raw = {}
    for ch in sorted(us_df["channel"].unique()):
        sub   = us_df[us_df["channel"] == ch].reset_index(drop=True)
        ch_ts = sub["timestamp"].values.astype(np.float64)
        idx   = _nearest_align(ch_ts, ref_ts)
        us_raw[int(ch)] = sub[dcols].values[idx].astype(np.float32)

    return us_raw, label


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_neutral(seg_dir: Path, channel: int, ref_frames: int = 30, max_frames: int = 500):
    print(f"加载数据: {seg_dir.name}")
    us_raw, label = load_seg(seg_dir)

    raw_full = us_raw[channel][:max_frames, S0:S1]
    print(f"帧数: {len(raw_full)}")

    print("计算 Hilbert 包络...")
    env  = hilbert_envelope(raw_full)
    ref  = env[:ref_frames].mean(axis=0)
    diff = np.clip(env - ref[np.newaxis, :], -1.0, 1.0)

    N      = len(env)
    frames = np.arange(N)
    lbl    = label[:N]

    all_ch_diff_mean = {}
    for ch in sorted(us_raw.keys()):
        e = hilbert_envelope(us_raw[ch][:max_frames, S0:S1])
        r = e[:ref_frames].mean(axis=0)
        d = np.clip(e - r[np.newaxis, :], -1.0, 1.0)
        all_ch_diff_mean[ch] = d.mean(axis=1)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Neutral Referencing — {seg_dir.name}  (Ch{channel})",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    extent = [0, N, S1, S0]

    ax1 = fig.add_subplot(gs[0, 0])
    raw_norm = (raw_full - raw_full.min()) / (raw_full.max() - raw_full.min() + 1e-6)
    ax1.imshow(raw_norm.T, aspect="auto", cmap="gray", extent=extent)
    ax1.set_title("原始 RF 信号"); ax1.set_xlabel("帧"); ax1.set_ylabel("深度")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(env.T, aspect="auto", cmap="gray", extent=extent, vmin=0, vmax=1)
    ax2.set_title("Hilbert 包络"); ax2.set_xlabel("帧"); ax2.set_ylabel("深度")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(np.arange(S0, S1), ref, linewidth=1.2, color="steelblue")
    ax3.set_title(f"中性参考帧（前 {ref_frames} 帧均值）")
    ax3.set_xlabel("深度"); ax3.set_ylabel("包络强度"); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    im = ax4.imshow(diff.T, aspect="auto", cmap="RdBu_r",
                    extent=extent, vmin=-0.5, vmax=0.5)
    ax4.set_title("差分信号（包络 - 参考）")
    ax4.set_xlabel("帧"); ax4.set_ylabel("深度")
    plt.colorbar(im, ax=ax4, fraction=0.03)

    ax5 = fig.add_subplot(gs[2, :])
    colors = ["#E53935", "#43A047", "#1E88E5", "#FB8C00"]
    for i, (ch, dmean) in enumerate(all_ch_diff_mean.items()):
        ax5.plot(frames, dmean, color=colors[i % 4], linewidth=1.0,
                 label=f"Ch{ch} diff mean", alpha=0.8)
    ax5b = ax5.twinx()
    ax5b.plot(frames, lbl, color="black", linewidth=1.5, linestyle="--",
              label="Hip Torque (Nm)", alpha=0.9)
    ax5b.set_ylabel("力矩 (Nm)")
    ax5.set_title("各通道差分均值 vs 真实力矩")
    ax5.set_xlabel("帧"); ax5.set_ylabel("差分均值"); ax5.grid(True, alpha=0.3)
    lines = ax5.get_legend_handles_labels()[0] + ax5b.get_legend_handles_labels()[0]
    lbls  = ax5.get_legend_handles_labels()[1] + ax5b.get_legend_handles_labels()[1]
    ax5.legend(lines, lbls, loc="upper right", fontsize=9)

    out_path = Path(__file__).parent / "neutral_ref_vis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {out_path}")
    plt.show()


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
        print(f"[ERROR] {data_root} 下没有 handle_data_* 目录"); sys.exit(1)

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
        print("输入无效，请重试")


def pick_channel() -> int:
    while True:
        try:
            ch = int(input("选择显示通道 (1~4) [默认 1]: ").strip() or "1")
            if 1 <= ch <= 4:
                return ch
        except (ValueError, EOFError):
            pass
        print("请输入 1~4")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seg_dir = pick_segment()
    channel = pick_channel()
    print(f"\n选择段: {seg_dir}")
    plot_neutral(seg_dir, channel)
