import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt

# 配置中文字体，以免绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 预测配置 =================
PREDICT_CONFIG = {
    'target_file': r"D:\nightmare\Documents\SRTP\AllData\Cleaned_Dataset_v1\Labeled_Data_12_aligned_seg1.xlsx",
    # 【需要修改】待预测的文件路径
    'model_path': r"training_logs\ultr_fusion_model_kfold_attention_added_v3_1\best_model_overall.keras",
    'scaler_path': r"training_logs\ultr_fusion_model_kfold_attention_added_v3_1\best_scaler_overall.pkl",
    'pca_path': r"training_logs\ultr_fusion_model_kfold_attention_added_v3_1\best_pca_overall.pkl",

    'sequence_length': 25,  # 必须与训练时一致
    'smoothing_strategy': 'lowpass',  # 可选: 'basic', 'monotonic', 'lowpass'
    'ema_alpha': 0.6,
    'butter_order': 3,
    'butter_cutoff': 0.1
}


# ================= 2. 依赖函数与类 =================

# 必须重新定义自定义评估指标，否则加载模型会报错
@tf.keras.utils.register_keras_serializable()
def phase_mae(y_true, y_pred):
    true_phase = tf.math.atan2(y_true[:, 0], y_true[:, 1])
    pred_phase = tf.math.atan2(y_pred[:, 0], y_pred[:, 1])
    true_phase = tf.math.floormod(true_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    pred_phase = tf.math.floormod(pred_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    diff = tf.math.abs(true_phase - pred_phase)
    circ_diff = tf.math.minimum(diff, 1.0 - diff)
    return tf.math.reduce_mean(circ_diff)


class EnhancedVectorSmoother:
    def __init__(self, alpha=PREDICT_CONFIG['ema_alpha']):
        self.alpha = alpha
        self.smooth_sin = 0.0
        self.smooth_cos = 1.0
        self.initialized = False

    def update(self, raw_sin, raw_cos):
        if not self.initialized:
            self.smooth_sin = raw_sin
            self.smooth_cos = raw_cos
            self.initialized = True
        else:
            self.smooth_sin = self.alpha * raw_sin + (1 - self.alpha) * self.smooth_sin
            self.smooth_cos = self.alpha * raw_cos + (1 - self.alpha) * self.smooth_cos

        phase_rad = np.arctan2(self.smooth_sin, self.smooth_cos)
        phase_linear = (phase_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        return phase_linear


def create_test_sequences(features, time_steps):
    Xs = []
    for i in range(len(features) - time_steps):
        Xs.append(features[i: i + time_steps])
    return np.array(Xs)


# ================= 3. 核心预测逻辑 =================

def predict_file():
    print("1. 加载模型与预处理组件...")
    # 加载带有自定义对象的模型
    model = load_model(PREDICT_CONFIG['model_path'], custom_objects={'phase_mae': phase_mae})
    scaler = joblib.load(PREDICT_CONFIG['scaler_path'])
    pca = joblib.load(PREDICT_CONFIG['pca_path'])

    print(f"2. 读取目标文件: {PREDICT_CONFIG['target_file']}")
    file_path = PREDICT_CONFIG['target_file']

    raw_features = None
    label_df = None  # 用于存储带有 Label 的数据表以供画图对比

    # 兼容 CSV 和 Excel，并加入多维度特征合并逻辑
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            feat_cols = [c for c in df.columns if str(c).startswith('Feat_')]
            if not feat_cols:
                raise ValueError("未找到以 'Feat_' 开头的特征列。")
            raw_features = df[feat_cols].values
            label_df = df

        elif file_path.endswith('.xlsx'):
            # 读取所有 sheet
            sheet_dict = pd.read_excel(file_path, sheet_name=None)
            sheet_features = []

            for sheet_name, df in sheet_dict.items():
                df.columns = df.columns.str.strip()
                feat_cols = [c for c in df.columns if str(c).startswith('Feat_')]
                if not feat_cols:
                    continue

                sheet_features.append(df[feat_cols].values)

                # 保留第一个含有 Label 的 df，用于画真实标签对比图
                if label_df is None and 'Label_Sin' in df.columns:
                    label_df = df

            if not sheet_features:
                raise ValueError("所有 sheet 中均未找到以 'Feat_' 开头的特征列。")

            # 拼接所有 sheet 的特征，沿列方向合并
            raw_features = np.concatenate(sheet_features, axis=1)

        else:
            raise ValueError("不支持的文件格式，仅支持 .csv 或 .xlsx")

    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print(f"成功提取特征，拼接后特征维度为: {raw_features.shape}")

    print("3. 数据预处理 (Scaler -> PCA -> 序列化)...")
    f_scaled = scaler.transform(raw_features)
    f_pca = pca.transform(f_scaled)
    X_test = create_test_sequences(f_pca, PREDICT_CONFIG['sequence_length'])

    print(f"4. 模型推理中 (序列数量: {len(X_test)})...")
    preds_raw = model.predict(X_test, verbose=1)

    print(f"5. 后处理与相位平滑 (策略: {PREDICT_CONFIG['smoothing_strategy']})...")
    smoother = EnhancedVectorSmoother(alpha=PREDICT_CONFIG['ema_alpha'])
    base_phases = [smoother.update(s, c) for s, c in preds_raw]

    pred_phases = []
    strategy = PREDICT_CONFIG['smoothing_strategy']

    if strategy == 'monotonic':
        for i, p in enumerate(base_phases):
            if i == 0:
                pred_phases.append(p)
                continue
            prev_p = pred_phases[-1]
            if prev_p > 0.8 and p < 0.2:
                pred_phases.append(p)
            elif p < prev_p:
                pred_phases.append(prev_p)
            else:
                pred_phases.append(p)
    elif strategy == 'lowpass':
        unwrapped_phase = np.unwrap(np.array(base_phases) * 2 * np.pi)
        b, a = butter(N=PREDICT_CONFIG['butter_order'], Wn=PREDICT_CONFIG['butter_cutoff'], btype='low')
        smoothed_unwrapped = filtfilt(b, a, unwrapped_phase)
        pred_phases = (smoothed_unwrapped / (2 * np.pi)) % 1.0
        pred_phases = pred_phases.tolist()
    else:
        pred_phases = base_phases

    # ================= 新增：将预测结果保存为 CSV =================
    print("6. 正在将预测结果导出为 CSV 文件...")
    # 填充前 sequence_length 个时间步的空缺，确保和原文件行数对齐
    pad_length = PREDICT_CONFIG['sequence_length']
    padded_predictions = [np.nan] * pad_length + pred_phases

    # 构建导出数据框
    output_df = pd.DataFrame({
        'Time_Step': range(len(padded_predictions)),
        'Predicted_Phase': padded_predictions
    })

    # 生成保存路径（保存在原始测试数据同一目录下，加了 _predicted 后缀）
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    output_csv_path = os.path.join(output_dir, f"{base_name}_predicted.csv")

    output_df.to_csv(output_csv_path, index=False)
    print(f"预测结果已成功保存至: {output_csv_path}")
    # ===============================================================

    print("7. 预测完成，正在绘制结果...")
    plt.figure(figsize=(15, 6))
    plt.plot(pred_phases, 'r-', linewidth=2, label=f'Predicted Phase ({strategy})')

    # 从保留的 label_df 中画出真实标签
    if label_df is not None and 'Label_Sin' in label_df.columns and 'Label_Cos' in label_df.columns:
        sin_vals = label_df['Label_Sin'].values[PREDICT_CONFIG['sequence_length']:]
        cos_vals = label_df['Label_Cos'].values[PREDICT_CONFIG['sequence_length']:]
        true_phases_rad = np.arctan2(sin_vals, cos_vals)
        true_phases = (true_phases_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        plt.plot(true_phases, 'k-', alpha=0.5, linewidth=2, label='Ground Truth Phase')

    plt.title(f"Phase Prediction for {os.path.basename(file_path)}")
    plt.xlabel('Time Steps')
    plt.ylabel('Phase (0~1)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict_file()