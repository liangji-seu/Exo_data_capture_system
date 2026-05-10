import os
import glob
import random
import numpy as np
import pandas as pd
import joblib
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, Multiply
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import resample, hilbert, butter, filtfilt
from sklearn.model_selection import KFold

# 配置中文字体，以免绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 全局配置与超参数控制台 =================
CONFIG = {
    # ---------------- 路径设置 ----------------
    'data_dir': r"D:\nightmare\Documents\SRTP\AllData\Cleaned_Dataset_v3",
    'save_dir': r"training_logs\ultr_fusion_model_kfold_attention_added_v3_1",

    # ---------------- 数据处理与增强 ----------------
    'pca_components': 25,  # 容纳双肌肉的协同特征
    'sequence_length': 25,  # LSTM的时间步长 (时间序列窗口长度)
    'augment_factors': [0.8, 0.9, 1.0, 1.1, 1.2],  # Time-warp 数据增强的拉伸比例

    # ---------------- 网络结构超参数 (已瘦身 + 强正则化) ----------------
    'lstm_1_units': 96,  #第一层节点数
    'lstm_2_units': 48,  # 第二层节点数
    'dense_units': 32,   #连接层节点数
    'dropout_rate': 0.3,  # dropout
    'l2_reg': 0.001,  # 保持强L2正则化

    # ---------------- 训练过程超参数 ----------------
    'k_folds': 5,
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.005,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 6,
    'early_stop_patience': 20,

    # ---------------- 后处理与平滑策略 ----------------
    'smoothing_strategy': 'lowpass', # 可选值: 'basic', 'monotonic', 'lowpass'
    'ema_alpha': 0.6,
    'butter_order': 3,
    'butter_cutoff': 0.1
}

if not os.path.exists(CONFIG['save_dir']):
    os.makedirs(CONFIG['save_dir'])


# ================= 2. 核心工具类与函数 =================

class EnhancedVectorSmoother:
    def __init__(self, alpha=CONFIG['ema_alpha']):
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


def augment_time_warp(features, labels, factor):
    if factor == 1.0: return features, labels
    new_len = int(len(features) * factor)
    aug_feat = resample(features, new_len, axis=0)
    aug_label = resample(labels, new_len, axis=0)
    return aug_feat, aug_label


def robust_cos_reconstruction(sin_values):
    sin_centered = sin_values - np.mean(sin_values)
    analytic_signal = hilbert(sin_centered)
    instantaneous_phase = np.angle(analytic_signal)
    return np.cos(instantaneous_phase)


def load_all_data_without_split(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.xlsx"))
    valid_trials = []

    for f in all_files:
        fname = os.path.basename(f)
        if fname.startswith('~$'): continue

        try:
            if f.endswith('.xlsx'):
                sheet_dict = pd.read_excel(f, sheet_name=None)
                sheet_features = []
                labels = None

                for sheet_name, df in sheet_dict.items():
                    df.columns = df.columns.str.strip()
                    feat_cols = [c for c in df.columns if str(c).startswith('Feat_')]
                    if not feat_cols: continue

                    sheet_features.append(df[feat_cols].values)

                    if labels is None and 'Label_Sin' in df.columns:
                        sin_vals = df['Label_Sin'].values
                        cos_vals = df['Label_Cos'].values if 'Label_Cos' in df.columns else robust_cos_reconstruction(
                            sin_vals)
                        labels = np.column_stack((sin_vals, cos_vals))

                if sheet_features and labels is not None:
                    combined_features = np.concatenate(sheet_features, axis=1)
                    valid_trials.append((fname, combined_features, labels))
                    print(f"[合并成功] 读取多sheet Excel: {fname}, 融合后特征维度: {combined_features.shape}")

            elif f.endswith('.csv'):
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                feat_cols = [c for c in df.columns if str(c).startswith('Feat_')]
                if not feat_cols or 'Label_Sin' not in df.columns: continue

                sin_vals = df['Label_Sin'].values
                cos_vals = df['Label_Cos'].values if 'Label_Cos' in df.columns else robust_cos_reconstruction(sin_vals)
                features = df[feat_cols].values
                labels = np.column_stack((sin_vals, cos_vals))
                valid_trials.append((fname, features, labels))
                print(f"[读取成功] 读取单表 CSV: {fname}, 特征维度: {features.shape}")

        except Exception as e:
            print(f"处理文件 {fname} 时出错: {e}")
            continue

    return valid_trials


def create_sequences(features, labels, time_steps):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i: i + time_steps])
        ys.append(labels[i + time_steps])
    return np.array(Xs), np.array(ys)


# ================= 3. 模型构建 (加入注意力机制) =================

def phase_mae(y_true, y_pred):
    true_phase = tf.math.atan2(y_true[:, 0], y_true[:, 1])
    pred_phase = tf.math.atan2(y_pred[:, 0], y_pred[:, 1])
    true_phase = tf.math.floormod(true_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    pred_phase = tf.math.floormod(pred_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    diff = tf.math.abs(true_phase - pred_phase)
    circ_diff = tf.math.minimum(diff, 1.0 - diff)
    return tf.math.reduce_mean(circ_diff)


def build_fusion_model(input_shape):
    inputs = Input(shape=input_shape)

    # 【新增核心】特征注意力层 (Feature Attention)
    # 自动学习各个维度的特征在当前时间步的重要性，打分区间 0~1
    attention_probs = Dense(input_shape[1], activation='sigmoid', name='feature_attention_weights')(inputs)
    # 将打分权重乘以原始输入特征，实现对劣质数据的动态屏蔽
    x = Multiply(name='attention_multiply')([inputs, attention_probs])

    # LSTM 主干网络 (已调小参数防止死记硬背)
    x = Bidirectional(LSTM(CONFIG['lstm_1_units'], return_sequences=True,
                           kernel_regularizer=l2(CONFIG['l2_reg']),
                           recurrent_regularizer=l2(CONFIG['l2_reg'])))(x)  # 注意这里输入变成了 x 而不是 inputs
    x = BatchNormalization()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    x = Bidirectional(LSTM(CONFIG['lstm_2_units'], return_sequences=False,
                           kernel_regularizer=l2(CONFIG['l2_reg'])))(x)
    x = BatchNormalization()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    x = Dense(CONFIG['dense_units'], activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    outputs = Dense(2, activation='tanh')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), loss='mse', metrics=['mae', phase_mae])
    return model


# ================= 4. 主程序管线 =================

def main():
    print("1. 加载所有原始数据...")
    all_trials = load_all_data_without_split(CONFIG['data_dir'])
    if not all_trials: raise ValueError("未找到数据，请检查 data_dir！")

    k_folds = CONFIG['k_folds']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_trials)):
        print(f"\n================ 开始训练第 {fold + 1}/{k_folds} 折 ================")
        train_trials = [all_trials[i] for i in train_idx]
        val_trials = [all_trials[i] for i in val_idx]

        aug_train_features, aug_train_labels = [], []
        for _, feat, label in train_trials:
            for factor in CONFIG['augment_factors']:
                f_aug, l_aug = augment_time_warp(feat, label, factor)
                aug_train_features.append(f_aug)
                aug_train_labels.append(l_aug)

        all_train_feat_concat = np.vstack(aug_train_features)
        scaler = StandardScaler()
        all_train_feat_scaled = scaler.fit_transform(all_train_feat_concat)
        pca = PCA(n_components=CONFIG['pca_components'])
        pca.fit(all_train_feat_scaled)
        joblib.dump(scaler, os.path.join(CONFIG['save_dir'], f'scaler_fold_{fold + 1}.pkl'))
        joblib.dump(pca, os.path.join(CONFIG['save_dir'], f'pca_fold_{fold + 1}.pkl'))
        X_train_seqs, y_train_seqs = [], []
        for feat, label in zip(aug_train_features, aug_train_labels):
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = create_sequences(f_pca, label, CONFIG['sequence_length'])
            X_train_seqs.append(X_seq)
            y_train_seqs.append(y_seq)
        X_train, y_train = np.vstack(X_train_seqs), np.vstack(y_train_seqs)

        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        X_val_seqs, y_val_seqs = [], []
        for _, feat, label in val_trials:
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = create_sequences(f_pca, label, CONFIG['sequence_length'])
            X_val_seqs.append(X_seq)
            y_val_seqs.append(y_seq)
        X_val, y_val = np.vstack(X_val_seqs), np.vstack(y_val_seqs)

        model = build_fusion_model(input_shape=(CONFIG['sequence_length'], CONFIG['pca_components']))
        callbacks = [
            ModelCheckpoint(os.path.join(CONFIG['save_dir'], f"best_model_fold_{fold + 1}.keras"),
                            monitor='val_phase_mae', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_phase_mae', factor=CONFIG['reduce_lr_factor'],
                              patience=CONFIG['reduce_lr_patience'], verbose=0, mode='min'),
            EarlyStopping(monitor='val_phase_mae', patience=CONFIG['early_stop_patience'],
                          restore_best_weights=True, verbose=0, mode='min')
        ]

        # 设置为 verbose=2 以便清晰地每轮输出一行
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=CONFIG['epochs'],
                  batch_size=CONFIG['batch_size'], callbacks=callbacks, verbose=2)

        print(f"\n正在评估第 {fold + 1} 折的所有验证文件 (策略: {CONFIG['smoothing_strategy']})...")
        fold_trial_maes = []

        for trial_name, test_feat, test_label in val_trials:
            f_scaled = scaler.transform(test_feat)
            f_pca = pca.transform(f_scaled)
            X_test_trial, y_test_trial = create_sequences(f_pca, test_label, CONFIG['sequence_length'])
            preds_raw = model.predict(X_test_trial, verbose=0)

            smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
            base_phases = [smoother.update(s, c) for s, c in preds_raw]

            pred_phases = []
            strategy = CONFIG['smoothing_strategy']

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
                b, a = butter(N=CONFIG['butter_order'], Wn=CONFIG['butter_cutoff'], btype='low')
                smoothed_unwrapped = filtfilt(b, a, unwrapped_phase)
                pred_phases = (smoothed_unwrapped / (2 * np.pi)) % 1.0
                pred_phases = pred_phases.tolist()

            else:
                pred_phases = base_phases

            true_phases_rad = np.arctan2(y_test_trial[:, 0], y_test_trial[:, 1])
            true_phases = (true_phases_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
            diff = np.abs(np.array(pred_phases) - true_phases)
            circ_diff = np.minimum(diff, 1.0 - diff)
            trial_mae = np.mean(circ_diff)
            fold_trial_maes.append(trial_mae)

        current_fold_mae = np.mean(fold_trial_maes)
        all_fold_maes.append(current_fold_mae)
        print(f"第 {fold + 1} 折验证集平均 MAE: {current_fold_mae * 100:.2f}%\n")

        plt.figure(figsize=(15, 6))
        limit = min(500, len(true_phases))
        plt.plot(true_phases[:limit], 'k-', alpha=0.5, linewidth=3, label='Ground Truth Phase')
        plt.plot(pred_phases[:limit], 'r--', linewidth=2, label=f'Predicted Phase ({CONFIG["smoothing_strategy"]})')
        plt.title(f"Fold {fold + 1} Phase Prediction - MAE: {current_fold_mae * 100:.2f}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['save_dir'], f"fold_{fold + 1}_result.png"))
        plt.close()

        # 【新增修复】在每一折训练结束后清理 Keras 会话，释放内存，防止后续爆出 retracing 警告
        tf.keras.backend.clear_session()

        # ================= 新增：自动挑选并保存全局最佳模型 =================
        best_fold_idx = np.argmin(all_fold_maes) + 1  # 找到 MAE 最小的那一折的索引
        best_mae = np.min(all_fold_maes)
        print(f"\n================================================================")
        print(f"自动筛选完成：表现最好的是第 {best_fold_idx} 折 (MAE: {best_mae * 100:.2f}%)")
        print(f"正在将其保存为最终部署模型...")

        # 将表现最好那一折的模型、Scaler、PCA 复制并重命名为 overall_best
        shutil.copy(os.path.join(CONFIG['save_dir'], f"best_model_fold_{best_fold_idx}.keras"),
                    os.path.join(CONFIG['save_dir'], "best_model_overall.keras"))
        shutil.copy(os.path.join(CONFIG['save_dir'], f"scaler_fold_{best_fold_idx}.pkl"),
                    os.path.join(CONFIG['save_dir'], "best_scaler_overall.pkl"))
        shutil.copy(os.path.join(CONFIG['save_dir'], f"pca_fold_{best_fold_idx}.pkl"),
                    os.path.join(CONFIG['save_dir'], "best_pca_overall.pkl"))
        # =====================================================================
    print("================================================================")
    print(f"K折交叉验证全部完成！")
    print(f"最终评估模型稳定泛化性能的平均 MAE: {np.mean(all_fold_maes) * 100:.2f}%")
    print("================================================================")


if __name__ == "__main__":
    main()