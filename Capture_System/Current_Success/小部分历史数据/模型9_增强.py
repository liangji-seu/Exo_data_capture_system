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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, Multiply, \
    UnitNormalization, GlobalAveragePooling1D, Reshape
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
    'data_dir': r"D:\nightmare\Documents\SRTP\AllData\Processed_Final_Training_Set_clean_v2",
    'save_dir': r"D:\nightmare\Documents\SRTP\LSTM\training_logs\ultr_fusion_model_9",

    # ---------------- 数据处理与增强 ----------------
    'pca_components': 25,
    'sequence_length': 40,  # [关键修正] 从 25 提升到 40，扩大模型视野以提前预测跳变
    'augment_factors': [0.8, 0.9, 1.0, 1.1, 1.2],

    # ---------------- 网络结构超参数 ----------------
    'lstm_1_units': 96,
    'lstm_2_units': 48,
    'dense_units': 32,
    'dropout_rate': 0.3,
    'l2_reg': 0.001,

    # ---------------- 训练过程超参数 ----------------
    'k_folds': 5,
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.005,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 6,
    'early_stop_patience': 30,

    # ---------------- 边界样本加权 (起步/收尾强化) ----------------
    'edge_weight_enable': True,
    'edge_ratio': 0.15,  # 相位 <0.15 或 >0.85 认为是边界
    'edge_weight': 3.0,  # [关键修正] 权重提升到 3.0，重罚预测滞后

    # ---------------- 后处理与平滑策略 ----------------
    'smoothing_strategy': 'monotonic',  # [关键修正] 默认用 raw 测试，抛弃平滑器带来的强行滞后
    'tracker_alpha': 0.85,  # 若必须用 tracker，极大降低历史惯性
    'tracker_beta': 0.05,
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


class RobustPhaseTracker:
    def __init__(self, alpha=CONFIG.get('tracker_alpha', 0.85), beta=CONFIG.get('tracker_beta', 0.05)):
        self.alpha = alpha
        self.beta = beta
        self.estimated_phase = None
        self.estimated_velocity = 0.04
        self.initialized = False

    def update(self, raw_sin, raw_cos):
        obs_phase_rad = np.arctan2(raw_sin, raw_cos)
        obs_phase = (obs_phase_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

        if not self.initialized:
            self.estimated_phase = obs_phase
            self.initialized = True
            return self.estimated_phase

        pred_phase = (self.estimated_phase + self.estimated_velocity) % 1.0

        error = obs_phase - pred_phase
        if error > 0.5:
            error -= 1.0
        elif error < -0.5:
            error += 1.0

        # 放宽截断范围，允许突变（从 0.15 放宽到 0.4）
        effective_error = np.clip(error, -0.4, 0.4)

        self.estimated_phase = (pred_phase + self.alpha * effective_error) % 1.0
        self.estimated_velocity = self.estimated_velocity + self.beta * effective_error
        self.estimated_velocity = np.clip(self.estimated_velocity, 0.015, 0.1)

        return self.estimated_phase


class SmartGatedPhaseTracker:
    def __init__(self, base_velocity=0.035):  # 假设平均步频大概一帧走 3.5%
        self.estimated_phase = None
        self.velocity = base_velocity
        self.initialized = False

    def update(self, raw_sin, raw_cos):
        # 1. 获取模型原始预测相位
        model_phase_rad = np.arctan2(raw_sin, raw_cos)
        model_phase = (model_phase_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

        if not self.initialized:
            self.estimated_phase = model_phase
            self.initialized = True
            return self.estimated_phase

        # 2. 惯性推演：按照当前步频往前走一步
        pred_phase = (self.estimated_phase + self.velocity) % 1.0

        # 3. 动态置信度门控 (核心灵魂)
        distance_to_edge = min(model_phase, 1.0 - model_phase)

        if distance_to_edge < 0.15:
            # 在边界区域(0或1附近)，极度信任模型，用于纠偏和锚定
            trust_in_model = 0.85
        else:
            # 在中间下垂区域，基本不相信模型，纯靠内部惯性匀速滑行跨越下垂区！
            trust_in_model = 0.05

            # 4. 计算环形误差
        error = model_phase - pred_phase
        if error > 0.5:
            error -= 1.0
        elif error < -0.5:
            error += 1.0

        # 5. 只有在高度信任模型时（处于边界），才更新对用户步频的认知
        if trust_in_model > 0.5:
            self.velocity = np.clip(self.velocity + 0.05 * error, 0.015, 0.08)

        # 6. 最终状态更新
        self.estimated_phase = (pred_phase + trust_in_model * error) % 1.0
        return self.estimated_phase

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


# ================= [重大修复] 正确的边缘加权逻辑 =================
def create_boundary_weights(y_seq, edge_ratio=0.15, edge_weight=2.0):
    """根据真实的相位值给跳变边缘样本更高权重，解决削顶现象。"""
    if len(y_seq) == 0:
        return np.array([], dtype=np.float32)

    true_rad = np.arctan2(y_seq[:, 0], y_seq[:, 1])
    true_phases = (true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

    weights = np.ones(len(y_seq), dtype=np.float32)
    # 当相位靠近 0 或靠近 1 时，赋予高权重
    edge_mask = (true_phases < edge_ratio) | (true_phases > (1.0 - edge_ratio))
    weights[edge_mask] = edge_weight
    return weights


# ================= 3. 模型构建 (加入注意力机制) =================

@tf.keras.utils.register_keras_serializable()
def phase_mae(y_true, y_pred):
    true_phase = tf.math.atan2(y_true[:, 0], y_true[:, 1])
    pred_phase = tf.math.atan2(y_pred[:, 0], y_pred[:, 1])
    true_phase = tf.math.floormod(true_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    pred_phase = tf.math.floormod(pred_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    diff = tf.math.abs(true_phase - pred_phase)
    circ_diff = tf.math.minimum(diff, 1.0 - diff)
    return tf.math.reduce_mean(circ_diff)


@tf.keras.utils.register_keras_serializable()
def cosine_loss(y_true, y_pred):
    # 确保证输出和标签都被 L2 归一化为单位圆上的点
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)

    # 计算点乘 (即余弦相似度)
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)

    # 只有最纯净的余弦距离，绝不能加刚才那个错误的导数项！
    return tf.reduce_mean(1.0 - dot_product)


def build_fusion_model(input_shape):
    inputs = Input(shape=input_shape)

    global_avg = GlobalAveragePooling1D()(inputs)
    att_bottleneck = Dense(8, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(global_avg)
    attention_probs = Dense(input_shape[1], activation='sigmoid', name='feature_attention_weights')(att_bottleneck)
    attention_probs_reshaped = Reshape((1, input_shape[1]))(attention_probs)
    x = Multiply(name='attention_multiply')([inputs, attention_probs_reshaped])

    x = Bidirectional(LSTM(CONFIG['lstm_1_units'], return_sequences=True,
                           kernel_regularizer=l2(CONFIG['l2_reg']),
                           recurrent_regularizer=l2(CONFIG['l2_reg'])))(x)
    x = BatchNormalization()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    x = Bidirectional(LSTM(CONFIG['lstm_2_units'], return_sequences=False,
                           kernel_regularizer=l2(CONFIG['l2_reg'])))(x)
    x = BatchNormalization()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    x = Dense(CONFIG['dense_units'], activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)

    # 这里的归一化写得非常好，完美保证了预测向量在单位圆上
    raw_outputs = Dense(2, activation='linear')(x)
    outputs = UnitNormalization(axis=-1, name='l2_norm_output')(raw_outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), loss=cosine_loss, metrics=['mae', phase_mae])
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
        X_train_seqs, y_train_seqs, train_weights = [], [], []

        for feat, label in zip(aug_train_features, aug_train_labels):
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = create_sequences(f_pca, label, CONFIG['sequence_length'])
            if len(X_seq) == 0: continue

            X_train_seqs.append(X_seq)
            y_train_seqs.append(y_seq)

            # [关键修正处调用] 直接传入 y_seq 来精准加权
            if CONFIG['edge_weight_enable']:
                w_seq = create_boundary_weights(
                    y_seq,
                    edge_ratio=CONFIG['edge_ratio'],
                    edge_weight=CONFIG['edge_weight']
                )
            else:
                w_seq = np.ones(len(X_seq), dtype=np.float32)
            train_weights.append(w_seq)

        X_train, y_train = np.vstack(X_train_seqs), np.vstack(y_train_seqs)
        sample_weights = np.concatenate(train_weights).astype(np.float32)

        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        sample_weights = sample_weights[idx]

        X_val_seqs, y_val_seqs = [], []
        for _, feat, label in val_trials:
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = create_sequences(f_pca, label, CONFIG['sequence_length'])
            if len(X_seq) == 0: continue
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

        model.fit(X_train, y_train, sample_weight=sample_weights, validation_data=(X_val, y_val),
                  epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'], callbacks=callbacks, verbose=2)

        print(f"\n正在评估第 {fold + 1} 折的所有验证文件 (策略: {CONFIG['smoothing_strategy']})...")
        fold_trial_maes = []

        for trial_name, test_feat, test_label in val_trials:
            f_scaled = scaler.transform(test_feat)
            f_pca = pca.transform(f_scaled)
            X_test_trial, y_test_trial = create_sequences(f_pca, test_label, CONFIG['sequence_length'])
            if len(X_test_trial) == 0: continue

            preds_raw = model.predict(X_test_trial, verbose=0)
            strategy = CONFIG['smoothing_strategy']

            # [新增] Raw 原生输出策略，不再经过任何平滑器，杜绝强行滞后
            if strategy == 'raw':
                raw_rad = np.arctan2(preds_raw[:, 0], preds_raw[:, 1])
                pred_phases = ((raw_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)).tolist()

            elif strategy == 'monotonic':
                smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
                base_phases = [smoother.update(s, c) for s, c in preds_raw]
                pred_phases = []
                for i, p in enumerate(base_phases):
                    if i == 0:
                        pred_phases.append(p)
                        continue
                    prev_p = pred_phases[-1]
                    if prev_p > 0.6 and p < 0.4:
                        pred_phases.append(p)
                    elif p < prev_p:
                        if (prev_p - p) < 0.05:
                            pred_phases.append(prev_p)
                        else:
                            pred_phases.append(prev_p)
                    else:
                        pred_phases.append(p)

            elif strategy == 'lowpass':
                smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
                base_phases = [smoother.update(s, c) for s, c in preds_raw]
                unwrapped_phase = np.unwrap(np.array(base_phases) * 2 * np.pi)
                b, a = butter(N=CONFIG['butter_order'], Wn=CONFIG['butter_cutoff'], btype='low')
                smoothed_unwrapped = filtfilt(b, a, unwrapped_phase)
                pred_phases = ((smoothed_unwrapped / (2 * np.pi)) % 1.0).tolist()

            elif strategy == 'robust_tracker':
                tracker = RobustPhaseTracker(alpha=CONFIG['tracker_alpha'], beta=CONFIG['tracker_beta'])
                pred_phases = [tracker.update(s, c) for s, c in preds_raw]

            else:
                smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
                pred_phases = [smoother.update(s, c) for s, c in preds_raw]

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

        tf.keras.backend.clear_session()

        best_fold_idx = np.argmin(all_fold_maes) + 1
        best_mae = np.min(all_fold_maes)
        print(f"\n================================================================")
        print(f"目前最好的是第 {best_fold_idx} 折 (MAE: {best_mae * 100:.2f}%)")
        print(f"正在将其保存为最终部署模型...")

        shutil.copy(os.path.join(CONFIG['save_dir'], f"best_model_fold_{best_fold_idx}.keras"),
                    os.path.join(CONFIG['save_dir'], "best_model_overall.keras"))
        shutil.copy(os.path.join(CONFIG['save_dir'], f"scaler_fold_{best_fold_idx}.pkl"),
                    os.path.join(CONFIG['save_dir'], "best_scaler_overall.pkl"))
        shutil.copy(os.path.join(CONFIG['save_dir'], f"pca_fold_{best_fold_idx}.pkl"),
                    os.path.join(CONFIG['save_dir'], "best_pca_overall.pkl"))

    print("================================================================")
    print(f"K折交叉验证全部完成！")
    print(f"最终评估模型稳定泛化性能的平均 MAE: {np.mean(all_fold_maes) * 100:.2f}%")
    print("================================================================")


if __name__ == "__main__":
    main()