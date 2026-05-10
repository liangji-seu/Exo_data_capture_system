import os
import glob
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

# 初始化绘图配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# ============================== 0. 全局中控台 (CONFIG) ==============================
# ==============================================================================
CONFIG = {
    # ---------------- [模块一] 路径与环境设置 ----------------
    'data_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "数据集文件", "Processed_Final_Training_Set_clean_v2_fix_shift_v2"),
    'save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_logs", "bilstm_model"),
    'cv_random_seed': 42,            # 交叉验证的全局随机种子

    # ---------------- [模块二] 数据处理与特征工程 ----------------
    'pca_components': 25,            # PCA 降维保留的特征维度
    'sequence_length': 40,           # 双向 LSTM 序列时间步长度
    
    # 功能开关：数据增强
    'enable_data_augmentation': True, 
    'augment_factors':[0.8, 0.9, 1.0, 1.1, 1.2],

    # 功能开关：边界样本加权 (强迫模型关注跳变点)
    'edge_weight_enable': True,      
    'edge_ratio': 0.15,              # 相位 <0.15 或 >0.85 认为是跳变边界
    'edge_weight': 3.0,              # 边界样本的损失权重放大倍数

    # ---------------- [模块三] 模型架构参数 ----------------
    'lstm_1_units': 96,              # 第一层双向 LSTM 单元数
    'lstm_2_units': 48,              # 第二层双向 LSTM 单元数
    'dense_units': 32,               # 全连接层单元数
    'dropout_rate': 0.3,             # 丢弃率
    'l2_reg': 0.001,                 # L2 正则化系数

    # ---------------- [模块四] 训练与评估控制 ----------------
    'k_folds': 5,                    # K折交叉验证折数
    'epochs': 150,                   # 最大训练轮数
    'batch_size': 32,                # 批次大小
    'learning_rate': 0.005,          # 初始学习率
    
    # 动态学习率与早停控制
    'reduce_lr_factor': 0.5,         
    'reduce_lr_patience': 6,         
    'early_stop_patience': 30,       

    # ---------------- [模块五] 原生后处理策略 (用于折内直接验证) ----------------
    # 可选字段: 'raw', 'monotonic', 'lowpass', 'robust_tracker', 'ema'
    'smoothing_strategy': 'monotonic',  
    
    # 追踪器与平滑器参数
    'tracker_alpha': 0.85,           
    'tracker_beta': 0.05,            
    'ema_alpha': 0.6,                
    'butter_order': 3,               
    'butter_cutoff': 0.1,            
    'plot_limit_frames': 500,        # 画图截取的最大帧数
}

if not os.path.exists(CONFIG['save_dir']):
    os.makedirs(CONFIG['save_dir'])


# ==============================================================================
# ========================= 模块一：后处理与平滑追踪器 =========================
# ==============================================================================
class EnhancedVectorSmoother:
    """对模型输出的 (sin, cos) 做 EMA 平滑，得到更稳定的相位（0~1）。

    - alpha 越大：更信任当前帧预测（跟得更紧，但可能更抖）
    - alpha 越小：更平滑（更抗抖，但可能引入相位滞后）
    """
    def __init__(self, alpha=CONFIG['ema_alpha']):
        self.alpha = alpha
        self.smooth_sin = 0.0
        self.smooth_cos = 1.0
        self.initialized = False

    def update(self, raw_sin, raw_cos):
        if not self.initialized:
            self.smooth_sin, self.smooth_cos = raw_sin, raw_cos
            self.initialized = True
        else:
            self.smooth_sin = self.alpha * raw_sin + (1 - self.alpha) * self.smooth_sin
            self.smooth_cos = self.alpha * raw_cos + (1 - self.alpha) * self.smooth_cos

        phase_rad = np.arctan2(self.smooth_sin, self.smooth_cos)
        return (phase_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

class RobustPhaseTracker:
    """“带惯性”的鲁棒相位跟踪器（近似卡尔曼更新思想）。

    输入为模型预测的 (sin, cos)，内部维护：
    - estimated_phase：当前相位估计（0~1）
    - estimated_velocity：相位推进速度（每帧步进的大致假设）

    其中：
    - alpha：误差融合比例（信任观测 vs 信任惯性）
    - beta：用于缓慢校准速度（避免长期漂移）
    """
    def __init__(self, alpha=CONFIG.get('tracker_alpha', 0.85), beta=CONFIG.get('tracker_beta', 0.05)):
        self.alpha = alpha
        self.beta = beta
        self.estimated_phase = None
        self.estimated_velocity = 0.04
        self.initialized = False

    def update(self, raw_sin, raw_cos):
        obs_phase = (np.arctan2(raw_sin, raw_cos) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

        if not self.initialized:
            self.estimated_phase = obs_phase
            self.initialized = True
            return self.estimated_phase

        pred_phase = (self.estimated_phase + self.estimated_velocity) % 1.0
        error = obs_phase - pred_phase
        if error > 0.5: error -= 1.0
        elif error < -0.5: error += 1.0

        effective_error = np.clip(error, -0.4, 0.4)
        self.estimated_phase = (pred_phase + self.alpha * effective_error) % 1.0
        self.estimated_velocity = np.clip(self.estimated_velocity + self.beta * effective_error, 0.015, 0.1)

        return self.estimated_phase

class SmartGatedPhaseTracker:
    """注：保留此高级追踪器以供外部脚本 (模型9_处理.py) 动态调用"""
    def __init__(self, base_velocity=0.035):  
        self.estimated_phase = None
        self.velocity = base_velocity
        self.initialized = False

    def update(self, raw_sin, raw_cos):
        model_phase = (np.arctan2(raw_sin, raw_cos) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

        if not self.initialized:
            self.estimated_phase = model_phase
            self.initialized = True
            return self.estimated_phase

        pred_phase = (self.estimated_phase + self.velocity) % 1.0
        distance_to_edge = min(model_phase, 1.0 - model_phase)
        trust_in_model = 0.85 if distance_to_edge < 0.15 else 0.05

        error = model_phase - pred_phase
        if error > 0.5: error -= 1.0
        elif error < -0.5: error += 1.0

        if trust_in_model > 0.5:
            self.velocity = np.clip(self.velocity + 0.05 * error, 0.015, 0.08)

        self.estimated_phase = (pred_phase + trust_in_model * error) % 1.0
        return self.estimated_phase


# ==============================================================================
# ========================= 模块二：数据加载与预处理 ===========================
# ==============================================================================
def robust_cos_reconstruction(sin_values):
    """当标签只有 Label_Sin（缺少 Label_Cos）时，用 Hilbert 解析信号估计瞬时相位，
    再从相位重建 cos，使标签落在单位圆上。"""
    sin_centered = sin_values - np.mean(sin_values)
    analytic_signal = hilbert(sin_centered)
    return np.cos(np.angle(analytic_signal))

def augment_time_warp(features, labels, factor):
    """时间伸缩数据增强：将序列长度按 factor 重采样（resample）。

    - factor > 1：拉长（速度变慢）
    - factor < 1：缩短（速度变快）
    """
    if factor == 1.0: 
        return features, labels
    new_len = int(len(features) * factor)
    return resample(features, new_len, axis=0), resample(labels, new_len, axis=0)

def load_all_data_without_split(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.xlsx"))
    valid_trials =[]

    for f in all_files:
        fname = os.path.basename(f)
        if fname.startswith('~$'): continue

        try:
            if f.endswith('.xlsx'):
                sheet_dict = pd.read_excel(f, sheet_name=None)
                sheet_features =[]
                labels = None
                for sheet_name, df in sheet_dict.items():
                    df.columns = df.columns.str.strip()
                    feat_cols =[c for c in df.columns if str(c).startswith('Feat_')]
                    if not feat_cols: continue
                    sheet_features.append(df[feat_cols].values)

                    if labels is None and 'Label_Sin' in df.columns:
                        sin_vals = df['Label_Sin'].values
                        cos_vals = df['Label_Cos'].values if 'Label_Cos' in df.columns else robust_cos_reconstruction(sin_vals)
                        labels = np.column_stack((sin_vals, cos_vals))
                if sheet_features and labels is not None:
                    valid_trials.append((fname, np.concatenate(sheet_features, axis=1), labels))

            elif f.endswith('.csv'):
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                feat_cols =[c for c in df.columns if str(c).startswith('Feat_')]
                if not feat_cols or 'Label_Sin' not in df.columns: continue
                
                sin_vals = df['Label_Sin'].values
                cos_vals = df['Label_Cos'].values if 'Label_Cos' in df.columns else robust_cos_reconstruction(sin_vals)
                valid_trials.append((fname, df[feat_cols].values, np.column_stack((sin_vals, cos_vals))))

        except Exception as e:
            print(f"处理文件 {fname} 时出错: {e}")
            continue

    return valid_trials

def create_sequences(features, labels, time_steps):
    Xs, ys = [],[]
    for i in range(len(features) - time_steps):
        Xs.append(features[i: i + time_steps])
        ys.append(labels[i + time_steps])
    return np.array(Xs), np.array(ys)

def create_boundary_weights(y_seq, edge_ratio=0.15, edge_weight=2.0):
    """对相位序列的跳变边界附近更高加权。

    将相位映射到 [0, 1] 后，如果相位落在：
    - [0, edge_ratio] 或 [1-edge_ratio, 1]
    则权重设置为 edge_weight，用于强化起步/收尾的拟合质量。
    """
    if len(y_seq) == 0:
        return np.array([], dtype=np.float32)

    true_rad = np.arctan2(y_seq[:, 0], y_seq[:, 1])
    true_phases = (true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

    weights = np.ones(len(y_seq), dtype=np.float32)
    edge_mask = (true_phases < edge_ratio) | (true_phases > (1.0 - edge_ratio))
    weights[edge_mask] = edge_weight
    return weights


# ==============================================================================
# ========================= 模块三：模型构建与损失函数 =========================
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
def phase_mae(y_true, y_pred):
    """周期相位 MAE：把 (sin, cos) 转成相位（0~1），再用圆周距离计算误差。"""
    true_phase = tf.math.atan2(y_true[:, 0], y_true[:, 1])
    pred_phase = tf.math.atan2(y_pred[:, 0], y_pred[:, 1])
    true_phase = tf.math.floormod(true_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    pred_phase = tf.math.floormod(pred_phase + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
    diff = tf.math.abs(true_phase - pred_phase)
    circ_diff = tf.math.minimum(diff, 1.0 - diff)
    return tf.math.reduce_mean(circ_diff)

@tf.keras.utils.register_keras_serializable()
def cosine_loss(y_true, y_pred):
    """余弦距离损失：把向量归一到单位圆上后，最小化 1 - cos(theta)。

    y_true/y_pred 对应 (sin, cos) 的方向一致性，方向越一致 loss 越小。
    """
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    return tf.reduce_mean(1.0 - dot_product)

def build_fusion_model(input_shape):
    """主网络：特征注意力(SE风格) + 双向 LSTM + L2 单位化输出。

    输出层使用 UnitNormalization 确保预测向量始终落在单位圆（sin/cos 合理）。
    """
    inputs = Input(shape=input_shape)

    # 特征注意力机制
    global_avg = GlobalAveragePooling1D()(inputs)
    att_bottleneck = Dense(8, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(global_avg)
    attention_probs = Dense(input_shape[1], activation='sigmoid', name='feature_attention_weights')(att_bottleneck)
    attention_probs_reshaped = Reshape((1, input_shape[1]))(attention_probs)
    x = Multiply(name='attention_multiply')([inputs, attention_probs_reshaped])

    # 双向 LSTM 处理层
    x = Bidirectional(LSTM(CONFIG['lstm_1_units'], return_sequences=True,
                           kernel_regularizer=l2(CONFIG['l2_reg']),
                           recurrent_regularizer=l2(CONFIG['l2_reg'])))(x)
    x = BatchNormalization()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    x = Bidirectional(LSTM(CONFIG['lstm_2_units'], return_sequences=False,
                           kernel_regularizer=l2(CONFIG['l2_reg'])))(x)
    x = BatchNormalization()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    # 全连接与输出单位化
    x = Dense(CONFIG['dense_units'], activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    raw_outputs = Dense(2, activation='linear')(x)
    outputs = UnitNormalization(axis=-1, name='l2_norm_output')(raw_outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), loss=cosine_loss, metrics=['mae', phase_mae])
    return model


# ==============================================================================
# ======================== 模块四：训练评估主程序管线 ==========================
# ==============================================================================
def main():
    """双向 LSTM 训练主流程：
    1) K-fold 划分（以 trial 为单位）
    2) 训练集可选数据增强（时间伸缩）
    3) StandardScaler + PCA
    4) 训练（含边界样本加权）并在每折保存最优模型
    """
    print("1. 加载所有原始数据...")
    all_trials = load_all_data_without_split(CONFIG['data_dir'])
    if not all_trials: 
        raise ValueError("未找到数据，请检查 data_dir！")

    k_folds = CONFIG['k_folds']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=CONFIG['cv_random_seed'])
    all_fold_maes =[]

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_trials)):
        print(f"\n================ 开始训练第 {fold + 1}/{k_folds} 折 ================")
        train_trials = [all_trials[i] for i in train_idx]
        val_trials = [all_trials[i] for i in val_idx]

        # ---------------- 1. 数据增强 ----------------
        aug_train_features, aug_train_labels = [],[]
        factors = CONFIG['augment_factors'] if CONFIG['enable_data_augmentation'] else [1.0]
        for _, feat, label in train_trials:
            for factor in factors:
                f_aug, l_aug = augment_time_warp(feat, label, factor)
                aug_train_features.append(f_aug)
                aug_train_labels.append(l_aug)

        # ---------------- 2. 特征缩放与 PCA 降维 ----------------
        all_train_feat_concat = np.vstack(aug_train_features)
        scaler = StandardScaler()
        all_train_feat_scaled = scaler.fit_transform(all_train_feat_concat)
        pca = PCA(n_components=CONFIG['pca_components'])
        pca.fit(all_train_feat_scaled)
        
        joblib.dump(scaler, os.path.join(CONFIG['save_dir'], f'scaler_fold_{fold + 1}.pkl'))
        joblib.dump(pca, os.path.join(CONFIG['save_dir'], f'pca_fold_{fold + 1}.pkl'))
        
        # ---------------- 3. 构造序列与样本加权 ----------------
        X_train_seqs, y_train_seqs, train_weights = [], [],[]
        for feat, label in zip(aug_train_features, aug_train_labels):
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = create_sequences(f_pca, label, CONFIG['sequence_length'])
            if len(X_seq) == 0: continue

            X_train_seqs.append(X_seq)
            y_train_seqs.append(y_seq)

            if CONFIG['edge_weight_enable']:
                w_seq = create_boundary_weights(y_seq, CONFIG['edge_ratio'], CONFIG['edge_weight'])
            else:
                w_seq = np.ones(len(X_seq), dtype=np.float32)
            train_weights.append(w_seq)

        X_train, y_train = np.vstack(X_train_seqs), np.vstack(y_train_seqs)
        sample_weights = np.concatenate(train_weights).astype(np.float32)

        # 打乱训练集
        idx = np.random.permutation(len(X_train))
        X_train, y_train, sample_weights = X_train[idx], y_train[idx], sample_weights[idx]

        # ---------------- 4. 构造验证集 ----------------
        X_val_seqs, y_val_seqs = [],[]
        for _, feat, label in val_trials:
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = create_sequences(f_pca, label, CONFIG['sequence_length'])
            if len(X_seq) == 0: continue
            X_val_seqs.append(X_seq)
            y_val_seqs.append(y_seq)
        X_val, y_val = np.vstack(X_val_seqs), np.vstack(y_val_seqs)

        # ---------------- 5. 模型训练 ----------------
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

        # ---------------- 6. 折内策略评估 ----------------
        print(f"\n正在评估第 {fold + 1} 折验证文件 (策略: {CONFIG['smoothing_strategy']})...")
        fold_trial_maes =[]
        strategy = CONFIG['smoothing_strategy']

        for trial_name, test_feat, test_label in val_trials:
            f_scaled = scaler.transform(test_feat)
            f_pca = pca.transform(f_scaled)
            X_test_trial, y_test_trial = create_sequences(f_pca, test_label, CONFIG['sequence_length'])
            if len(X_test_trial) == 0: continue

            preds_raw = model.predict(X_test_trial, verbose=0)

            # 根据所选策略进行平滑
            if strategy == 'raw':
                raw_rad = np.arctan2(preds_raw[:, 0], preds_raw[:, 1])
                pred_phases = ((raw_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)).tolist()
            elif strategy == 'monotonic':
                smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
                base_phases =[smoother.update(s, c) for s, c in preds_raw]
                pred_phases =[]
                for i, p in enumerate(base_phases):
                    if i == 0:
                        pred_phases.append(p)
                        continue
                    prev_p = pred_phases[-1]
                    if prev_p > 0.6 and p < 0.4:
                        pred_phases.append(p)
                    elif p < prev_p:
                        pred_phases.append(prev_p if (prev_p - p) < 0.05 else prev_p)
                    else:
                        pred_phases.append(p)
            elif strategy == 'lowpass':
                smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
                base_phases =[smoother.update(s, c) for s, c in preds_raw]
                unwrapped_phase = np.unwrap(np.array(base_phases) * 2 * np.pi)
                b, a = butter(N=CONFIG['butter_order'], Wn=CONFIG['butter_cutoff'], btype='low')
                smoothed_unwrapped = filtfilt(b, a, unwrapped_phase)
                pred_phases = ((smoothed_unwrapped / (2 * np.pi)) % 1.0).tolist()
            elif strategy == 'robust_tracker':
                tracker = RobustPhaseTracker(alpha=CONFIG['tracker_alpha'], beta=CONFIG['tracker_beta'])
                pred_phases =[tracker.update(s, c) for s, c in preds_raw]
            else:
                smoother = EnhancedVectorSmoother(alpha=CONFIG['ema_alpha'])
                pred_phases =[smoother.update(s, c) for s, c in preds_raw]

            # 计算 MAE
            true_phases_rad = np.arctan2(y_test_trial[:, 0], y_test_trial[:, 1])
            true_phases = (true_phases_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
            circ_diff = np.minimum(np.abs(np.array(pred_phases) - true_phases), 1.0 - np.abs(np.array(pred_phases) - true_phases))
            fold_trial_maes.append(np.mean(circ_diff))

        current_fold_mae = np.mean(fold_trial_maes)
        all_fold_maes.append(current_fold_mae)
        print(f"第 {fold + 1} 折验证集平均 MAE: {current_fold_mae * 100:.2f}%\n")

        # ---------------- 7. 折内绘图保存 ----------------
        plt.figure(figsize=(15, 6))
        limit = min(CONFIG['plot_limit_frames'], len(true_phases))
        plt.plot(true_phases[:limit], 'k-', alpha=0.5, linewidth=3, label='Ground Truth Phase')
        plt.plot(pred_phases[:limit], 'r--', linewidth=2, label=f'Predicted Phase ({strategy})')
        plt.title(f"Fold {fold + 1} Phase Prediction - MAE: {current_fold_mae * 100:.2f}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['save_dir'], f"fold_{fold + 1}_result.png"))
        plt.close()

        tf.keras.backend.clear_session()

        # 更新最优折权重并保存
        best_fold_idx = np.argmin(all_fold_maes) + 1
        print(f"目前最好的是第 {best_fold_idx} 折，正在将其保存为最终部署模型...")
        shutil.copy(os.path.join(CONFIG['save_dir'], f"best_model_fold_{best_fold_idx}.keras"),
                    os.path.join(CONFIG['save_dir'], "best_model_overall.keras"))
        shutil.copy(os.path.join(CONFIG['save_dir'], f"scaler_fold_{best_fold_idx}.pkl"),
                    os.path.join(CONFIG['save_dir'], "best_scaler_overall.pkl"))
        shutil.copy(os.path.join(CONFIG['save_dir'], f"pca_fold_{best_fold_idx}.pkl"),
                    os.path.join(CONFIG['save_dir'], "best_pca_overall.pkl"))

    print("================================================================")
    print(f"K折交叉验证全部完成！最终泛化性能平均 MAE: {np.mean(all_fold_maes) * 100:.2f}%")
    print("================================================================")


if __name__ == "__main__":
    main()