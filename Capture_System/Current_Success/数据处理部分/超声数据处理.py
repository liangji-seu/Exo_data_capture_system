import os
import glob
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d

# ================= 1. 全局配置区域 =================
CONFIG = {
    'BASE_DIR': r"D:\nightmare\Documents\SRTP\AllData\2026_1_26\2026_1_26_unpr_data",
    'TOTAL_OUTPUT_DIR': r"D:\nightmare\Documents\SRTP\AllData\2026_1_26\Processed_Final_Training_Set_1_26_1",
    'TARGET_FOLDERS': [str(i) for i in range(7, 31)],

    'POINTS_PER_GROUP': 50,
    'WINDOW_SIZE': 1000,

    'ANTICIPATE_STEPS': 1,
    'SEARCH_RANGE': 100,
    'MIN_CORR_THRESH': 0.20,

    # 🌟 新增：智能切片参数
    'MAX_STEP_GAP_IMU': 150,  # 约 1.5 秒，两次落地间隔超过此值视为“停止休息”
    'MIN_STEPS_PER_SEG': 3,  # 至少连续走 3 步才算一个有效片段 (过滤碎步)
}


# ================= 2. 核心算法函数 =================

def extract_walking_segments(gyro_data, fs=100):
    """【智能切片器】：找出所有连续行走的片段，剔除长时间停止"""
    b, a = butter(3, 5 / (fs / 2), btype='low')
    gyro_smooth = filtfilt(b, a, gyro_data)

    peak_height = max(np.max(gyro_smooth) * 0.4, 300)
    swing_peaks, _ = find_peaks(gyro_smooth, height=peak_height, distance=50)

    hs_indices = []
    for p in swing_peaks:
        search_window = gyro_smooth[p: p + 50]
        if len(search_window) > 0:
            hs_indices.append(p + np.argmin(search_window))

    hs_indices = np.unique(hs_indices)
    segments = []
    if len(hs_indices) < 2: return segments

    # 如果两步之间间隔大于设定值，就切断成新片段
    current_seg = [hs_indices[0]]
    for i in range(1, len(hs_indices)):
        if hs_indices[i] - hs_indices[i - 1] > CONFIG['MAX_STEP_GAP_IMU']:
            if len(current_seg) >= CONFIG['MIN_STEPS_PER_SEG']:
                segments.append(current_seg)
            current_seg = [hs_indices[i]]
        else:
            current_seg.append(hs_indices[i])

    if len(current_seg) >= CONFIG['MIN_STEPS_PER_SEG']:
        segments.append(current_seg)

    return segments


def find_best_shift_and_corr(features, target_sin, search_range=50):
    envelope = features.mean(axis=1)
    env_norm = (envelope - envelope.mean()) / (envelope.std() + 1e-6)
    tgt_norm = (target_sin - target_sin.mean()) / (target_sin.std() + 1e-6)

    best_corr, best_shift = -1, 0
    for shift in range(-search_range, search_range + 1):
        if shift < 0:
            e, t = env_norm[:shift], tgt_norm[-shift:]
        elif shift > 0:
            e, t = env_norm[shift:], tgt_norm[:-shift]
        else:
            e, t = env_norm, tgt_norm

        corr = np.corrcoef(e, t)[0, 1]
        if corr > best_corr:
            best_corr, best_shift = corr, shift
    return best_shift, best_corr


def find_matched_files(folder_path):
    us_files = glob.glob(os.path.join(folder_path, "**", "*ultr*.csv"), recursive=True)
    imu_files = [f for f in glob.glob(os.path.join(folder_path, "**", "*.[cx][sl]*"), recursive=True)
                 if '右' in os.path.basename(f).lower() or 'right' in os.path.basename(
            f).lower() and 'ultr' not in f.lower() and 'ready' not in f.lower()]
    return us_files[0] if us_files else None, imu_files[0] if imu_files else None


# ================= 3. 主干流水线 =================

def process_all_subjects():
    if not os.path.exists(CONFIG['TOTAL_OUTPUT_DIR']): os.makedirs(CONFIG['TOTAL_OUTPUT_DIR'])
    print("开始【智能切片版】数据清洗与对齐流水线...\n")
    success_count = fail_count = 0

    for folder_name in CONFIG['TARGET_FOLDERS']:
        subject_dir = os.path.join(CONFIG['BASE_DIR'], folder_name)
        if not os.path.exists(subject_dir): continue

        print(f"{'-' * 55}\n扫描受试者: [{folder_name}]")
        us_file, imu_file = find_matched_files(subject_dir)
        if not us_file or not imu_file:
            print("  跳过: 缺失配对文件")
            continue

        try:
            df_imu = pd.read_csv(imu_file) if imu_file.endswith('.csv') else pd.read_excel(imu_file)

            # 🌟 核心：获取智能切片
            segments = extract_walking_segments(df_imu['X轴角速度'].values)
            if not segments: raise ValueError("未检测到足够长的连续行走片段(全是碎步或静止)")

            df_us = pd.read_csv(us_file).interpolate(method='linear').ffill().bfill().fillna(130.0)
            ultr_columns = [c for c in df_us.columns if str(c).startswith('ultr')]

            test_raw = pd.to_numeric(df_us[ultr_columns[0]], errors='coerce').fillna(130.0).values
            dynamic_num_groups = len(test_raw) // CONFIG['WINDOW_SIZE']
            ratio = dynamic_num_groups / len(df_imu)  # 超声帧率与IMU帧率的换算比

            test_groups = [test_raw[i * CONFIG['WINDOW_SIZE']: i * CONFIG['WINDOW_SIZE'] + CONFIG['POINTS_PER_GROUP']]
                           for i in range(dynamic_num_groups)]
            test_features = np.array(test_groups)

            # 针对切出来的每一个有效连续行走片段，独立打包！
            for seg_idx, seg_hs in enumerate(segments):
                # 预留 25 帧的 LSTM 冷启动缓冲期
                start_us = max(0, int(seg_hs[0] * ratio) - 25)
                end_us = min(dynamic_num_groups, int(seg_hs[-1] * ratio) + 25)

                if end_us - start_us < 60: continue  # 片段过短

                local_features = test_features[start_us: end_us]
                local_hs_us = [max(0, min(int(hs * ratio) - start_us, end_us - start_us - 1)) for hs in seg_hs]

                f_phase = interp1d(local_hs_us, np.arange(len(local_hs_us)), fill_value='extrapolate')
                t_us_local = np.arange(end_us - start_us)
                base_gait_phase = f_phase(t_us_local) % 1.0
                base_label_sin = np.sin(base_gait_phase * 2 * np.pi)

                best_shift, best_corr = find_best_shift_and_corr(local_features, base_label_sin, CONFIG['SEARCH_RANGE'])
                if best_corr < CONFIG['MIN_CORR_THRESH']:
                    print(f"  片段 {seg_idx + 1} 对齐失败 (Corr:{best_corr:.2f})，剔除。")
                    continue

                final_shift = best_shift + CONFIG['ANTICIPATE_STEPS']
                t_us_shifted = t_us_local - final_shift
                final_gait_phase = f_phase(t_us_shifted) % 1.0

                label_df = pd.DataFrame({
                    'GaitPhase': final_gait_phase,
                    'Label_Sin': np.sin(final_gait_phase * 2 * np.pi),
                    'Label_Cos': np.cos(final_gait_phase * 2 * np.pi)
                })

                out_filename = f"Sub_{folder_name}_Seg_{seg_idx + 1}_Aligned.xlsx"
                out_path = os.path.join(CONFIG['TOTAL_OUTPUT_DIR'], out_filename)

                with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                    for col in ultr_columns:
                        raw = pd.to_numeric(df_us[col], errors='coerce').fillna(130.0).values
                        local_raw = raw[start_us * CONFIG['WINDOW_SIZE']: end_us * CONFIG['WINDOW_SIZE']]
                        groups = [
                            local_raw[i * CONFIG['WINDOW_SIZE']: i * CONFIG['WINDOW_SIZE'] + CONFIG['POINTS_PER_GROUP']]
                            for i in range(end_us - start_us)]

                        df_out = pd.DataFrame(np.array(groups),
                                              columns=[f"Feat_{j + 1}" for j in range(CONFIG['POINTS_PER_GROUP'])])
                        df_out = pd.concat([label_df, df_out], axis=1)
                        sheet_name = re.sub(r'[\\/*?:[\]]', '_', col.strip())[:31]
                        df_out.to_excel(writer, sheet_name=sheet_name, index=False)

                print(
                    f"  提炼成功: 受试者 {folder_name} 的第 {seg_idx + 1} 个连续行走片段 (长 {end_us - start_us} 帧, Corr:{best_corr:.2f})")
                success_count += 1

        except Exception as e:
            print(f"  提取异常: {str(e)}")
            fail_count += 1

    print(f"\n{'=' * 55}\n流水线执行完毕！成功切割并提炼了 {success_count} 个连续行走片段。")


if __name__ == "__main__":
    process_all_subjects()