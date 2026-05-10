#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "imu_msgs/msg/imu_data_array.hpp"
#include "exo_msgs/msg/joint_torque.hpp"

namespace gait_verify_manager
{

// ==================== 步态检测器常量 ====================
constexpr double DEFAULT_STRIDE_PERIOD     = 1.2;      // 默认步态周期 (s)
constexpr double SWING_THRESHOLD           = 0.5;      // 摆动相检测阈值 (rad/s)
constexpr double MOTION_TIMEOUT            = 2.0;      // 静止超时 (s)
constexpr double STRIDE_PERIOD_MIN         = 0.5;      // 步态周期下限 (s)
constexpr double STRIDE_PERIOD_MAX         = 2.5;      // 步态周期上限 (s)
constexpr double STRIDE_PERIOD_FILTER_OLD  = 0.8;      // 步态周期低通滤波旧值权重
constexpr double STRIDE_PERIOD_FILTER_NEW  = 0.2;      // 步态周期低通滤波新值权重
constexpr double SWING_EXIT_FACTOR         = 0.5;      // 退出摆动相的阈值因子
constexpr double MOTION_DETECT_FACTOR      = 0.3;      // 运动检测阈值因子
constexpr double PHASE_RESTART_THRESHOLD   = 0.8;      // 相位重置最低阈值

// ==================== 节点默认参数 ====================
constexpr double DEFAULT_USER_WEIGHT       = 72.0;     // 用户体重 (kg)
constexpr double DEFAULT_PHASE_OFFSET      = 60.0;     // 相位偏移 (%)
constexpr double DEFAULT_GAIT_FILTER_ALPHA = 0.3;      // 滤波系数 (0-1)
constexpr double REFERENCE_WEIGHT          = 72.0;     // 参考体重 (kg)，用于归一化
constexpr double DT_CLAMP_MAX              = 0.1;      // dt 最大钳位 (s)
constexpr double DT_CLAMP_DEFAULT          = 0.01;     // dt 异常时的默认值 (s)
constexpr int    GAIT_PUB_QOS              = 10;       // 发布队列深度
constexpr int    LOG_INTERVAL              = 100;      // 日志输出间隔 (帧数)

// 默认话题名 / ID
constexpr const char* DEFAULT_IMU_TOPIC        = "xsens_imu_data";
constexpr const char* DEFAULT_TORQUE_OUTPUT     = "/exo/joint_torque_spline";
constexpr const char* DEFAULT_LEFT_KNEE_IMU_ID  = "2626";  // 左腿外侧 (10B42626)
constexpr const char* DEFAULT_RIGHT_KNEE_IMU_ID = "260D";  // 右腿外侧 (10B4260D)

// 默认 CSV 路径
constexpr const char* DEFAULT_HIP_TORQUE_LUT_PATH =
    "/home/liangji/exo/src/my_exo/gait_verify_manager/src/hip_torque_lut.csv";

// 髋关节扭矩查找表（基于 Winter 步态数据）
class HipTorqueLUT
{
public:
    HipTorqueLUT();
    bool loadFromCSV(const std::string& csv_path);
    double getTorque(double phase_percent) const;  // phase: 0-100

private:
    std::vector<double> torque_data_;  // 1001 个数据点
};

class KneeGaitDetector
{
public:
    KneeGaitDetector();

    // Update with new angular velocity (Z-axis) and return current phase (0.0-1.0)
    double update(double angular_velocity_z, double dt);

    // Reset detector state
    void reset();

private:
    double prev_velocity_;
    double phase_;
    double last_peak_time_;
    double time_;
    double stride_period_;
    bool in_swing_;
    double last_motion_time_;
};

class GaitVerifyNode : public rclcpp::Node
{
public:
    explicit GaitVerifyNode(const std::string & node_name = "gait_verify_manager");
    ~GaitVerifyNode() = default;

private:
    void imuCallback(const imu_msgs::msg::IMUDataArray::SharedPtr msg);

    // 一阶低通滤波器
    double lowPassFilter(double current, double previous, double alpha);

    // ROS interfaces
    rclcpp::Subscription<imu_msgs::msg::IMUDataArray>::SharedPtr imu_sub_;
    rclcpp::Publisher<exo_msgs::msg::JointTorque>::SharedPtr torque_pub_;

    // Parameters
    std::string left_knee_id_;
    std::string right_knee_id_;
    double user_weight_;      // 用户体重 (kg)
    double phase_offset_;     // 相位偏移 (%)
    double filter_alpha_;     // 滤波器系数

    // Gait detectors
    std::unique_ptr<KneeGaitDetector> left_detector_;
    std::unique_ptr<KneeGaitDetector> right_detector_;

    // Hip torque lookup table
    std::unique_ptr<HipTorqueLUT> hip_torque_lut_;

    // Timing
    rclcpp::Time last_time_;

    // 力矩滤波状态
    double left_torque_filtered_;
    double right_torque_filtered_;
};

}  // namespace gait_verify_manager
