#include "gait_verify_manager/gait_verify_node.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace gait_verify_manager
{

// ============================================================================
// HipTorqueLUT Implementation
// ============================================================================

HipTorqueLUT::HipTorqueLUT()
{
    torque_data_.reserve(1001);
}

bool HipTorqueLUT::loadFromCSV(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) return false;

    torque_data_.clear();
    std::string line;
    while (std::getline(file, line))
    {
        size_t comma_pos = line.find(',');
        if (comma_pos != std::string::npos)
        {
            torque_data_.push_back(std::stod(line.substr(comma_pos + 1)));
        }
    }
    return torque_data_.size() == 1001;
}

double HipTorqueLUT::getTorque(double phase_percent) const
{
    if (torque_data_.empty()) return 0.0;

    // 限制范围 0-100
    phase_percent = std::max(0.0, std::min(100.0, phase_percent));

    // 直接索引查找 + 线性插值
    double index = phase_percent * 10.0;  // 0.1% 间隔 -> index
    size_t i0 = static_cast<size_t>(index);
    size_t i1 = std::min(i0 + 1, torque_data_.size() - 1);

    double t = index - i0;
    return torque_data_[i0] + t * (torque_data_[i1] - torque_data_[i0]);
}

// ============================================================================
// KneeGaitDetector Implementation
// ============================================================================

KneeGaitDetector::KneeGaitDetector()
    : prev_velocity_(0.0)
    , phase_(0.0)
    , last_peak_time_(0.0)
    , time_(0.0)
    , stride_period_(DEFAULT_STRIDE_PERIOD)
    , in_swing_(false)
    , last_motion_time_(0.0)
{
}

double KneeGaitDetector::update(double angular_velocity_z, double dt)
{
    time_ += dt;

    // 检测运动状态
    if (std::abs(angular_velocity_z) > SWING_THRESHOLD * MOTION_DETECT_FACTOR)
    {
        last_motion_time_ = time_;
    }

    // 超时无运动，相位归零
    if (time_ - last_motion_time_ > MOTION_TIMEOUT)
    {
        phase_ = 0.0;
        prev_velocity_ = angular_velocity_z;
        return phase_;
    }

    // 检测摆动相开始（Toe-Off 时刻，定义为 0%）
    if (!in_swing_ && std::abs(angular_velocity_z) > SWING_THRESHOLD)
    {
        in_swing_ = true;

        if (last_peak_time_ > 0.0 && phase_ > PHASE_RESTART_THRESHOLD)
        {
            // 计算步态周期并更新（低通滤波）
            double period = time_ - last_peak_time_;
            if (period > STRIDE_PERIOD_MIN && period < STRIDE_PERIOD_MAX)
            {
                stride_period_ = STRIDE_PERIOD_FILTER_OLD * stride_period_
                               + STRIDE_PERIOD_FILTER_NEW * period;
            }
            last_peak_time_ = time_;
            phase_ = 0.0;
        }
        else if (last_peak_time_ == 0.0)
        {
            last_peak_time_ = time_;
            phase_ = 0.0;
        }
    }
    else if (in_swing_ && std::abs(angular_velocity_z) < SWING_THRESHOLD * SWING_EXIT_FACTOR)
    {
        in_swing_ = false;
    }

    // 平滑相位进展（smootherstep 曲线）
    if (last_peak_time_ > 0.0)
    {
        double elapsed = time_ - last_peak_time_;
        double linear_phase = std::min(elapsed / stride_period_, 1.0);

        // smootherstep: 6x^5 - 15x^4 + 10x^3
        phase_ = linear_phase * linear_phase * linear_phase *
                 (linear_phase * (linear_phase * 6.0 - 15.0) + 10.0);
    }

    prev_velocity_ = angular_velocity_z;
    return phase_;
}

void KneeGaitDetector::reset()
{
    prev_velocity_ = 0.0;
    phase_ = 0.0;
    last_peak_time_ = 0.0;
    time_ = 0.0;
    stride_period_ = DEFAULT_STRIDE_PERIOD;
    in_swing_ = false;
}

}  // namespace gait_verify_manager

// ============================================================================
// GaitVerifyNode Implementation
// ============================================================================

namespace gait_verify_manager
{

GaitVerifyNode::GaitVerifyNode(const std::string & node_name)
    : Node(node_name)
    , left_torque_filtered_(0.0)
    , right_torque_filtered_(0.0)
{
    // Parameters
    this->declare_parameter("imu_topic", std::string(DEFAULT_IMU_TOPIC));
    this->declare_parameter("output_topic", std::string(DEFAULT_TORQUE_OUTPUT));
    this->declare_parameter("left_knee_imu_id", std::string(DEFAULT_LEFT_KNEE_IMU_ID));
    this->declare_parameter("right_knee_imu_id", std::string(DEFAULT_RIGHT_KNEE_IMU_ID));
    this->declare_parameter("user_weight", DEFAULT_USER_WEIGHT);
    this->declare_parameter("phase_offset", DEFAULT_PHASE_OFFSET);
    this->declare_parameter("filter_alpha", DEFAULT_GAIT_FILTER_ALPHA);
    this->declare_parameter("hip_torque_lut_path", std::string(DEFAULT_HIP_TORQUE_LUT_PATH));

    auto imu_topic = this->get_parameter("imu_topic").as_string();
    auto output_topic = this->get_parameter("output_topic").as_string();
    left_knee_id_ = this->get_parameter("left_knee_imu_id").as_string();
    right_knee_id_ = this->get_parameter("right_knee_imu_id").as_string();
    user_weight_ = this->get_parameter("user_weight").as_double();
    phase_offset_ = this->get_parameter("phase_offset").as_double();
    filter_alpha_ = this->get_parameter("filter_alpha").as_double();
    auto csv_path = this->get_parameter("hip_torque_lut_path").as_string();

    // Create detectors
    left_detector_ = std::make_unique<KneeGaitDetector>();
    right_detector_ = std::make_unique<KneeGaitDetector>();

    // Load hip torque lookup table
    hip_torque_lut_ = std::make_unique<HipTorqueLUT>();
    if (!hip_torque_lut_->loadFromCSV(csv_path))
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to load hip torque LUT from: %s", csv_path.c_str());
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Hip torque LUT loaded successfully");
    }

    // ROS interfaces
    imu_sub_ = this->create_subscription<imu_msgs::msg::IMUDataArray>(
        imu_topic, GAIT_PUB_QOS,
        std::bind(&GaitVerifyNode::imuCallback, this, std::placeholders::_1));

    torque_pub_ = this->create_publisher<exo_msgs::msg::JointTorque>(output_topic, GAIT_PUB_QOS);

    last_time_ = this->now();

    RCLCPP_INFO(this->get_logger(),
        "GaitVerifyManager started. Left knee: %s, Right knee: %s",
        left_knee_id_.c_str(), right_knee_id_.c_str());
    RCLCPP_INFO(this->get_logger(),
        "User weight: %.1f kg, Phase offset: %.1f%%, Filter alpha: %.2f",
        user_weight_, phase_offset_, filter_alpha_);
}

double GaitVerifyNode::lowPassFilter(double current, double previous, double alpha)
{
    return alpha * current + (1.0 - alpha) * previous;
}

void GaitVerifyNode::imuCallback(const imu_msgs::msg::IMUDataArray::SharedPtr msg)
{
    // Calculate dt
    auto current_time = this->now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt <= 0.0 || dt > DT_CLAMP_MAX) dt = DT_CLAMP_DEFAULT;

    // Find knee IMUs
    const imu_msgs::msg::IMUData* left_knee = nullptr;
    const imu_msgs::msg::IMUData* right_knee = nullptr;

    // 大小写不敏感的子串查找
    auto contains_ci = [](const std::string& haystack, const std::string& needle) {
        auto it = std::search(
            haystack.begin(), haystack.end(),
            needle.begin(), needle.end(),
            [](char a, char b) { return std::tolower(a) == std::tolower(b); });
        return it != haystack.end();
    };

    for (const auto& imu : msg->imu_data)
    {
        RCLCPP_DEBUG(this->get_logger(), "IMU ID: '%s'", imu.id.c_str());

        if (contains_ci(imu.id, left_knee_id_))
        {
            left_knee = &imu;
            RCLCPP_DEBUG(this->get_logger(), "Matched left knee: %s", imu.id.c_str());
        }
        else if (contains_ci(imu.id, right_knee_id_))
        {
            right_knee = &imu;
            RCLCPP_DEBUG(this->get_logger(), "Matched right knee: %s", imu.id.c_str());
        }
    }

    if (!left_knee || !right_knee)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Could not find both knee IMUs. Left: %s, Right: %s",
            left_knee ? "found" : "missing", right_knee ? "found" : "missing");
        return;
    }

    // 更新相位检测器
    double left_phase = left_detector_->update(-left_knee->angular_velocity.z, dt);
    double right_phase = right_detector_->update(right_knee->angular_velocity.z, dt);

    // ========================================================================
    // 相位对齐与力矩计算
    // ========================================================================
    double left_phase_corrected = std::fmod(left_phase * 100.0 + phase_offset_, 100.0);
    double right_phase_corrected = std::fmod(right_phase * 100.0 + phase_offset_, 100.0);

    // 从查找表获取标准化力矩
    double left_torque_base = hip_torque_lut_->getTorque(left_phase_corrected);
    double right_torque_base = hip_torque_lut_->getTorque(right_phase_corrected);

    // 体重归一化
    double weight_factor = user_weight_ / REFERENCE_WEIGHT;
    double left_torque_raw = left_torque_base * weight_factor;
    double right_torque_raw = right_torque_base * weight_factor;

    // 一阶低通滤波
    left_torque_filtered_ = lowPassFilter(left_torque_raw, left_torque_filtered_, filter_alpha_);
    right_torque_filtered_ = lowPassFilter(right_torque_raw, right_torque_filtered_, filter_alpha_);

    // 周期性日志输出
    static int counter = 0;
    if (++counter % LOG_INTERVAL == 0)
    {
        RCLCPP_INFO(this->get_logger(),
            "L_phase: %.1f%% (%.1f%%), R_phase: %.1f%% (%.1f%%) | Hip: L=%.2f R=%.2f Nm",
            left_phase * 100.0, left_phase_corrected,
            right_phase * 100.0, right_phase_corrected,
            left_torque_filtered_, right_torque_filtered_);
    }

    // 发布关节扭矩消息
    auto torque_msg = exo_msgs::msg::JointTorque();
    torque_msg.header.stamp = current_time;
    torque_msg.header.frame_id = "base_link";
    torque_msg.source_label = "spline";
    torque_msg.left_phase = left_phase;
    torque_msg.right_phase = right_phase;
    torque_msg.gait_phase = (left_phase + right_phase) / 2.0;
    torque_msg.joint_torques = {left_torque_filtered_, 0.0, right_torque_filtered_, 0.0};

    torque_pub_->publish(torque_msg);
}

}  // namespace gait_verify_manager
