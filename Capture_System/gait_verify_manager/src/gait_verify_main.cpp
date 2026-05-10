#include "gait_verify_manager/gait_verify_node.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<gait_verify_manager::GaitVerifyNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
