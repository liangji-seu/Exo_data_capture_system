cmake_minimum_required(VERSION 3.8)
project(gait_verify_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(imu_msgs REQUIRED)
find_package(exo_msgs REQUIRED)

# Include directories
include_directories(include)

# Executable
add_executable(gait_verify_node
  src/gait_verify_node.cpp
  src/gait_verify_main.cpp
)

ament_target_dependencies(gait_verify_node
  rclcpp
  imu_msgs
  exo_msgs
)

# Install
install(TARGETS gait_verify_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
