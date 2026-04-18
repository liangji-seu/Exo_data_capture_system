"""Launch file for the gait_verify_manager node."""
from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch configuration for the gait_verify_manager node."""
    pkg_path = get_package_share_path('gait_verify_manager')
    config_file = Path(pkg_path) / 'config/params.yaml'

    gait_verify_node = Node(
        package='gait_verify_manager',
        name='gait_verify_manager',
        executable='gait_verify_node',
        output='screen',
        parameters=[config_file],
    )

    return LaunchDescription([gait_verify_node])
