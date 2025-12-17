from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 无人机控制器节点
    drone_controller_node = Node(
        package='soft_drone_controller',
        executable='drone_controller',
        name='drone_controller_node',
        output='screen',
        parameters=[
            # 可在这里覆盖配置参数（可选）
            {'DSHOT_MIN': 48},
            {'DSHOT_MAX': 2047}
        ]
    )

    return LaunchDescription([
        drone_controller_node
    ])

