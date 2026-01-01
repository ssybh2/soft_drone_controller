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
    
    position_control_node = Node(
        package='soft_drone_controller',  # 同一个功能包
        executable='position_control',    # 对应setup.py中新增的console_scripts键
        name='position_control_node',  # 节点名（自定义，便于区分）
        output='screen',                  # 日志输出到终端
        # 可选：添加参数/重映射话题（比如动捕话题名）
        # parameters=[{'CONTROL_FREQ': 100}],  # 覆盖配置参数
        # remappings=[('/Tracker0/pose', '/Tracker2/pose')]  # 话题重映射
    )
    
    position_cmd_node = Node(
        package='soft_drone_controller',
        executable='position_cmd',
        name='position_cmd_node',
        output='screen',
    )
    pos_path_to_nav_path_node = Node(
        package='soft_drone_controller',
        executable='pos_path_to_nav_path',
        name='pos_path_to_nav_path_node',
        output='screen',
    )


    return LaunchDescription([
        drone_controller_node,
        position_control_node,
        position_cmd_node,
        pos_path_to_nav_path_node
    ])

