#!/usr/bin/env python3
"""
命令行目标位置发布节点 - 简化版
使用方法：ros2 run soft_drone_controller send_target <x> <y> <z>
"""

import sys
import time

def main():
    # 检查参数
    if len(sys.argv) < 4:
        print("📝 使用方法：ros2 run soft_drone_controller send_target <x> <y> <z>")
        print("示例：ros2 run soft_drone_controller send_target 0.0 0.0 1.2")
        return
    
    try:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        z = float(sys.argv[3])
    except ValueError:
        print("❌ 参数必须是数字！")
        return
    
    # 导入ROS2库（延迟导入，避免在不需要时导入）
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
    except ImportError:
        print("❌ 无法导入ROS2库，请确认ROS2环境已设置")
        return
    
    # 初始化ROS2
    rclpy.init()
    
    # 创建节点
    class TargetPublisher(Node):
        def __init__(self):
            super().__init__('target_publisher')
            self.publisher = self.create_publisher(PoseStamped, '/drone_target_pose', 10)
            
        def publish(self, x, y, z):
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.w = 1.0
            
            self.publisher.publish(msg)
            
            print(f"✅ 已发布目标位置:")
            print(f"   X (前向): {x:.2f} 米")
            print(f"   Y (左向): {y:.2f} 米")
            print(f"   Z (高度): {z:.2f} 米")
    
    # 创建节点并发布消息
    try:
        node = TargetPublisher()
        
        # 发布消息
        node.publish(x, y, z)
        
        # 等待消息发布
        time.sleep(0.1)
        
        # 销毁节点
        node.destroy_node()
        
    finally:
        # 关闭ROS2
        rclpy.shutdown()

if __name__ == '__main__':
    main()