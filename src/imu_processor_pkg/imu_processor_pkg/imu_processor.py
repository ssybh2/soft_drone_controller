import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np
# 导入QoS相关模块
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

class ImuProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # 配置和发布者完全匹配的QoS
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5  # 和发布者的队列深度一致
        )

        # 用匹配的QoS创建订阅者
        self.imu_subscription = self.create_subscription(
            Imu,
            '/ecat/sn2228293/app2/read',
            self.listener_callback,
            qos_profile
        )
        self.imu_subscription

        # 创建发布者
        self.accel_publisher = self.create_publisher(
            Vector3,
            '/filtered_angular_acceleration',
            10
        )

        self.last_angular_velocity = None
        self.last_time = None
        self.alpha = 0.1
        self.filtered_acceleration = np.array([0.0, 0.0, 0.0])

    def listener_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        current_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        if self.last_angular_velocity is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                angular_acceleration = (current_angular_velocity - self.last_angular_velocity) / dt
                self.filtered_acceleration = self.alpha * angular_acceleration + (1 - self.alpha) * self.filtered_acceleration

                # 发布滤波后的角加速度
                accel_msg = Vector3()
                accel_msg.x = self.filtered_acceleration[0]
                accel_msg.y = self.filtered_acceleration[1]
                accel_msg.z = self.filtered_acceleration[2]
                self.accel_publisher.publish(accel_msg)
                self.get_logger().info(f"Filtered Accel: x={accel_msg.x:.4f}, y={accel_msg.y:.4f}, z={accel_msg.z:.4f}")

        self.last_angular_velocity = current_angular_velocity
        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    imu_processor = ImuProcessor()
    rclpy.spin(imu_processor)
    imu_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
