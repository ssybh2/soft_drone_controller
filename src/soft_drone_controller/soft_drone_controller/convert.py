#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

def quat_mult(q1, q2):
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_norm(q):
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

class MocapY180Relay(Node):
    def __init__(self):
        super().__init__('mocap_y180_relay')

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            PoseStamped, '/Tracker0/pose', self.cb, qos_best_effort
        )
        self.pub = self.create_publisher(
            PoseStamped, '/Tracker0/pose_y180', qos_best_effort
        )

        # 绕机体系Y轴旋转180°：q_y180 = [cos(pi/2), 0, sin(pi/2), 0] = [0, 0, 1, 0]
        self.q_y180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

        # 默认：body-fixed（右乘）。如果你发现方向不对，把这个改成 "LEFT"
        self.mode = "RIGHT"  # "RIGHT" or "LEFT"

        self.get_logger().info("✅ Mocap relay: /Tracker0/pose -> /Tracker0/pose_y180 (Y+180deg, body-fixed RIGHT multiply)")

    def cb(self, msg: PoseStamped):
        q = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ], dtype=float)

        if self.mode == "RIGHT":
            q_new = quat_mult(q, self.q_y180)     # q' = q ⊗ q_y180
        else:
            q_new = quat_mult(self.q_y180, q)     # q' = q_y180 ⊗ q

        q_new = quat_norm(q_new)

        out = PoseStamped()
        out.header = msg.header
        out.pose = msg.pose
        out.pose.orientation.w = float(q_new[0])
        out.pose.orientation.x = float(q_new[1])
        out.pose.orientation.y = float(q_new[2])
        out.pose.orientation.z = float(q_new[3])

        self.pub.publish(out)

def main():
    rclpy.init()
    node = MocapY180Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
