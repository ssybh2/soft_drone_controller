import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
# 如果你有自定义消息
from custom_msgs.msg import ReadDJIRC  # 确认你的包和消息名，与C++一致
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
class SquareTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('square_trajectory_publisher')
        
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # 发布目标点
        self.publisher = self.create_publisher(Float64MultiArray, '/pos_path', 10)
        # TF可选，不用主控就略过
        self.tf_broadcaster = TransformBroadcaster(self)

        # 遥控器订阅
        self.rc_sub = self.create_subscription(
            ReadDJIRC,
            '/ecat/sn2228293/app1/read',
            self.rc_callback,
            qos_best_effort
        )
        self.rc_data = None
        self.recv = False

        # 参数
        self.declare_parameter('side_length', 0.5)
        self.declare_parameter('fly_speed', 0.1)
        self.side_length = self.get_parameter('side_length').value
        self.fly_speed = self.get_parameter('fly_speed').value

        self.total_time_per_side = self.side_length / self.fly_speed
        self.omega = self.fly_speed / self.side_length

        self.x = 0.0
        self.y = 0.0
        self.segment_time = 0.0
        self.current_side = 0
        self.dt = 1.0 / 90.0
        self.start_time = self.get_clock().now().nanoseconds / 1e9  # 秒

        self.timer = self.create_timer(self.dt, self.do_control)
        self.get_logger().info("path started.")

    def rc_callback(self, msg):
        self.rc_data = msg
        self.recv = True

    def do_control(self):
        #if not self.recv:
            #return
        # 只有遥控通道6拉到1，才真的运动,
        #if self.rc_data.right_switch == 1:
            #return

        # 遥控器通道4归零后，轨迹复位
        #if self.rc_data.left_switch == 1:
            #self.x = 0
            #self.y = 0
            #self.segment_time = 0
            #self.current_side = 0
            #self.start_time = self.get_clock().now().nanoseconds / 1e9
        #else:
        t = self.get_clock().now().nanoseconds / 1e9 - self.start_time
            # 简化成画圆，也可按你注释放出来写方形轨迹
        self.x = 0.0#self.side_length * math.cos(self.omega * t + math.pi) + self.side_length
        self.y = 0.0#self.side_length * math.sin(self.omega * t + math.pi)
            # 也可以用切换4段，每段走一条边，代码见原C++注释

        # 发布TF（可选）
        transformStamped = TransformStamped()
        transformStamped.header.stamp = self.get_clock().now().to_msg()
        transformStamped.header.frame_id = "armed_pos"
        transformStamped.child_frame_id = "desired_pos"
        transformStamped.transform.translation.x = float(self.x)
        transformStamped.transform.translation.y = float(self.y)
        transformStamped.transform.translation.z = 0.0
        transformStamped.transform.rotation.x = 0.0
        transformStamped.transform.rotation.y = 0.0
        transformStamped.transform.rotation.z = 0.0
        transformStamped.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transformStamped)

        # 发布目标点
        msg = Float64MultiArray()
        msg.data = [self.x, self.y]
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
