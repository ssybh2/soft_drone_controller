# position_controller.py (已修正)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3
from custom_msgs.msg import ReadDJIRC
import numpy as np
import tf_transformations
import tf2_ros
import time
from soft_drone_controller.config import controller_params as cfg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class AlphaFilter:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.filtered_ = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def update(self, raw):
        self.filtered_ = self.alpha * raw + (1.0 - self.alpha) * self.filtered_

class SimplePID:
    def __init__(self, kp, ki, kd, i_max=1.0, i_min=-1.0):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.i_max = i_max
        self.i_min = i_min
        self.integral = 0.0
        self.prev_error = 0.0

    def calc(self, measurement, setpoint):
        error = setpoint - measurement
        self.integral += error
        self.integral = np.clip(self.integral, self.i_min, self.i_max)
        d_error = error - self.prev_error
        self.prev_error = error
        out = self.Kp * error + self.Ki * self.integral + self.Kd * d_error
        return out

    def clear(self):
        self.integral = 0.0
        self.prev_error = 0.0

class PositionController(Node):
    def __init__(self):
        super().__init__('position_controller')
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # 主题相关
        self.rc_sub = self.create_subscription(
            ReadDJIRC, '/ecat/sn2228293/app1/read', self.rc_callback, qos_best_effort
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/Tracker0/pose', self.pose_callback, qos_best_effort
        )
        self.path_sub = self.create_subscription(
            Float64MultiArray, '/pos_path', self.path_callback, qos_best_effort
        )
        self.pos_cmd_pub = self.create_publisher(
            Vector3, '/attitude_position_cmd', qos_reliable
        )
        

        # 参数读取
        self.x_kp = cfg.POSITION_XY_KP
        self.x_ki = cfg.POSITION_XY_KI
        self.x_kd = cfg.POSITION_XY_KD
        self.y_kp = cfg.POSITION_XY_KP
        self.y_ki = cfg.POSITION_XY_KI
        self.y_kd = cfg.POSITION_XY_KD
        self.z_kp = cfg.POSITION_Z_KP
        self.z_ki = cfg.POSITION_Z_KI
        self.z_kd = cfg.POSITION_Z_KD
        self.height_sp = getattr(cfg, "POSITION_DEFAULT_HEIGHT", 0.5)
        self.hover_th = getattr(cfg, "POSITION_BASE_THROTTLE", 1000.0)
        self.hover_th_ratio = getattr(cfg, "HOVER_THROTTLE_RATIO", 0.5)

        # PID初始化
        self.x_loop = SimplePID(self.x_kp, self.x_ki, self.x_kd, i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.y_loop = SimplePID(self.y_kp, self.y_ki, self.y_kd, i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.z_loop = SimplePID(self.z_kp, self.z_ki, self.z_kd, i_max=cfg.POSITION_Z_INT_LIMIT, i_min=-cfg.POSITION_Z_INT_LIMIT)
        self.x_filter = AlphaFilter(cfg.POSITION_FILTER_ALPHA_POS)
        self.y_filter = AlphaFilter(cfg.POSITION_FILTER_ALPHA_POS)
        self.z_filter = AlphaFilter(cfg.POSITION_FILTER_ALPHA_POS)

        # 状态量
        self.rc_data = None
        self.path_data = Float64MultiArray()
        self.pose_data = None
        self.rc_reset = 0
        # [修正] 不再需要记录初始位置，因为我们直接使用绝对坐标
        # self.posx_zero = 0.0
        # self.posy_zero = 0.0
        # self.posz_zero = 0.0

        self.get_logger().info('📍位置控制节点启动完成 (已修正坐标系逻辑)')

        # 控制循环
        self.timer = self.create_timer(
            1.0 / cfg.POSITION_CONTROL_FREQ, self.do_control
        )

    def rc_callback(self, msg):
        self.rc_data = msg

    def path_callback(self, msg):
        self.path_data = msg

    def pose_callback(self, msg):
        self.pose_data = msg
        self.x_filter.update(msg.pose.position.x)
        self.y_filter.update(msg.pose.position.y)
        self.z_filter.update(msg.pose.position.z)

    def do_control(self):
        # 未收到遥控器数据，不处理
        if self.rc_data is None or self.pose_data is None:
            return

        if self.rc_data.left_switch == cfg.LOCK_SWITCH_VALUE:
            self.rc_reset = 0
            # [修正] 上锁时重置PID，防止下次解锁时积分饱和
            self.x_loop.clear()
            self.y_loop.clear()
            self.z_loop.clear()
            return

        # [修正] 删除了记录初始位置的逻辑块，因为不再需要
        # if self.rc_reset == 0:
        #     self.rc_reset = 1
        #     self.posx_zero = self.x_filter.filtered_
        #     self.posy_zero = self.y_filter.filtered_
        #     self.posz_zero = self.z_filter.filtered_
        #     self.get_logger().info("armed pos updated")
        #     return

        # 计算当前位置
        current_x = self.x_filter.filtered_ 
        current_y = self.y_filter.filtered_ 
        current_z = self.z_filter.filtered_ 
        
        # [修正] 删除了计算相对偏移(delta)的代码

        # 路径参考点
        path = self.path_data.data if len(self.path_data.data) >= 2 else [0.0, 0.0]
        
        # [修正] PID计算：使用当前绝对位置作为测量值，目标绝对位置作为设定值
        x_out = self.x_loop.calc(current_x, path[0])
        y_out = -self.y_loop.calc(current_y, path[1])  # 注意左手系
        z_out = self.z_loop.calc(current_z, self.height_sp)

        # 构建指令并发布
        cmd = Vector3()
        cmd.x = x_out
        cmd.y = y_out
        cmd.z = self.hover_th + z_out
        self.pos_cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 位置控制节点退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

