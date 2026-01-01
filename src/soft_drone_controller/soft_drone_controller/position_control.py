# position_controller.py (已修正)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float64MultiArray
from custom_msgs.msg import ReadDJIRC
import numpy as np
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
        
        # --- 修正 1: 参数读取 ---
        # 移除了旧的、基于PWM的悬停油门 self.hover_th
        self.x_kp = cfg.POSITION_XY_KP
        self.x_ki = cfg.POSITION_XY_KI
        self.x_kd = cfg.POSITION_XY_KD
        self.y_kp = cfg.POSITION_XY_KP
        self.y_ki = cfg.POSITION_XY_KI
        self.y_kd = cfg.POSITION_XY_KD
        self.z_kp = cfg.POSITION_Z_KP
        self.z_ki = cfg.POSITION_Z_KI
        self.z_kd = cfg.POSITION_Z_KD
        self.xv_kp = cfg.VELOCITY_XY_KP
        self.xv_ki = cfg.VELOCITY_XY_KI
        self.xv_kd = cfg.VELOCITY_XY_KD
        self.yv_kp = cfg.VELOCITY_XY_KP
        self.yv_ki = cfg.VELOCITY_XY_KI
        self.yv_kd = cfg.VELOCITY_XY_KD
        self.zv_kp = cfg.VELOCITY_Z_KP
        self.zv_ki = cfg.VELOCITY_Z_KI
        self.zv_kd = cfg.VELOCITY_Z_KD

        self.vx_sp_filter = AlphaFilter(alpha=0.5)
        self.vy_sp_filter = AlphaFilter(alpha=0.5)
        self.vz_sp_filter = AlphaFilter(alpha=0.4)

        self.height_sp = getattr(cfg, "POSITION_DEFAULT_HEIGHT", 0.5)
        self.hover_th_ratio = getattr(cfg, "HOVER_THROTTLE_RATIO", 0.5)

        # PID初始化
        self.x_loop = SimplePID(self.x_kp, self.x_ki, self.x_kd, i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.y_loop = SimplePID(self.y_kp, self.y_ki, self.y_kd, i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.z_loop = SimplePID(self.z_kp, self.z_ki, self.z_kd, i_max=cfg.POSITION_Z_INT_LIMIT, i_min=-cfg.POSITION_Z_INT_LIMIT)
        self.x_filter = AlphaFilter(cfg.POSITION_FILTER_ALPHA_POS)
        self.y_filter = AlphaFilter(cfg.POSITION_FILTER_ALPHA_POS)
        self.z_filter = AlphaFilter(cfg.POSITION_FILTER_ALPHA_POS)
        self.x_vel_loop = SimplePID(self.xv_kp, self.xv_ki, self.xv_kd, i_max=cfg.VELOCITY_XY_INT_LIMIT, i_min=-cfg.VELOCITY_XY_INT_LIMIT)
        self.y_vel_loop = SimplePID(self.yv_kp, self.yv_ki, self.yv_kd, i_max=cfg.VELOCITY_XY_INT_LIMIT, i_min=-cfg.VELOCITY_XY_INT_LIMIT)
        self.z_vel_loop = SimplePID(self.zv_kp, self.zv_ki, self.zv_kd, i_max=cfg.VELOCITY_Z_INT_LIMIT, i_min=-cfg.VELOCITY_Z_INT_LIMIT)
        # 状态量
        self.rc_data = None
        self.path_data = Float64MultiArray()
        self.pose_data = None
        self.rc_reset = 0
        self.last_x = None
        self.last_y = None
        self.last_z = None
        self.last_time = None

        # --- 修正 2: 更新日志信息 ---
        self.get_logger().info(f'📍位置控制节点启动完成 (悬停油门比例: {self.hover_th_ratio:.3f})')

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
        if self.rc_data is None or self.pose_data is None:
            return

        # --- 修正 3: 修复拼写错误 LOCK_SWITCH_VALUE ---
        if self.rc_data.left_switch == cfg.LOCK_SWITCH_VALUE:
            self.rc_reset = 0
            self.x_loop.clear()
            self.y_loop.clear()
            self.z_loop.clear()
            self.x_vel_loop.clear()
            self.y_vel_loop.clear()
            self.z_vel_loop.clear()
            return

        current_x = self.x_filter.filtered_ 
        current_y = self.y_filter.filtered_ 
        current_z = self.z_filter.filtered_ 

        now = time.time()
        dt = now - self.last_time if self.last_time else 0.02

        vx_now = ((current_x - self.last_x) / dt) if self.last_x is not None else 0.0
        vy_now = ((current_y - self.last_y) / dt) if self.last_y is not None else 0.0
        vz_now = ((current_z - self.last_z) / dt) if self.last_z is not None else 0.0   
        self.last_x = current_x
        self.last_y = current_y
        self.last_z = current_z
        self.last_time = now
        
        path = self.path_data.data if len(self.path_data.data) >= 2 else [0.0, 0.0]
        target_x = path[0]
        target_y = path[1]
        target_z = self.height_sp
        vx_sp = self.x_loop.calc(current_x, target_x)
        vy_sp = self.y_loop.calc(current_y, target_y)
        vz_sp = self.z_loop.calc(current_z, target_z)

        self.vx_sp_filter.update(vx_sp)
        self.vy_sp_filter.update(vy_sp)
        self.vz_sp_filter.update(vz_sp)

        vx_sp_filtered = self.vx_sp_filter.filtered_
        vy_sp_filtered = self.vy_sp_filter.filtered_
        vz_sp_filtered = self.vz_sp_filter.filtered_

        pitch_cmd = self.x_vel_loop.calc(vx_now, vx_sp_filtered)
        roll_cmd = self.y_vel_loop.calc(vy_now, vy_sp_filtered)
        #x_out = self.x_loop.calc(current_x, path[0])
        #y_out = -self.y_loop.calc(current_y, path[1])
        
        # 计算Z轴的油门比例增量
        z_out_ratio_increment = self.z_loop.calc(current_z, self.height_sp)

        # --- 修正 4: 正确计算并限幅最终的归一化油门指令 ---
        # 最终油门比例 = 悬停比例 + PID计算出的比例增量
        final_throttle_ratio = self.hover_th_ratio + z_out_ratio_increment
        
        # 将最终比例限制在 [0, 1] 范围内
        final_throttle_ratio = np.clip(final_throttle_ratio, cfg.MIN_DESCEND_THROTTLE_RATIO, 1.0)
        final_throttle_pwm = 1000.0 + final_throttle_ratio * 1000.0

        # 构建指令并发布 (现在发布的是0-1之间的浮点数)
        cmd = Vector3()
        cmd.x = roll_cmd
        cmd.y = pitch_cmd
        cmd.z = final_throttle_pwm
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
