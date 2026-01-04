# position_controller.py  (世界系->机体系 后再做速度环，集成方案A：解锁锁高度 + 平滑爬升)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float64MultiArray
from custom_msgs.msg import ReadDJIRC
import numpy as np
import time
from soft_drone_controller.config import controller_params as cfg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def deg2rad(d):
    return float(d) * np.pi / 180.0


class AlphaFilter:
    def __init__(self, alpha=0.6, init=0.0):
        self.alpha = float(alpha)
        self.filtered_ = float(init)
        self.inited = False

    def reset(self, value=0.0):
        self.filtered_ = float(value)
        self.inited = False

    def update(self, raw):
        raw = float(raw)
        if not self.inited:
            self.filtered_ = raw
            self.inited = True
        else:
            self.filtered_ = self.alpha * raw + (1.0 - self.alpha) * self.filtered_
        return self.filtered_


class FirstOrderLPF:
    """一阶低通：alpha = dt/(tau+dt)"""
    def __init__(self, tau=0.08, init=0.0):
        self.tau = float(tau)
        self.y = float(init)
        self.inited = False

    def reset(self, value=0.0):
        self.y = float(value)
        self.inited = False

    def update(self, x, dt):
        x = float(x)
        dt = float(max(dt, 1e-4))
        if not self.inited:
            self.y = x
            self.inited = True
            return self.y
        a = dt / (self.tau + dt)
        self.y = self.y + a * (x - self.y)
        return self.y


class PID:
    """带 dt 的简单 PID（建议速度环先不加D）"""
    def __init__(self, kp, ki, kd, i_max=1.0, i_min=-1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_max = float(i_max)
        self.i_min = float(i_min)
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, measurement, setpoint, dt):
        dt = float(np.clip(dt if dt is not None else 0.01, 1e-4, 0.1))
        error = float(setpoint - measurement)

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, self.i_min, self.i_max))

        if self.prev_error is None:
            d_error = 0.0
        else:
            d_error = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * d_error


class PositionController(Node):
    def __init__(self):
        super().__init__('position_control_node')

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

        # ===== 订阅 =====
        self.rc_sub = self.create_subscription(
            ReadDJIRC, '/ecat/sn2228293/app1/read', self.rc_callback, qos_best_effort
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/Tracker0/pose', self.pose_callback, qos_best_effort
        )
        self.path_sub = self.create_subscription(
            Float64MultiArray, '/pos_path', self.path_callback, qos_best_effort
        )
        # yaw（来自 drone_controller 的 /imu_angle，msg.z=deg）
        self.imu_angle_sub = self.create_subscription(
            Vector3, '/imu_angle', self.imu_angle_callback, qos_best_effort
        )

        # ===== 发布 =====
        self.pos_cmd_pub = self.create_publisher(
            Vector3, '/attitude_position_cmd', qos_reliable
        )

        # ===== 参数 =====
        self.default_height = float(getattr(cfg, "POSITION_DEFAULT_HEIGHT", 0.7))
        self.hover_th_ratio = float(getattr(cfg, "HOVER_THROTTLE_RATIO", 0.5))

        # 方案A：解锁先锁高度，然后再平滑爬升到 default_height
        self.takeoff_active = True  # 这里默认自动起飞（想手动触发就改为 False 并加开关逻辑）
        self.height_lock_on_arm = True
        self.height_sp = None              # 解锁时会设置为当前高度
        self.height_target = self.default_height
        self.max_climb_rate = float(getattr(cfg, "TAKEOFF_CLIMB_RATE", 0.15))  # m/s（没配置就 0.25）

        # deadzone
        self.pos_dead_xy = float(getattr(cfg, "POSITION_DEADZONE_XY", 0.02))

        # 限速
        self.vxy_limit = float(getattr(cfg, "POSITION_VXY_LIMIT", 0.8))
        self.vz_limit = float(getattr(cfg, "POSITION_VZ_LIMIT", 0.6))

        # ===== PID =====
        # 位置环（输出世界系速度 m/s）
        self.x_loop = PID(cfg.POSITION_XY_KP, cfg.POSITION_XY_KI, 0.0,
                          i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.y_loop = PID(cfg.POSITION_XY_KP, cfg.POSITION_XY_KI, 0.0,
                          i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.z_loop = PID(cfg.POSITION_Z_KP, cfg.POSITION_Z_KI, 0.0,
                          i_max=cfg.POSITION_Z_INT_LIMIT, i_min=-cfg.POSITION_Z_INT_LIMIT)

        # 速度环（输入机体系速度，输出角度 rad）
        self.x_vel_loop = PID(cfg.VELOCITY_XY_KP, 0.0, 0.0,
                              i_max=cfg.VELOCITY_XY_INT_LIMIT, i_min=-cfg.VELOCITY_XY_INT_LIMIT)
        self.y_vel_loop = PID(cfg.VELOCITY_XY_KP, 0.0, 0.0,
                              i_max=cfg.VELOCITY_XY_INT_LIMIT, i_min=-cfg.VELOCITY_XY_INT_LIMIT)

        # ===== 滤波 =====
        self.x_f = AlphaFilter(alpha=cfg.POSITION_FILTER_ALPHA_POS, init=0.0)
        self.y_f = AlphaFilter(alpha=cfg.POSITION_FILTER_ALPHA_POS, init=0.0)
        self.z_f = AlphaFilter(alpha=cfg.POSITION_FILTER_ALPHA_POS, init=0.0)

        # 世界系速度估计低通
        self.vx_lpf = FirstOrderLPF(tau=0.08, init=0.0)
        self.vy_lpf = FirstOrderLPF(tau=0.08, init=0.0)
        self.vz_lpf = FirstOrderLPF(tau=0.12, init=0.0)

        # 期望速度低通
        self.vx_sp_lpf = FirstOrderLPF(tau=0.06, init=0.0)
        self.vy_sp_lpf = FirstOrderLPF(tau=0.06, init=0.0)
        self.vz_sp_lpf = FirstOrderLPF(tau=0.10, init=0.0)

        # ===== 角度限制 =====
        self.max_angle = float(getattr(cfg, "POSITION_XY_MAX_ANGLE", 0.35))  # rad
        self.max_angle_rate = float(getattr(cfg, "POSITION_XY_MAX_ANGLE_RATE", 3.0))  # rad/s
        self.last_roll_cmd = 0.0
        self.last_pitch_cmd = 0.0

        # ===== 状态 =====
        self.rc_data = None
        self.pose_data = None
        self.path_data = Float64MultiArray()

        self.last_pose_time = None
        self.last_pose_x = None
        self.last_pose_y = None
        self.last_pose_z = None

        self.vx_est_w = 0.0
        self.vy_est_w = 0.0
        self.vz_est = 0.0

        self.yaw_rad = 0.0
        self.last_yaw_time = 0.0

        self.last_ctrl_time = time.time()

        # 用于检测“解锁边沿”
        self.prev_locked = True

        self.get_logger().info(
            f"📍位置控制启动：世界系速度->机体系速度 + 方案A(解锁锁高再爬升) | "
            f"default_height={self.default_height:.2f}m, climb_rate={self.max_climb_rate:.2f}m/s"
        )

        self.timer = self.create_timer(1.0 / cfg.POSITION_CONTROL_FREQ, self.do_control)

    # ===== callbacks =====
    def rc_callback(self, msg):
        self.rc_data = msg

    def path_callback(self, msg):
        self.path_data = msg

    def imu_angle_callback(self, msg: Vector3):
        self.yaw_rad = deg2rad(msg.z)
        self.last_yaw_time = time.time()

    def pose_callback(self, msg: PoseStamped):
        """动捕到来时：更新滤波位置 + 更新世界系速度估计"""
        self.pose_data = msg

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        xf = self.x_f.update(x)
        yf = self.y_f.update(y)
        zf = self.z_f.update(z)

        now = time.time()
        if self.last_pose_time is None:
            self.last_pose_time = now
            self.last_pose_x = xf
            self.last_pose_y = yf
            self.last_pose_z = zf
            self.vx_lpf.reset(0.0)
            self.vy_lpf.reset(0.0)
            self.vz_lpf.reset(0.0)
            self.vx_est_w = self.vy_est_w = self.vz_est = 0.0
            return

        dt = float(np.clip(now - self.last_pose_time, 1e-3, 0.05))

        vx_raw = (xf - self.last_pose_x) / dt
        vy_raw = (yf - self.last_pose_y) / dt
        vz_raw = (zf - self.last_pose_z) / dt

        self.vx_est_w = self.vx_lpf.update(vx_raw, dt)
        self.vy_est_w = self.vy_lpf.update(vy_raw, dt)
        self.vz_est   = self.vz_lpf.update(vz_raw, dt)

        self.last_pose_x = xf
        self.last_pose_y = yf
        self.last_pose_z = zf
        self.last_pose_time = now

    # ===== helpers =====
    def _slew(self, target, last, dt, max_rate):
        dt = float(max(dt, 1e-4))
        max_step = float(max_rate) * dt
        return float(np.clip(target, last - max_step, last + max_step))

    def _world_to_body_2d(self, vx_w, vy_w, yaw_rad):
        """
        世界系XY速度 -> 机体系XY速度（绕Z轴旋转 -yaw）
        body_x 前、body_y 右（如果你机体系定义不同，需要改符号）
        """
        cy = np.cos(yaw_rad)
        sy = np.sin(yaw_rad)
        vx_b =  cy * vx_w + sy * vy_w
        vy_b = -sy * vx_w + cy * vy_w
        return float(vx_b), float(vy_b)

    def _update_height_sp_planA(self, current_z, dt_ctrl):
        """
        方案A：
        - 解锁时：height_sp = current_z（锁高）
        - takeoff_active=True 时：height_sp 按 max_climb_rate 平滑逼近 height_target(default_height)
        """
        if self.height_sp is None:
            self.height_sp = float(current_z)

        if not self.takeoff_active:
            return

        target = float(self.height_target)
        step = float(self.max_climb_rate) * float(dt_ctrl)
        # 只允许每次增加/减少 step，避免一下跳到 0.7
        if self.height_sp < target:
            self.height_sp = min(self.height_sp + step, target)
        else:
            self.height_sp = max(self.height_sp - step, target)

    # ===== main loop =====
    def do_control(self):
        if self.rc_data is None or self.pose_data is None:
            return

        locked = (self.rc_data.left_switch == cfg.LOCK_SWITCH_VALUE)

        # 上锁：清状态
        if locked:
            self.prev_locked = True
            self.height_sp = None
            self.x_loop.reset(); self.y_loop.reset(); self.z_loop.reset()
            self.x_vel_loop.reset(); self.y_vel_loop.reset()
            self.vx_sp_lpf.reset(0.0); self.vy_sp_lpf.reset(0.0); self.vz_sp_lpf.reset(0.0)
            self.last_roll_cmd = 0.0
            self.last_pitch_cmd = 0.0
            return

        # 解锁边沿：第一次从锁定到解锁
        if self.prev_locked and not locked:
            # 方案A关键：解锁瞬间锁高度
            current_z = float(self.z_f.filtered_)
            self.height_sp = float(current_z)
            # 可选：把目标高度设为 default_height（后续按爬升速度渐进）
            self.height_target = float(self.default_height)
            # 重置Z积分避免解锁瞬间积累
            self.z_loop.reset()
            self.get_logger().info(f"🟢 解锁：锁定当前高度 height_sp={self.height_sp:.2f}m，然后平滑爬升到 {self.height_target:.2f}m")
            self.prev_locked = False

        now = time.time()
        dt_ctrl = float(np.clip(now - self.last_ctrl_time, 1e-3, 0.02))
        self.last_ctrl_time = now
        self.prev_locked = False

        # 当前滤波位置（世界系）
        current_x = float(self.x_f.filtered_)
        current_y = float(self.y_f.filtered_)
        current_z = float(self.z_f.filtered_)

        # 方案A：更新高度 setpoint（锁高 + 平滑爬升）
        self._update_height_sp_planA(current_z, dt_ctrl)

        # 当前速度（世界系）
        vx_now_w = float(self.vx_est_w)
        vy_now_w = float(self.vy_est_w)
        vz_now   = float(self.vz_est)

        # 目标（世界系XY来自pos_path，Z来自height_sp）
        path = self.path_data.data if len(self.path_data.data) >= 2 else [0.0, 0.0]
        target_x = float(path[0])
        target_y = float(path[1])
        target_z = float(self.height_sp if self.height_sp is not None else current_z)

        # ===== 位置deadzone真正生效：在deadzone内让 setpoint=measurement =====
        ex = target_x - current_x
        ey = target_y - current_y
        if abs(ex) < self.pos_dead_xy:
            target_x_eff = current_x
        else:
            target_x_eff = target_x
        if abs(ey) < self.pos_dead_xy:
            target_y_eff = current_y
        else:
            target_y_eff = target_y

        # ===== 位置环：输出世界系期望速度 =====
        vx_sp_w = self.x_loop.step(current_x, target_x_eff, dt_ctrl)
        vy_sp_w = self.y_loop.step(current_y, target_y_eff, dt_ctrl)
        vz_sp   = self.z_loop.step(current_z, target_z, dt_ctrl)

        # 限速
        vx_sp_w = float(np.clip(vx_sp_w, -self.vxy_limit, self.vxy_limit))
        vy_sp_w = float(np.clip(vy_sp_w, -self.vxy_limit, self.vxy_limit))
        vz_sp   = float(np.clip(vz_sp,   -self.vz_limit,  self.vz_limit))

        # 期望速度低通
        vx_sp_w_f = self.vx_sp_lpf.update(vx_sp_w, dt_ctrl)
        vy_sp_w_f = self.vy_sp_lpf.update(vy_sp_w, dt_ctrl)
        vz_sp_f   = self.vz_sp_lpf.update(vz_sp,   dt_ctrl)

        # ===== 核心：世界系速度 -> 机体系速度（用yaw） =====
        yaw = float(self.yaw_rad)
        vx_sp_b,  vy_sp_b  = self._world_to_body_2d(vx_sp_w_f, vy_sp_w_f, yaw)
        vx_now_b, vy_now_b = self._world_to_body_2d(vx_now_w,  vy_now_w,  yaw)

        # ===== 速度环：机体系速度 -> roll/pitch =====
        pitch_cmd = self.x_vel_loop.step(vx_now_b, vx_sp_b, dt_ctrl)
        roll_cmd  = self.y_vel_loop.step(vy_now_b, vy_sp_b, dt_ctrl)

        # 限幅（rad）
        roll_cmd  = float(np.clip(roll_cmd,  -self.max_angle, self.max_angle))
        pitch_cmd = float(np.clip(pitch_cmd, -self.max_angle, self.max_angle))

        # 斜率限制
        roll_cmd  = self._slew(roll_cmd,  self.last_roll_cmd,  dt_ctrl, self.max_angle_rate)
        pitch_cmd = self._slew(pitch_cmd, self.last_pitch_cmd, dt_ctrl, self.max_angle_rate)
        self.last_roll_cmd = roll_cmd
        self.last_pitch_cmd = pitch_cmd

        # ===== Z油门（比例增量 -> PWM）=====
        z_out_ratio_increment = float(np.clip(vz_sp_f, -0.08, 0.08))
        final_throttle_ratio = float(self.hover_th_ratio + z_out_ratio_increment)
        final_throttle_ratio = float(np.clip(final_throttle_ratio, cfg.MIN_DESCEND_THROTTLE_RATIO, 1.0))
        final_throttle_pwm = 1000.0 + final_throttle_ratio * 1000.0

        # 发布给 drone_controller（roll/pitch 单位：rad）
        cmd = Vector3()
        cmd.x = float(roll_cmd)
        cmd.y = float(pitch_cmd)
        cmd.z = float(final_throttle_pwm)
        self.pos_cmd_pub.publish(cmd)

        # 打印调试（约1Hz）
        if int(now * 10) % 10 == 0:
            self.get_logger().info(
                f"[POS] yaw={np.rad2deg(yaw):.1f}deg | "
                f"ex={ex:.2f}, ey={ey:.2f} | "
                f"z_sp={target_z:.2f}, z={current_z:.2f} | "
                f"vx_w={vx_now_w:.2f}, vy_w={vy_now_w:.2f} -> "
                f"vx_b={vx_now_b:.2f}, vy_b={vy_now_b:.2f} | "
                f"vx_sp_w={vx_sp_w_f:.2f}, vy_sp_w={vy_sp_w_f:.2f} -> "
                f"vx_sp_b={vx_sp_b:.2f}, vy_sp_b={vy_sp_b:.2f} | "
                f"roll={np.rad2deg(roll_cmd):.2f}deg, pitch={np.rad2deg(pitch_cmd):.2f}deg | "
                f"thr={final_throttle_pwm:.0f}"
            )


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


"""
无人机飞控主控制器 - 四元数版本 + 位置控制模式（带详细调试信息）（基于四元数误差的姿态控制改版）
"""

import rclpy
import numpy as np
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from custom_msgs.msg import ReadDJIRC, WriteDSHOT
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import sys
from soft_drone_controller.config import controller_params as cfg

# ====================== 四元数工具函数及新控制函数 ======================
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def eul2quat(roll, pitch, yaw):
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def eul2quat_matlab(eul):
    # eul: [pitch, roll, yaw] 顺序与MATLAB对应
    pitch, roll, yaw = eul
    return eul2quat(roll, pitch, yaw)

def quat2eul(w, x, y, z):
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

def quaternion_multiply(q1, q2): # (MATLAB形式)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def calculateErrorQuaternion(q_cmd, q_meas):
    q_conj = np.array([q_meas[0], -q_meas[1], -q_meas[2], -q_meas[3]])
    q_e = np.zeros(4)
    q_e[0] = q_conj[0] * q_cmd[0] - q_conj[1] * q_cmd[1] - q_conj[2] * q_cmd[2] - q_conj[3] * q_cmd[3]
    q_e[1] = q_conj[0] * q_cmd[1] + q_conj[1] * q_cmd[0] + q_conj[2] * q_cmd[3] - q_conj[3] * q_cmd[2]
    q_e[2] = q_conj[0] * q_cmd[2] - q_conj[1] * q_cmd[3] + q_conj[2] * q_cmd[0] + q_conj[3] * q_cmd[1]
    q_e[3] = q_conj[0] * q_cmd[3] + q_conj[1] * q_cmd[2] - q_conj[2] * q_cmd[1] + q_conj[3] * q_cmd[0]
    return q_e

def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro])

def pwm_to_dshot(pwm_val):
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))


# ===================== PID控制器类 =====================
class ImprovedPID:
    def __init__(self, kp, ki, kd, i_max=0.5, i_min=-0.5, use_angular_acc=True, node=None, axis=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_max = i_max
        self.i_min = i_min
        self.use_angular_acc = use_angular_acc
        self.node = node
        self.axis = axis
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0
        self.d_term_sign = 1.0
        if axis in ["roll_rate", "pitch_rate", "yaw_rate"]:
            self.d_term_sign = -1.0
        self.pub_pid_error = None
        if self.node is not None:
            self.pub_pid_error = self.node.create_publisher(
                Vector3, f"/pid_error_{self.axis}",
                QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
            )
    def update(self, setpoint, measurement, dt, angular_acc=None):
        if dt <= 0:
            dt = 1.0 / cfg.CONTROL_FREQ
        measurement_rate = 0.0
        if dt > 0:
            measurement_rate = (measurement-self.prev_measurement) / dt
        error = setpoint - measurement 
        error_threhold = 0.009
        if self.axis in ["roll","pitch"]:
            error_threhold = 0.015
        elif self.axis == "yaw":
            error_threhold = 0.02
        if abs(error) < error_threhold:
            self.integral = 0.0
            self.last_output = 0.0
            return 0.0
        if self.axis == "yaw" and abs(error) < 0.005:
            error = 0.0
            self.integral = 0.0
        if self.pub_pid_error is not None:
            error_msg = Vector3()
            error_msg.x = error
            error_msg.y = self.integral
            error_msg.z = self.last_output
            self.pub_pid_error.publish(error_msg)
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.i_min, self.i_max)
        i_term = self.ki * self.integral
        if self.use_angular_acc and angular_acc is not None:
            d_term = self.kd * angular_acc * self.d_term_sign
        else:
            d_term = self.kd * measurement_rate * self.d_term_sign
        output = p_term + i_term + d_term
        self.prev_error = error
        self.prev_measurement = measurement
        self.last_output = output
        return output
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0

# ===================== 主控制器 =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")
        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        self.get_logger().info("✅ 飞控启动完成 - 支持手动/位置控制模式")
        self.get_logger().info("🎮 控制模式 (双开关):")
        self.get_logger().info("   - 左开关: 上/中 = 解锁, 下 = 上锁")
        self.get_logger().info("   - 右开关: 上/中 = 位置控制, 下 = 手动模式")

    def _init_ros(self):
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST, 
            depth=5
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, 
            depth=10
        )
        self.pub_imu_angle = self.create_publisher(Vector3, "/imu_angle", qos_reliable)
        self.pub_imu_gyro = self.create_publisher(Vector3, "/imu_gyro", qos_reliable)
        self.pub_dshot = self.create_publisher(WriteDSHOT, "/ecat/sn2228293/app3/write", 10)
        self.pub_torque_output = self.create_publisher(Vector3, "/torque_output", qos_reliable)
        self.pub_control_status = self.create_publisher(Vector3, "/fc_control_status", qos_reliable)
        self.pub_attitude_debug = self.create_publisher(Vector3, "/attitude_debug", qos_reliable)
        self.pub_attitude_error = self.create_publisher(Vector3, "/attitude_error", qos_reliable)
        self.pub_control_mode_info = self.create_publisher(Vector3, "/control_mode_info", qos_reliable)
        self.pub_position_cmd_status = self.create_publisher(Vector3, "/position_cmd_status", qos_reliable)
        self.pub_control_details = self.create_publisher(Vector3, "/control_details", qos_reliable)
        self.sub_rc = self.create_subscription(
            ReadDJIRC, 
            "/ecat/sn2228293/app1/read", 
            self._rc_callback, 
            qos_best_effort
        )
        self.sub_imu = self.create_subscription(
            Imu, 
            "/ecat/sn2228293/app2/read", 
            self._imu_callback, 
            qos_best_effort
        )
        self.sub_filtered_acc = self.create_subscription(
            Vector3, 
            "/filtered_angular_acceleration", 
            self._filtered_acc_callback, 
            qos_reliable
        )
        self.sub_pos_cmd = self.create_subscription(
            Vector3,
            "/attitude_position_cmd",
            self._pos_cmd_callback,
            qos_reliable
        )
        self.lock = threading.Lock()

    def _init_data(self):
        self.rc_data = {
            "left_y": 0.0,
            "left_x": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
            "left_switch": cfg.LOCK_SWITCH_VALUE,
            "right_switch": 2
        }
        self.imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "gyro": np.array([0.0, 0.0, 0.0]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0
        }
        self.filtered_acc = np.array([0.0, 0.0, 0.0])
        self.pos_cmd_data = {
            "roll": 0.0,
            "pitch": 0.0,
            "throttle": 1000.0
        }
        self.last_pos_cmd_time = 0.0
        self.pos_cmd_timeout = 0.2
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0
        self.last_published_dshot = [1200]*4
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_yaw_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.control_mode = "MANUAL"
        self.control_debug = {
            "target_roll": 0.0,
            "target_pitch": 0.0,
            "target_yaw": 0.0,
            "position_cmd_valid": False
        }

    def _init_controllers(self):
        self.pid_roll_angle = ImprovedPID(
            kp=cfg.PID_ROLL_ANGLE["kp"] * 3.0,
            ki=cfg.PID_ROLL_ANGLE["ki"] * 0,
            kd=cfg.PID_ROLL_ANGLE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="roll"
        )
        self.pid_roll_rate = ImprovedPID(
            kp=cfg.PID_ROLL_RATE["kp"] * 3.0,
            ki=cfg.PID_ROLL_RATE["ki"] * 0,
            kd=cfg.PID_ROLL_RATE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="roll_rate"
        )
        self.pid_pitch_angle = ImprovedPID(
            kp=cfg.PID_PITCH_ANGLE["kp"] * 3.0,
            ki=cfg.PID_PITCH_ANGLE["ki"] * 0,
            kd=cfg.PID_PITCH_ANGLE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="pitch"
        )
        self.pid_pitch_rate = ImprovedPID(
            kp=cfg.PID_PITCH_RATE["kp"] * 3.0,
            ki=cfg.PID_PITCH_RATE["ki"] * 0,
            kd=cfg.PID_PITCH_RATE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="pitch_rate"
        )
        self.pid_yaw_angle = ImprovedPID(
            kp=cfg.PID_YAW_ANGLE["kp"] * 7.0,
            ki=cfg.PID_YAW_ANGLE["ki"] * 0.0,
            kd=cfg.PID_YAW_ANGLE["kd"] * 0.4,
            i_max=0.05,
            i_min=-0.05,
            use_angular_acc=False,
            node=self,
            axis="yaw"
        )
        self.pid_yaw_rate = ImprovedPID(
            kp=cfg.PID_YAW_RATE["kp"] * 15.0,
            ki=cfg.PID_YAW_RATE["ki"] * 0.0,
            kd=cfg.PID_YAW_RATE["kd"] * 0.6,
            i_max=0.1,
            i_min=-0.1,
            use_angular_acc=False,
            node=self,
            axis="yaw_rate"
        )

    def _init_state(self):
        self.state = {
            "armed": False,
            #"stick_deadband": cfg.RC_DEAD_ZONE * 0.3,
            "motor_outputs": np.array([1000.0]*4),
            "init_quat": None,
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.8,
            "last_debug_time": 0.0,
            "last_control_debug_time": 0.0
        }
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        self.yaw_stick_scale = 0.2
        self.yaw_dshot_gain = 0.6

    # ========== 数据回调 ==========

    def _rc_callback(self, msg):
        with self.lock:
            self.rc_data["left_y"] = msg.left_y
            self.rc_data["left_x"] = msg.left_x
            self.rc_data["right_x"] = msg.right_x
            self.rc_data["right_y"] = -msg.right_y
            self.rc_data["left_switch"] = msg.left_switch
            self.rc_data["right_switch"] = msg.right_switch
            self.last_rc_time = self.get_clock().now().nanoseconds / 1e9
            self._update_arming_and_mode()

    def _imu_callback(self, msg):
        with self.lock:
            current_quat = np.array([
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z
            ])
            current_quat = self._correct_quat_sign(current_quat)
            if self.state["init_quat"] is None:
                self.state["init_quat"] = current_quat.copy()
                self.state["initialized"] = True
                self.last_yaw_quat = self._extract_yaw_quat(current_quat)
                self.target_quat = current_quat.copy()
                self.get_logger().info("📡 IMU初始化完成")
            rel_quat = quat_mult(current_quat, quat_inv(self.state["init_quat"]))
            roll_zeroed, pitch_zeroed, current_yaw = quat2eul(*rel_quat)
            current_yaw = current_yaw
            current_yaw_quat = self._extract_yaw_quat(current_quat)
            self.target_quat = self._set_quat_yaw(rel_quat, self.last_yaw_quat)
            self.last_yaw_quat = current_yaw_quat
            self.imu_data["quat"] = rel_quat
            self.imu_data["roll"] = roll_zeroed
            self.imu_data["pitch"] = pitch_zeroed
            self.imu_data["yaw"] = current_yaw
            gyro_rotated = rotate_gyro_data(
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            )
            self.imu_data["gyro"] = gyro_rotated
            if int(time.time() * 10) % 10 == 0:
                self.get_logger().info(
                    f"📡 IMU数据: Roll={np.rad2deg(roll_zeroed):.3f}°, "
                    f"Pitch={np.rad2deg(pitch_zeroed):.3f}°, "
                    f"Yaw={np.rad2deg(current_yaw):.3f}°"
                )
            angle_msg = Vector3()
            angle_msg.x = np.rad2deg(roll_zeroed)
            angle_msg.y = np.rad2deg(pitch_zeroed)
            angle_msg.z = np.rad2deg(current_yaw)
            self.pub_imu_angle.publish(angle_msg)
            gyro_msg = Vector3()
            gyro_msg.x = gyro_rotated[0]
            gyro_msg.y = gyro_rotated[1]
            gyro_msg.z = gyro_rotated[2]
            self.pub_imu_gyro.publish(gyro_msg)
            self.last_imu_time = self.get_clock().now().nanoseconds / 1e9

    def _filtered_acc_callback(self, msg):
        with self.lock:
            self.filtered_acc = np.array([msg.x, msg.y, msg.z])

    def _pos_cmd_callback(self, msg):
        with self.lock:
            self.pos_cmd_data["roll"] = msg.x
            self.pos_cmd_data["pitch"] = msg.y
            self.pos_cmd_data["throttle"] = msg.z
            self.last_pos_cmd_time = self.get_clock().now().nanoseconds / 1e9
            self.control_debug["target_roll"] = msg.x
            self.control_debug["target_pitch"] = msg.y

    def _correct_quat_sign(self, quat):
        w, x, y, z = quat
        y = -y
        z = -z
        return np.array([w, x, y, z])
    def _extract_yaw_quat(self, quat):
        roll, pitch, yaw = quat2eul(*quat)
        return eul2quat(0, 0, yaw)
    def _set_quat_yaw(self, quat, yaw_quat):
        roll, pitch, _ = quat2eul(*quat)
        _, _, yaw = quat2eul(*yaw_quat)
        return eul2quat(roll, pitch, yaw)
    def _update_arming_and_mode(self):
        left_switch = self.rc_data["left_switch"]
        right_switch = self.rc_data["right_switch"]
        if left_switch == 2:
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                self.get_logger().info("🔒 已锁定 (左开关拨至底部)")
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                self._reset_all_controllers()
            return
        if left_switch in {1, 3} and not self.state["armed"]:
            if not self.state["initialized"]:
                self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                return
            self.state["armed"] = True
            if right_switch in {1, 3}:
                self.control_mode = "POSITION"
                self.get_logger().info("🔓 已解锁 - 位置控制模式 (右开关在上/中)")
            else:
                self.control_mode = "MANUAL"
                self.get_logger().info("🔓 已解锁 - 手动模式 (右开关在底部)")
            self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
            self._reset_all_controllers()
            return
        if not self.state["armed"]:
            return
        if right_switch in {1, 3} and self.control_mode != "POSITION":
            self.control_mode = "POSITION"
            self.get_logger().info("🎯 切换到位置控制模式 (右开关在上/中)")
            self._reset_all_controllers()
        elif right_switch == 2 and self.control_mode != "MANUAL":
            self.control_mode = "MANUAL"
            self.get_logger().info("✈️ 切换到手动模式 (右开关在底部)")
            self._reset_all_controllers()
    def _reset_all_controllers(self):
        for pid in [self.pid_roll_angle, self.pid_pitch_angle, self.pid_yaw_angle,
                    self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        self.get_logger().info("🔄 所有PID控制器已重置")

    # ========== 核心控制循环 ==========
    def _control_loop(self):
        with self.lock:
            if not self._check_data_validity():
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
            if not self.state["armed"]:
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
            dt = 1.0 / cfg.CONTROL_FREQ
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.control_mode == "MANUAL":
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            elif self.control_mode == "POSITION":
                pos_cmd_timeout = (current_time - self.last_pos_cmd_time) > self.pos_cmd_timeout
                if pos_cmd_timeout:
                    self.control_mode = "MANUAL"
                    self.get_logger().warn("⚠️ 位置指令超时，自动切换回手动模式")
                    throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                else:
                    
                    throttle_raw = np.clip(self.pos_cmd_data["throttle"], 1000.0, 2000.0)
                    roll_target = np.clip(self.pos_cmd_data["roll"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    pitch_target = np.clip(self.pos_cmd_data["pitch"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    throttle = (throttle_raw - 1000.0) / 1000.0 * 1000.0
                    #throttle, _, _, yaw_stick = self._process_stick()

                    _, _, _, yaw_stick = self._process_stick()
                    self.control_debug["position_cmd_valid"] = True
                    if int(current_time * 100) % 100 == 0:
                        self.get_logger().info(
                            f"🎯 位置控制指令: "
                            f"Roll={np.rad2deg(roll_target):.1f}°, "
                            f"Pitch={np.rad2deg(pitch_target):.1f}°, "
                            f"Throttle={throttle_raw:.0f}"
                        )
            else:
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(
                roll_target, pitch_target, yaw_stick, dt
            )
            self._publish_control_debug(roll_target, pitch_target, dt)
            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
            self._publish_control_status()

    def _process_stick(self):
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > cfg.RC_DEAD_ZONE_ROLL else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > cfg.RC_DEAD_ZONE_PITCH else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > cfg.RC_DEAD_ZONE_YAW else 0.0
        throttle_raw = self.rc_data["left_y"] if abs(self.rc_data["left_y"]) > cfg.RC_DEAD_ZONE_THROTTLE else 0.0
        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        throttle = np.clip((throttle_raw + 1.0) / 2.0 * 1000.0, 0.0, 1000.0)
        return throttle, roll_target, pitch_target, yaw_raw

    # ========== 四元数误差姿态外环 ========
    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        roll_measured = self.imu_data["roll"]
        pitch_measured = self.imu_data["pitch"]
        yaw_measured = self.imu_data["yaw"]
        # 新增：yaw setpoint永远是上一帧的yaw（不变）
        if not hasattr(self, "last_yaw_target"):
            self.last_yaw_target = yaw_measured
        yaw_target = self.last_yaw_target
        quat_setpoint = eul2quat_matlab([pitch_target, roll_target, yaw_target])
        quat_measured = eul2quat_matlab([pitch_measured, roll_measured, yaw_measured])
        q_e = calculateErrorQuaternion(quat_setpoint, quat_measured)
        q_e0 = q_e[0]
        q_ej = q_e[1:4]
        A = 1.0 if q_e0 >= 0 else -1.0
        TIME_CONSTANT = 0.09
        Kp_angle = cfg.PID_ROLL_ANGLE["kp"] * 3.0
        B = q_ej * (2.0 / TIME_CONSTANT)
        ANGLE_ERROR = np.array([roll_target - roll_measured, pitch_target - pitch_measured, 0.0])
        Omega_sp = A * B * cfg.Kp_ANGLE
        omega_sp_roll, omega_sp_pitch, _ = Omega_sp
        # 新增：yaw速率环目标为摇杆输入，不用外环
        omega_sp_yaw_pid = yaw_rate_cmd
        self.last_yaw_target = yaw_measured  # 更新下次用
        gyro = self.imu_data["gyro"].copy()
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]
        torque_roll = self.pid_roll_rate.update(omega_sp_roll, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(omega_sp_pitch, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(omega_sp_yaw_pid, gyro[2], dt, self.filtered_acc[2])
        torque_roll = torque_roll
        torque_pitch = torque_pitch
        torque_yaw = torque_yaw
        torque_roll = np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_pitch = np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_yaw = np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"])
        self._publish_attitude_debug(roll_target, pitch_target, roll_measured, pitch_measured, omega_sp_roll, omega_sp_pitch, omega_sp_yaw_pid)
        torque_msg = Vector3()
        torque_msg.x = torque_roll
        torque_msg.y = torque_pitch
        torque_msg.z = torque_yaw
        self.pub_torque_output.publish(torque_msg)
        return torque_roll, torque_pitch, torque_yaw

    def _publish_attitude_debug(self, roll_target, pitch_target, roll_current, pitch_current, roll_err, pitch_err, yaw_err):
        attitude_debug = Vector3()
        attitude_debug.x = np.rad2deg(roll_target)
        attitude_debug.y = np.rad2deg(pitch_target)
        attitude_debug.z = np.rad2deg(roll_current)
        self.pub_attitude_debug.publish(attitude_debug)
        attitude_error = Vector3()
        attitude_error.x = np.rad2deg(roll_err)
        attitude_error.y = np.rad2deg(pitch_err)
        attitude_error.z = np.rad2deg(yaw_err)
        self.pub_attitude_error.publish(attitude_error)

    def _publish_control_debug(self, roll_target, pitch_target, dt):
        current_time = time.time()
        mode_info = Vector3()
        if self.control_mode == "MANUAL":
            mode_info.x = 1.0
        elif self.control_mode == "POSITION":
            mode_info.x = 2.0
        else:
            mode_info.x = 0.0
        cmd_timeout = time.time() - self.last_pos_cmd_time > self.pos_cmd_timeout
        mode_info.y = 1.0 if (self.control_mode == "POSITION" and not cmd_timeout) else 0.0
        motor_outputs = self.state["motor_outputs"]
        mode_info.z = np.max(motor_outputs) - np.min(motor_outputs)
        self.pub_control_mode_info.publish(mode_info)
        pos_cmd_status = Vector3()
        pos_cmd_status.x = self.pos_cmd_data["roll"]
        pos_cmd_status.y = self.pos_cmd_data["pitch"]
        pos_cmd_status.z = self.pos_cmd_data["throttle"]
        self.pub_position_cmd_status.publish(pos_cmd_status)
        if current_time - self.state["last_control_debug_time"] > 0.2:
            control_details = Vector3()
            control_details.x = np.rad2deg(self.imu_data["roll"])
            control_details.y = np.rad2deg(self.imu_data["pitch"])
            control_details.z = np.rad2deg(self.imu_data["yaw"])
            self.pub_control_details.publish(control_details)
            self.state["last_control_debug_time"] = current_time

    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        base = 1000.0 + throttle
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain
        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor_max = np.max([motor1, motor2, motor3, motor4])
        motor_min = np.min([motor1, motor2, motor3, motor4])
        if motor_max - motor_min > 600:
            scale = 600 / (motor_max - motor_min) if motor_max != motor_min else 1.0
            motor_avg = (motor1 + motor2 + motor3 + motor4) / 4
            motor1 = motor_avg + (motor1 - motor_avg) * scale
            motor2 = motor_avg + (motor2 - motor_avg) * scale
            motor3 = motor_avg + (motor3 - motor_avg) * scale
            motor4 = motor_avg + (motor4 - motor_avg) * scale
        alpha = 0.18
        smoothed_motors = alpha * np.array([motor1, motor2, motor3, motor4]) + (1 - alpha) * self.state["motor_outputs"]
        return smoothed_motors

    def _check_data_validity(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        rc_timeout = current_time - self.last_rc_time > cfg.DATA_TIMEOUT
        imu_timeout = current_time - self.last_imu_time > cfg.DATA_TIMEOUT
        if rc_timeout or imu_timeout:
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                error_msg = ""
                if rc_timeout:
                    error_msg += "遥控器数据超时"
                if imu_timeout:
                    error_msg += "IMU数据超时"
                self.get_logger().error(f"🔴 {error_msg}，强制上锁")
            return False
        return True

    def _publish_dshot(self, motor_pwm):
        msg = WriteDSHOT()
        if isinstance(motor_pwm, (int, float)):
            dshot_val = pwm_to_dshot(motor_pwm)
            msg.channel1 = dshot_val
            msg.channel2 = dshot_val
            msg.channel3 = dshot_val
            msg.channel4 = dshot_val
            self.last_published_dshot = [dshot_val]*4
        elif isinstance(motor_pwm, (list, np.ndarray)):
            if len(motor_pwm) >= 4:
                dshot = [pwm_to_dshot(p) for p in motor_pwm[:4]]
                msg.channel1 = dshot[2]
                msg.channel2 = dshot[0]
                msg.channel3 = dshot[1]
                msg.channel4 = dshot[3]
                self.last_published_dshot = dshot
            else:
                self.get_logger().error(f"❌ 电机PWM数组长度不足: {len(motor_pwm)}")
                return
        else:
            self.get_logger().error(f"❌ 不支持的motor_pwm类型: {type(motor_pwm)}")
            return
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ 发布DSHOT失败: {e}")

    def _publish_control_status(self):
        status_msg = Vector3()
        if self.control_mode == "MANUAL":
            status_msg.x = 1.0
        elif self.control_mode == "POSITION":
            status_msg.x = 2.0
        else:
            status_msg.x = 0.0
        status_msg.y = 1.0 if self.state["armed"] else 0.0
        status_msg.z = float(self.rc_data.get("left_switch", 0))
        self.pub_control_status.publish(status_msg)

    def _publish_status(self):
        if not self.state["armed"]:
            return
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch"]))
            if hasattr(self, 'yaw_setpoint'):
                yaw_error = abs(np.rad2deg(self.yaw_setpoint - self.imu_data["yaw"]))
            else:
                yaw_error = 0.0
            self.get_logger().info(
                f"📊 飞行状态 | "
                f"模式:{self.control_mode} | "
                f"Roll误差:{roll_error:.2f}° | "
                f"Pitch误差:{pitch_error:.2f}° | "
                f"Yaw误差:{yaw_error:.2f}° | "
                f"DSHOT:{dshot}"
            )
            self.state["last_debug_time"] = current_time

    def destroy_node(self):
        self.get_logger().info("🛑 正在关闭飞控...")
        try:
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
            time.sleep(0.05)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 销毁时发布上锁指令失败: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    try:
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info("🛑 用户中断，停止飞控")
    except Exception as e:
        controller.get_logger().error(f"❌ 飞控运行出错: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()







"""
无人机飞控主控制器 - 四元数版本 + 位置控制模式（带详细调试信息）（基于四元数误差的姿态控制改版）

✅ 保持 MANUAL 模式：仍然使用 init_quat 归零的相对姿态（不改你的手动逻辑）
✅ POSITION 模式：全姿态对齐世界系（以 MOCAP 为准），MOCAP 掉线时用 IMU + 对齐四元数维持连续世界姿态
✅ POSITION 模式支持 /yaw_hold_sp 外部航向目标（地面站/position_controller）
✅ yaw 测量优先 MOCAP，失效退回 IMU（aligned）
✅ yaw 误差 wrap 到 [-pi, pi]

本版本核心修复（针对“POSITION解锁后pitch想翻180°”）：
1) ✅ POSITION 模式下，roll/pitch 外环四元数 setpoint 的 yaw 必须用 yaw_target（/yaw_hold_sp 或 yaw_hold）
   - 之前用 yaw_measured 会导致 yaw 通道在转、roll/pitch误差参考也在变 -> 四元数等价解跳变 -> 可能出现180°翻转
2) ✅ 位置指令 pitch 不再在飞控里取反：pos_cmd_data["pitch"] = msg.y
   - 避免 position_controller 和 drone_controller 双重翻转造成“方向乱/改了没用”

可选开关：
- ENABLE_FRD_TO_FLU_FIX：是否对 mocap 的 Body(FRD)->Body(FLU) 进行180°修正
  如果你确认自己整条链路应当统一成 FRD，建议关闭（默认 False）。
"""

import rclpy
import numpy as np
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, PoseStamped
from std_msgs.msg import Float64
from custom_msgs.msg import ReadDJIRC, WriteDSHOT
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
from soft_drone_controller.config import controller_params as cfg


def wrap_pi(a):
    a = float(a)
    return float(np.arctan2(np.sin(a), np.cos(a)))


# ====================== Quaternion / Rotation Utilities ======================
def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def rotmat_from_quat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def quat_from_rotmat(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def quat_to_eul(q):
    # roll(x), pitch(y), yaw(z)
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return float(roll), float(pitch), float(yaw)


def eul2quat(roll, pitch, yaw):
    cr = np.cos(roll/2); sr = np.sin(roll/2)
    cp = np.cos(pitch/2); sp = np.sin(pitch/2)
    cy = np.cos(yaw/2); sy = np.sin(yaw/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_normalize(np.array([w, x, y, z]))


def eul2quat_matlab(eul):
    # 兼容你原来的 matlab 顺序：eul=[pitch, roll, yaw]
    pitch, roll, yaw = eul
    return eul2quat(roll, pitch, yaw)


def calculateErrorQuaternion(q_cmd, q_meas):
    q_conj = np.array([q_meas[0], -q_meas[1], -q_meas[2], -q_meas[3]])
    q_e = np.zeros(4)
    q_e[0] = q_conj[0] * q_cmd[0] - q_conj[1] * q_cmd[1] - q_conj[2] * q_cmd[2] - q_conj[3] * q_cmd[3]
    q_e[1] = q_conj[0] * q_cmd[1] + q_conj[1] * q_cmd[0] + q_conj[2] * q_cmd[3] - q_conj[3] * q_cmd[2]
    q_e[2] = q_conj[0] * q_cmd[2] - q_conj[1] * q_cmd[3] + q_conj[2] * q_cmd[0] + q_conj[3] * q_cmd[1]
    q_e[3] = q_conj[0] * q_cmd[3] + q_conj[1] * q_cmd[2] - q_conj[2] * q_cmd[1] + q_conj[3] * q_cmd[0]
    return q_e


def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    # 保持你的手动逻辑不变（你现在就是这么用的）
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro], dtype=float)


def pwm_to_dshot(pwm_val):
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))


# ===================== PID控制器类 =====================
class ImprovedPID:
    def __init__(self, kp, ki, kd, i_max=0.5, i_min=-0.5, use_angular_acc=True, node=None, axis=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_max = i_max
        self.i_min = i_min
        self.use_angular_acc = use_angular_acc
        self.node = node
        self.axis = axis
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0
        self.d_term_sign = 1.0
        if axis in ["roll_rate", "pitch_rate", "yaw_rate"]:
            self.d_term_sign = -1.0
        self.pub_pid_error = None
        if self.node is not None:
            self.pub_pid_error = self.node.create_publisher(
                Vector3, f"/pid_error_{self.axis}",
                QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
            )

    def update(self, setpoint, measurement, dt, angular_acc=None):
        if dt <= 0:
            dt = 1.0 / cfg.CONTROL_FREQ

        measurement_rate = 0.0
        if dt > 0:
            measurement_rate = (measurement - self.prev_measurement) / dt

        if self.axis == "yaw":
            error = wrap_pi(setpoint - measurement)
        else:
            error = setpoint - measurement

        if self.axis in ["roll", "pitch"]:
            error_threshold = 0.015
            if abs(error) < error_threshold:
                self.integral = 0.0
                self.last_output = 0.0
                return 0.0

        elif self.axis == "yaw":
            error_threshold = 0.003
            if abs(error) < error_threshold:
                error = 0.0

        elif self.axis in ["roll_rate", "pitch_rate", "yaw_rate"]:
            error_threshold = 0.009
            if abs(error) < error_threshold:
                self.integral = 0.0
                self.last_output = 0.0
                return 0.0

        if self.pub_pid_error is not None:
            error_msg = Vector3()
            error_msg.x = float(error)
            error_msg.y = float(self.integral)
            error_msg.z = float(self.last_output)
            self.pub_pid_error.publish(error_msg)

        p_term = self.kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, self.i_min, self.i_max)
        i_term = self.ki * self.integral

        if self.use_angular_acc and angular_acc is not None:
            d_term = self.kd * angular_acc * self.d_term_sign
        else:
            d_term = self.kd * measurement_rate * self.d_term_sign

        output = p_term + i_term + d_term
        self.prev_error = error
        self.prev_measurement = measurement
        self.last_output = output
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0


# ===================== 主控制器 =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")

        # ✅ 可选：如果你确认 mocap 姿态链路需要 FRD->FLU 才对，再改 True
        # 建议默认 False：因为你“机体坐标系 x前y右z下(FRD)”更应该统一 FRD
        self.ENABLE_FRD_TO_FLU_FIX = True

        # ✅ 你的定义：从上往下看，顺时针(CW)为正yaw
        # 标准数学/ROS ENU 通常是逆时针(CCW)为正。
        # 如果你希望“手动拿着机体顺时针转 -> yaw 增大”，就保持 True。
        self.YAW_POSITIVE_CW = True

        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)

        self.get_logger().info("✅ 飞控启动完成 - 支持手动/位置控制模式")
        self.get_logger().info("🎮 控制模式 (双开关):")
        self.get_logger().info("   - 左开关: 上/中 = 解锁, 下 = 上锁")
        self.get_logger().info("   - 右开关: 上/中 = 位置控制, 下 = 手动模式")
        self.get_logger().info("🧭 POSITION模式Yaw目标优先级: /yaw_hold_sp(外部) > yaw_hold(切入时锁定一次)")

    def _init_ros(self):
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )

        self.pub_imu_angle = self.create_publisher(Vector3, "/imu_angle", qos_reliable)
        self.pub_imu_gyro = self.create_publisher(Vector3, "/imu_gyro", qos_reliable)
        self.pub_dshot = self.create_publisher(WriteDSHOT, "/ecat/sn2228293/app3/write", 10)
        self.pub_torque_output = self.create_publisher(Vector3, "/torque_output", qos_reliable)
        self.pub_control_status = self.create_publisher(Vector3, "/fc_control_status", qos_reliable)
        self.pub_attitude_debug = self.create_publisher(Vector3, "/attitude_debug", qos_reliable)
        self.pub_attitude_error = self.create_publisher(Vector3, "/attitude_error", qos_reliable)
        self.pub_control_mode_info = self.create_publisher(Vector3, "/control_mode_info", qos_reliable)
        self.pub_position_cmd_status = self.create_publisher(Vector3, "/position_cmd_status", qos_reliable)
        self.pub_control_details = self.create_publisher(Vector3, "/control_details", qos_reliable)

        self.sub_rc = self.create_subscription(
            ReadDJIRC,
            "/ecat/sn2228293/app1/read",
            self._rc_callback,
            qos_best_effort
        )
        self.sub_imu = self.create_subscription(
            Imu,
            "/ecat/sn2228293/app2/read",
            self._imu_callback,
            qos_best_effort
        )
        self.sub_filtered_acc = self.create_subscription(
            Vector3,
            "/filtered_angular_acceleration",
            self._filtered_acc_callback,
            qos_reliable
        )
        self.sub_pos_cmd = self.create_subscription(
            Vector3,
            "/attitude_position_cmd",
            self._pos_cmd_callback,
            qos_reliable
        )
        self.sub_mocap_pose = self.create_subscription(
            PoseStamped,
            "/Tracker0/pose",
            self._mocap_pose_callback,
            qos_best_effort
        )
        self.sub_yaw_sp = self.create_subscription(
            Float64,
            "/yaw_hold_sp",
            self._yaw_sp_callback,
            qos_reliable
        )

        self.lock = threading.Lock()

    def _init_data(self):
        self.rc_data = {
            "left_y": 0.0,
            "left_x": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
            "left_switch": cfg.LOCK_SWITCH_VALUE,
            "right_switch": 2
        }

        # ===== IMU数据（保持你手动逻辑）=====
        self.imu_data = {
            "quat_rel": np.array([1.0, 0.0, 0.0, 0.0]),   # rel_quat: current * inv(init)
            "quat_abs": np.array([1.0, 0.0, 0.0, 0.0]),   # current_quat(修正后)
            "gyro": np.array([0.0, 0.0, 0.0]),
            "roll_rel": 0.0,
            "pitch_rel": 0.0,
            "yaw_rel": 0.0
        }

        # ===== POSITION模式“世界对齐姿态” =====
        self.pos_att = {
            "quat_abs_wi": np.array([1.0, 0.0, 0.0, 0.0]),  # 在 Wi world 下的绝对姿态（对齐MOCAP）
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "src": "NONE"
        }

        self.filtered_acc = np.array([0.0, 0.0, 0.0])

        self.pos_cmd_data = {"roll": 0.0, "pitch": 0.0, "throttle": 1000.0}
        self.last_pos_cmd_time = 0.0
        self.pos_cmd_timeout = 0.2
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0

        self.last_published_dshot = [1200]*4

        self.control_mode = "MANUAL"
        self.control_debug = {"target_roll": 0.0, "target_pitch": 0.0, "target_yaw": 0.0, "position_cmd_valid": False}

        # ===== MOCAP 姿态（原始）=====
        self.mocap_quat_wm = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_mocap_time = 0.0
        self.mocap_timeout = 0.15

        # yaw hold / yaw_sp
        self.yaw_hold = None
        self.yaw_sp = None
        self.last_yaw_sp_time = 0.0
        self.yaw_sp_timeout = 0.3

        # ===== 对齐：q_align 让 IMU abs 对齐到 MOCAP abs（在 Wi world 下）=====
        self.q_align_wi = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_align_valid = False

        # ===== 固定世界系变换：Wm(x=R,y=F,z=U) -> Wi(x=F,y=L,z=U) =====
        self.R_wi_wm = np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0]
        ], dtype=float)
        self.q_wi_wm = quat_from_rotmat(self.R_wi_wm)

        # ===== 固定外参：Tracker坐标系 -> 机体Body坐标系 =====
        # Tracker x = Body y, Tracker y = Body x, Tracker z = -Body z
        self.R_b_t = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=float)
        self.R_t_b = self.R_b_t.T
        self.q_t_b = quat_from_rotmat(self.R_t_b)

        # ===== 可选修正：FRD->FLU（绕x轴180°）=====
        self.R_frd_to_flu = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=float)
        self.q_frd_to_flu = quat_from_rotmat(self.R_frd_to_flu)

    def _init_controllers(self):
        self.pid_roll_rate = ImprovedPID(
            kp=cfg.PID_ROLL_RATE["kp"] * 3.0,
            ki=cfg.PID_ROLL_RATE["ki"] * 0.0,
            kd=cfg.PID_ROLL_RATE["kd"],
            i_max=0.5, i_min=-0.5,
            use_angular_acc=False, node=self, axis="roll_rate"
        )
        self.pid_pitch_rate = ImprovedPID(
            kp=cfg.PID_PITCH_RATE["kp"] * 3.0,
            ki=cfg.PID_PITCH_RATE["ki"] * 0.0,
            kd=cfg.PID_PITCH_RATE["kd"],
            i_max=0.5, i_min=-0.5,
            use_angular_acc=False, node=self, axis="pitch_rate"
        )

        self.pid_yaw_angle = ImprovedPID(
            kp=cfg.PID_YAW_ANGLE["kp"] * 7.0,
            ki=cfg.PID_YAW_ANGLE["ki"] * 0.2,
            kd=cfg.PID_YAW_ANGLE["kd"] * 0.0,
            i_max=0.2, i_min=-0.2,
            use_angular_acc=False, node=self, axis="yaw"
        )
        self.pid_yaw_rate = ImprovedPID(
            kp=cfg.PID_YAW_RATE["kp"] * 15.0,
            ki=cfg.PID_YAW_RATE["ki"] * 0.0,
            kd=cfg.PID_YAW_RATE["kd"] * 0.6,
            i_max=0.1, i_min=-0.1,
            use_angular_acc=False, node=self, axis="yaw_rate"
        )

    def _init_state(self):
        self.state = {
            "armed": False,
            "motor_outputs": np.array([1000.0]*4),
            "init_quat": None,
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.8,
            "last_debug_time": 0.0,
            "last_control_debug_time": 0.0
        }
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        self.yaw_dshot_gain = 0.6
        self.max_yaw_rate_cmd = 1.5

    # ========== callbacks ==========
    def _rc_callback(self, msg):
        with self.lock:
            self.rc_data["left_y"] = msg.left_y
            self.rc_data["left_x"] = msg.left_x
            self.rc_data["right_x"] = msg.right_x
            self.rc_data["right_y"] = -msg.right_y
            self.rc_data["left_switch"] = msg.left_switch
            self.rc_data["right_switch"] = msg.right_switch
            self.last_rc_time = self.get_clock().now().nanoseconds / 1e9
            self._update_arming_and_mode()

    def _correct_quat_sign(self, quat):
        # 保持你的做法（不动手动逻辑）
        w, x, y, z = quat
        y = -y
        z = -z
        return quat_normalize(np.array([w, x, y, z]))

    def _imu_callback(self, msg: Imu):
        # ✅ MANUAL 模式逻辑保持不变
        with self.lock:
            q_abs = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z], dtype=float)
            q_abs = self._correct_quat_sign(q_abs)

            if self.state["init_quat"] is None:
                self.state["init_quat"] = q_abs.copy()
                self.state["initialized"] = True
                self.get_logger().info("📡 IMU初始化完成")

            # ===== 手动模式相对姿态（保留）=====
            q_rel = quat_mult(q_abs, quat_inv(self.state["init_quat"]))
            roll_rel, pitch_rel, yaw_rel = quat_to_eul(q_rel)
            yaw_rel = wrap_pi(yaw_rel)

            self.imu_data["quat_abs"] = q_abs
            self.imu_data["quat_rel"] = q_rel
            self.imu_data["roll_rel"] = roll_rel
            self.imu_data["pitch_rel"] = pitch_rel
            self.imu_data["yaw_rel"] = yaw_rel

            # gyro（保留你的符号逻辑）
            gyro_rotated = rotate_gyro_data(
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            )
            self.imu_data["gyro"] = gyro_rotated

            # 发布 /imu_angle 仍然是“相对init”的（保持你的工具链不破坏）
            angle_msg = Vector3()
            angle_msg.x = np.rad2deg(roll_rel)
            angle_msg.y = np.rad2deg(pitch_rel)
            angle_msg.z = np.rad2deg(yaw_rel)
            self.pub_imu_angle.publish(angle_msg)

            gyro_msg = Vector3()
            gyro_msg.x = float(gyro_rotated[0])
            gyro_msg.y = float(gyro_rotated[1])
            gyro_msg.z = float(gyro_rotated[2])
            self.pub_imu_gyro.publish(gyro_msg)

            self.last_imu_time = self.get_clock().now().nanoseconds / 1e9

    def _mocap_pose_callback(self, msg: PoseStamped):
        with self.lock:
            q_wm = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ], dtype=float)
            q_wm = quat_normalize(q_wm)

            # ✅ 额外稳健：保证 mocap 四元数符号连续（避免 q/-q 导致欧拉跳）
            if np.dot(q_wm, self.mocap_quat_wm) < 0:
                q_wm = -q_wm

            self.mocap_quat_wm = q_wm
            self.last_mocap_time = self.get_clock().now().nanoseconds / 1e9

    def _filtered_acc_callback(self, msg):
        with self.lock:
            self.filtered_acc = np.array([msg.x, msg.y, msg.z], dtype=float)

    def _pos_cmd_callback(self, msg):
        with self.lock:
            self.pos_cmd_data["roll"] = msg.x
            self.pos_cmd_data["pitch"] = msg.y   # ✅关键：不再取反（避免双翻转）
            self.pos_cmd_data["throttle"] = msg.z
            self.last_pos_cmd_time = self.get_clock().now().nanoseconds / 1e9
            self.control_debug["target_roll"] = msg.x
            self.control_debug["target_pitch"] = msg.y

    def _yaw_sp_callback(self, msg: Float64):
        with self.lock:
            self.yaw_sp = wrap_pi(float(msg.data))
            self.last_yaw_sp_time = self.get_clock().now().nanoseconds / 1e9

    # ========== mode / arm ==========
    def _reset_all_controllers(self):
        for pid in [self.pid_yaw_angle, self.pid_yaw_rate, self.pid_roll_rate, self.pid_pitch_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        self.get_logger().info("🔄 所有PID控制器已重置")

    def _update_arming_and_mode(self):
        left_switch = self.rc_data["left_switch"]
        right_switch = self.rc_data["right_switch"]

        if left_switch == 2:
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                self.yaw_hold = None
                self.yaw_sp = None
                self.q_align_valid = False
                self.get_logger().info("🔒 已锁定 (左开关拨至底部)")
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                self._reset_all_controllers()
            return

        if left_switch in {1, 3} and not self.state["armed"]:
            if not self.state["initialized"]:
                self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                return

            self.state["armed"] = True
            if right_switch in {1, 3}:
                self.control_mode = "POSITION"
                self.yaw_hold = None  # ✅ 进入 POSITION 让 yaw_hold 重新用世界yaw初始化一次
                self.get_logger().info("🔓 已解锁 - 位置控制模式 | 全姿态世界对齐(MOCAP优先，IMU对齐fallback)")
            else:
                self.control_mode = "MANUAL"
                self.yaw_hold = None
                self.get_logger().info("🔓 已解锁 - 手动模式（相对init姿态）")

            self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
            self._reset_all_controllers()
            return

        if not self.state["armed"]:
            return

        if right_switch in {1, 3} and self.control_mode != "POSITION":
            self.control_mode = "POSITION"
            self.yaw_hold = None  # ✅ 切入 POSITION 重新锁一次 yaw_hold
            self.get_logger().info("🎯 切到位置控制模式 | 全姿态世界对齐")
            self._reset_all_controllers()
        elif right_switch == 2 and self.control_mode != "MANUAL":
            self.control_mode = "MANUAL"
            self.yaw_hold = None
            self.get_logger().info("✈️ 切到手动模式")
            self._reset_all_controllers()

    # ========== core loop ==========
    def _check_data_validity(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        rc_timeout = current_time - self.last_rc_time > cfg.DATA_TIMEOUT
        imu_timeout = current_time - self.last_imu_time > cfg.DATA_TIMEOUT
        if rc_timeout or imu_timeout:
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                self.yaw_hold = None
                self.yaw_sp = None
                self.q_align_valid = False
                error_msg = ""
                if rc_timeout:
                    error_msg += "遥控器数据超时 "
                if imu_timeout:
                    error_msg += "IMU数据超时"
                self.get_logger().error(f"🔴 {error_msg}，强制上锁")
            return False
        return True

    def _control_loop(self):
        with self.lock:
            if not self._check_data_validity():
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
            if not self.state["armed"]:
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return

            dt = 1.0 / cfg.CONTROL_FREQ
            current_time = self.get_clock().now().nanoseconds / 1e9

            if self.control_mode == "MANUAL":
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()

            elif self.control_mode == "POSITION":
                pos_cmd_timeout = (current_time - self.last_pos_cmd_time) > self.pos_cmd_timeout
                if pos_cmd_timeout:
                    self.control_mode = "MANUAL"
                    self.yaw_hold = None
                    self.get_logger().warn("⚠️ 位置指令超时，自动切回手动")
                    throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                else:
                    throttle_raw = np.clip(self.pos_cmd_data["throttle"], 1000.0, 2000.0)
                    roll_target = np.clip(self.pos_cmd_data["roll"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    pitch_target = np.clip(self.pos_cmd_data["pitch"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    throttle = (throttle_raw - 1000.0) / 1000.0 * 1000.0
                    yaw_stick = 0.0
                    self.control_debug["position_cmd_valid"] = True
            else:
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()

            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(
                roll_target, pitch_target, yaw_stick, dt
            )
            self._publish_control_debug(roll_target, pitch_target, dt)

            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
            self._publish_control_status()

    def _process_stick(self):
        # ✅ 保持你的手动逻辑不变
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > cfg.RC_DEAD_ZONE_ROLL else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > cfg.RC_DEAD_ZONE_PITCH else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > cfg.RC_DEAD_ZONE_YAW else 0.0
        throttle_raw = self.rc_data["left_y"] if abs(self.rc_data["left_y"]) > cfg.RC_DEAD_ZONE_THROTTLE else 0.0

        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        throttle = np.clip((throttle_raw + 1.0) / 2.0 * 1000.0, 0.0, 1000.0)
        return throttle, roll_target, pitch_target, yaw_raw

    # ===== POSITION 世界对齐姿态（全姿态）=====
    def _update_position_world_attitude(self):
        """POSITION 模式世界对齐姿态（满足你最新诉求）：

        - roll / pitch：始终使用 IMU_ALIGNED（用 mocap 更新对齐四元数 q_align_wi，但姿态细节跟随 IMU）
        - yaw：始终以 MOCAP 为准；MOCAP 掉线时保持最后一次 MOCAP yaw（HOLD）
        """
        now = self.get_clock().now().nanoseconds / 1e9
        mocap_ok = (now - self.last_mocap_time) < self.mocap_timeout

        q_imu_abs = self.imu_data["quat_abs"]  # 修正后的 IMU abs quaternion (body)

        # 用于 yaw hold：保存最后一次 MOCAP yaw（Wi 语义 + CW 语义）
        if not hasattr(self, "last_mocap_yaw_wi"):
            self.last_mocap_yaw_wi = 0.0
            self.has_mocap_yaw = False

        # ----------------------------
        # 1) 若 mocap 有效：计算 Wi 下的 mocap 姿态，用于更新对齐 & 提取 yaw
        # ----------------------------
        if mocap_ok:
            q_wm_t = self.mocap_quat_wm

            # R_Wm_B = R_Wm_T * R_T_B   (四元数：q_wm_b = q_wm_t ⊗ q_t_b)
            q_wm_b = quat_mult(q_wm_t, self.q_t_b)

            # 转到 Wi world：q_wi_b_mocap = q_wi_wm ⊗ q_wm_b
            q_wi_b_mocap = quat_mult(self.q_wi_wm, q_wm_b)

            # 可选：FRD -> FLU 修正（绕 body-x 180°）
            if self.ENABLE_FRD_TO_FLU_FIX:
                q_wi_b_mocap = quat_mult(q_wi_b_mocap, self.q_frd_to_flu)

            q_wi_b_mocap = quat_normalize(q_wi_b_mocap)

            # mocap 四元数符号连续（避免 q / -q 欧拉跳）
            if np.dot(q_wi_b_mocap, self.pos_att["quat_abs_wi"]) < 0:
                q_wi_b_mocap = -q_wi_b_mocap

            # 更新对齐：希望满足 q_align_wi ⊗ q_imu_abs ≈ q_wi_b_mocap
            q_align_new = quat_mult(q_wi_b_mocap, quat_inv(q_imu_abs))
            q_align_new = quat_normalize(q_align_new)

            # ✅ 关键：对齐更新做“慢更新”（互补思想：mocap 纠偏慢，IMU 提供快）
            # 你可以把 alpha 调小（更信 IMU），或调大（更贴 mocap）
            alpha = 0.08
            if not self.q_align_valid:
                self.q_align_wi = q_align_new
                self.q_align_valid = True
            else:
                # 线性插值后归一化（近似 slerp，足够用）
                if np.dot(q_align_new, self.q_align_wi) < 0:
                    q_align_new = -q_align_new
                self.q_align_wi = quat_normalize((1.0 - alpha) * self.q_align_wi + alpha * q_align_new)

            # 提取 mocap yaw（Wi 下），并应用 CW 语义
            r_m, p_m, y_m = quat_to_eul(q_wi_b_mocap)
            y_m = wrap_pi(float(y_m))
            if self.YAW_POSITIVE_CW:
                y_m = wrap_pi(-y_m)
            self.last_mocap_yaw_wi = float(y_m)
            self.has_mocap_yaw = True

        # ----------------------------
        # 2) roll/pitch：始终用 IMU_ALIGNED（若 align 还不可用，则退化 IMU_RAW）
        # ----------------------------
        if self.q_align_valid:
            q_meas = quat_mult(self.q_align_wi, q_imu_abs)
            q_meas = quat_normalize(q_meas)
            src = "IMU_ALIGNED"
        else:
            q_meas = quat_normalize(q_imu_abs)
            src = "IMU_RAW"

        # roll/pitch 从 q_meas 提取；yaw 永远不用它
        r, p, _ = quat_to_eul(q_meas)

        # yaw：永远来自 mocap（掉线时 HOLD）
        if self.has_mocap_yaw:
            y = float(self.last_mocap_yaw_wi)
        else:
            y = 0.0

        # pitch 符号统一：抬头为正
        p = -float(p)

        # ✅ yaw 注入：roll/pitch 来自 IMU_ALIGNED，但 measured quaternion 的 yaw 强制使用 mocap yaw/hold
        quat_yawfix = eul2quat_matlab([float(p), float(r), float(y)])
        # 四元数符号连续
        if np.dot(quat_yawfix, self.pos_att["quat_abs_wi"]) < 0:
            quat_yawfix = -quat_yawfix

        self.pos_att["quat_abs_wi"] = quat_yawfix
        self.pos_att["roll"] = float(r)
        self.pos_att["pitch"] = float(p)
        self.pos_att["yaw"] = wrap_pi(float(y))
        self.pos_att["src"] = "IMU_ALIGNED_YAWFIX" if src == "IMU_ALIGNED" else src



    # ========== 四元数误差姿态外环 + yaw闭环 ==========

    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        # ===== 手动模式测量仍然用相对姿态（不变）=====
        if self.control_mode == "MANUAL":
            roll_measured = float(self.imu_data["roll_rel"])
            pitch_measured = float(self.imu_data["pitch_rel"])
            yaw_measured = float(self.imu_data["yaw_rel"])
            yaw_src = "IMU_REL"
        else:
            # ===== 位置模式：使用全姿态世界对齐测量 =====
            self._update_position_world_attitude()
            roll_measured = float(self.pos_att["roll"])
            pitch_measured = float(self.pos_att["pitch"])
            yaw_measured = float(self.pos_att["yaw"])
            yaw_src = str(self.pos_att["src"])

        # ===== yaw target =====
        now = self.get_clock().now().nanoseconds / 1e9
        yaw_target_src = "NONE"
        yaw_target = yaw_measured

        if self.control_mode == "POSITION":
            if self.yaw_hold is None:
                self.yaw_hold = float(yaw_measured)

            yaw_sp_ok = (self.yaw_sp is not None) and ((now - self.last_yaw_sp_time) < self.yaw_sp_timeout)
            if yaw_sp_ok:
                yaw_target = wrap_pi(float(self.yaw_sp))
                yaw_target_src = "YAW_SP"
            else:
                yaw_target = wrap_pi(float(self.yaw_hold))
                yaw_target_src = "HOLD"

            omega_sp_yaw = self.pid_yaw_angle.update(yaw_target, yaw_measured, dt, angular_acc=None)
            omega_sp_yaw = float(np.clip(omega_sp_yaw, -self.max_yaw_rate_cmd, self.max_yaw_rate_cmd))
        else:
            omega_sp_yaw = float(yaw_rate_cmd)

        # =========================
        # ✅核心修复：POSITION下 setpoint yaw 用 yaw_target
        # =========================
        yaw_for_setpoint = yaw_target if self.control_mode == "POSITION" else yaw_measured

        # ===== roll/pitch 外环（保留你的结构，但 yaw 用上面修复后的）=====
        # ✅ 关键：POSITION 模式下，把 pitch 统一到“抬头为正”的约定再去做四元数误差
        # eul2quat_matlab([pitch, roll, yaw]) 这一套在很多实现里等价于“绕 +Y 的正转=低头为正”，
        # 所以这里对 pitch 做一次取反，避免出现“越抬头越给反向修正”的现象。
        #if self.control_mode == "POSITION":
            #pitch_target_use = -pitch_target
            #pitch_measured_use = pitch_measured
        #else:
        pitch_target_use = pitch_target
        pitch_measured_use = pitch_measured

        quat_setpoint = eul2quat_matlab([pitch_target_use, roll_target, yaw_for_setpoint])

        # ✅ POSITION：measured quaternion 直接用 yaw 注入后的 quat_abs_wi（避免 IMU yaw 漂移/源切换引入等价解跳变）
        if self.control_mode == "POSITION":
            quat_measured = self.pos_att["quat_abs_wi"]
        else:
            quat_measured = eul2quat_matlab([pitch_measured_use, roll_measured, yaw_measured])

        # ✅ 防止 q / -q 等价解导致突然走 180° 路径
        if np.dot(quat_setpoint, quat_measured) < 0:
            quat_setpoint = -quat_setpoint

        q_e = calculateErrorQuaternion(quat_setpoint, quat_measured)
        q_e0 = q_e[0]
        q_ej = q_e[1:4]
        A = 1.0 if q_e0 >= 0 else -1.0
        TIME_CONSTANT = 0.09
        B = q_ej * (2.0 / TIME_CONSTANT)
        Omega_sp = A * B * cfg.Kp_ANGLE
        omega_sp_roll, omega_sp_pitch, _ = Omega_sp

        gyro = self.imu_data["gyro"].copy()
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]

        gyro_yaw_for_control = gyro[2]

        # ✅ yaw 角速度符号修正：POSITION下保持与世界yaw正方向一致
        # （原代码这里算了 gyro_yaw_for_control，但后面又用 gyro[2] 覆盖，导致yaw通道符号错乱）
        if self.control_mode == "POSITION":
            gyro_yaw_for_control = gyro[2]

        torque_roll = self.pid_roll_rate.update(omega_sp_roll, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(omega_sp_pitch, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(omega_sp_yaw, gyro_yaw_for_control, dt, self.filtered_acc[2])

        torque_roll = float(np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"]))
        torque_pitch = float(np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"]))
        torque_yaw = float(np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"]))

        # 低频打印
        if time.time() - self.state["last_debug_time"] > 0.5:
            if self.control_mode == "POSITION":
                yaw_err_deg = np.rad2deg(wrap_pi(yaw_target - yaw_measured)) if yaw_target_src != "NONE" else 0.0
                self.get_logger().info(
                    f"🧭 POS-ATT | meas_src={yaw_src} | "
                    f"RPY=({np.rad2deg(roll_measured):.2f},{np.rad2deg(pitch_measured):.2f},{np.rad2deg(yaw_measured):.2f})deg | "
                    f"tgt_src={yaw_target_src} tgt_yaw={np.rad2deg(yaw_target):.2f}deg err={yaw_err_deg:.2f}deg "
                    f"rate_sp={omega_sp_yaw:.2f}rad/s | "
                    f"setpointYawUsed={np.rad2deg(yaw_for_setpoint):.2f}deg"
                )
            self.state["last_debug_time"] = time.time()

        torque_msg = Vector3()
        torque_msg.x = float(torque_roll)
        torque_msg.y = float(torque_pitch)
        torque_msg.z = float(torque_yaw)
        self.pub_torque_output.publish(torque_msg)
        return torque_roll, torque_pitch, torque_yaw

    # ===== 原样保留你的发布/混控 =====
    def _publish_control_debug(self, roll_target, pitch_target, dt):
        current_time = time.time()
        mode_info = Vector3()
        mode_info.x = 1.0 if self.control_mode == "MANUAL" else (2.0 if self.control_mode == "POSITION" else 0.0)
        cmd_timeout = time.time() - self.last_pos_cmd_time > self.pos_cmd_timeout
        mode_info.y = 1.0 if (self.control_mode == "POSITION" and not cmd_timeout) else 0.0
        motor_outputs = self.state["motor_outputs"]
        mode_info.z = float(np.max(motor_outputs) - np.min(motor_outputs))
        self.pub_control_mode_info.publish(mode_info)

        pos_cmd_status = Vector3()
        pos_cmd_status.x = float(self.pos_cmd_data["roll"])
        pos_cmd_status.y = float(self.pos_cmd_data["pitch"])
        pos_cmd_status.z = float(self.pos_cmd_data["throttle"])
        self.pub_position_cmd_status.publish(pos_cmd_status)

        if current_time - self.state["last_control_debug_time"] > 0.2:
            control_details = Vector3()
            # 这里仍然发布“相对init”的IMU角度，避免破坏你现有调试工具链
            control_details.x = float(np.rad2deg(self.imu_data["roll_rel"]))
            control_details.y = float(np.rad2deg(self.imu_data["pitch_rel"]))
            control_details.z = float(np.rad2deg(self.imu_data["yaw_rel"]))
            self.pub_control_details.publish(control_details)
            self.state["last_control_debug_time"] = current_time

    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        base = 1000.0 + throttle
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain

        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE

        motor_max = np.max([motor1, motor2, motor3, motor4])
        motor_min = np.min([motor1, motor2, motor3, motor4])
        if motor_max - motor_min > 600:
            scale = 600 / (motor_max - motor_min) if motor_max != motor_min else 1.0
            motor_avg = (motor1 + motor2 + motor3 + motor4) / 4
            motor1 = motor_avg + (motor1 - motor_avg) * scale
            motor2 = motor_avg + (motor2 - motor_avg) * scale
            motor3 = motor_avg + (motor3 - motor_avg) * scale
            motor4 = motor_avg + (motor4 - motor_avg) * scale

        alpha = 0.18
        smoothed_motors = alpha * np.array([motor1, motor2, motor3, motor4]) + (1 - alpha) * self.state["motor_outputs"]
        return smoothed_motors

    def _publish_dshot(self, motor_pwm):
        msg = WriteDSHOT()
        if isinstance(motor_pwm, (int, float)):
            dshot_val = pwm_to_dshot(motor_pwm)
            msg.channel1 = dshot_val
            msg.channel2 = dshot_val
            msg.channel3 = dshot_val
            msg.channel4 = dshot_val
            self.last_published_dshot = [dshot_val]*4
        elif isinstance(motor_pwm, (list, np.ndarray)):
            if len(motor_pwm) >= 4:
                dshot = [pwm_to_dshot(p) for p in motor_pwm[:4]]
                msg.channel1 = dshot[2]
                msg.channel2 = dshot[0]
                msg.channel3 = dshot[1]
                msg.channel4 = dshot[3]
                self.last_published_dshot = dshot
            else:
                self.get_logger().error(f"❌ 电机PWM数组长度不足: {len(motor_pwm)}")
                return
        else:
            self.get_logger().error(f"❌ 不支持的motor_pwm类型: {type(motor_pwm)}")
            return
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ 发布DSHOT失败: {e}")

    def _publish_control_status(self):
        status_msg = Vector3()
        status_msg.x = 1.0 if self.control_mode == "MANUAL" else (2.0 if self.control_mode == "POSITION" else 0.0)
        status_msg.y = 1.0 if self.state["armed"] else 0.0
        status_msg.z = float(self.rc_data.get("left_switch", 0))
        self.pub_control_status.publish(status_msg)

    def _publish_status(self):
        if not self.state["armed"]:
            return
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll_rel"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch_rel"]))
            self.get_logger().info(
                f"📊 飞行状态 | 模式:{self.control_mode} | "
                f"Roll(rel)={roll_error:.2f}° | Pitch(rel)={pitch_error:.2f}° | "
                f"DSHOT:{dshot}"
            )
            self.state["last_debug_time"] = current_time

    def destroy_node(self):
        self.get_logger().info("🛑 正在关闭飞控...")
        try:
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
            time.sleep(0.05)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 销毁时发布上锁指令失败: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    try:
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info("🛑 用户中断，停止飞控")
    except Exception as e:
        controller.get_logger().error(f"❌ 飞控运行出错: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
