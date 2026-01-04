import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float64MultiArray, Float64
from custom_msgs.msg import ReadDJIRC
import numpy as np
import time
from soft_drone_controller.config import controller_params as cfg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def deg2rad(d):
    return float(d) * np.pi / 180.0


def wrap_pi(a):
    a = float(a)
    return float(np.arctan2(np.sin(a), np.cos(a)))


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def rotmat_from_quat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)


def yaw_from_rotmat(R):
    # yaw around +Z (Z-up), CCW positive
    return float(np.arctan2(R[1, 0], R[0, 0]))


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
    """带 dt 的简单 PID"""
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

        # 与飞控一致：yaw 顺时针为正
        self.YAW_POSITIVE_CW = True

        # pitch 正方向：nose up 为正
        self.PITCH_POS_IS_NOSE_UP = True

        # yaw 全由 mocap
        self.YAW_ONLY_MOCAP = True

        # 统一 FRD
        self.ENABLE_FRD_TO_FLU_FIX = False

        # ✅✅✅ 输出姿态增益（放大 roll/pitch 2~3 倍）
        self.ATTITUDE_CMD_GAIN = 2.6   # 建议：2.0 -> 2.6 -> 3.0 逐步试

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
        self.imu_angle_sub = self.create_subscription(
            Vector3, '/imu_angle', self.imu_angle_callback, qos_best_effort
        )
        self.mocap_pose_sub = self.create_subscription(
            PoseStamped, '/Tracker0/pose', self.mocap_pose_callback, qos_best_effort
        )

        # ===== 发布 =====
        self.pos_cmd_pub = self.create_publisher(Vector3, '/attitude_position_cmd', qos_reliable)
        self.yaw_sp_pub = self.create_publisher(Float64, '/yaw_hold_sp', qos_reliable)

        self.YAW_HOLD_MODE = "CONST"
        self.YAW_HOLD_SP_DEG = 0.0

        self.PATH_IN_WM = True

        # Wm -> Wi
        # Wm: x=Right, y=Front, z=Up
        # Wi: x=Front, y=Left,  z=Up
        self.R_wi_wm = np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0]
        ], dtype=float)

        # Tracker -> Body 外参
        self.R_b_t = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=float)
        self.R_t_b = self.R_b_t.T

        # ===== 参数 =====
        self.default_height = float(getattr(cfg, "POSITION_DEFAULT_HEIGHT", 0.7))
        self.hover_th_ratio = float(getattr(cfg, "HOVER_THROTTLE_RATIO", 0.5))

        self.takeoff_active = True
        self.height_sp = None
        self.height_target = self.default_height
        self.max_climb_rate = float(getattr(cfg, "TAKEOFF_CLIMB_RATE", 0.15))

        self.pos_dead_xy = float(getattr(cfg, "POSITION_DEADZONE_XY", 0.02))
        self.vxy_limit = float(getattr(cfg, "POSITION_VXY_LIMIT", 0.8))
        self.vz_limit = float(getattr(cfg, "POSITION_VZ_LIMIT", 0.6))

        # ===== PID =====
        self.x_loop = PID(cfg.POSITION_XY_KP, cfg.POSITION_XY_KI, 0.0,
                          i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.y_loop = PID(cfg.POSITION_XY_KP, cfg.POSITION_XY_KI, 0.0,
                          i_max=cfg.POSITION_XY_INT_LIMIT, i_min=-cfg.POSITION_XY_INT_LIMIT)
        self.z_loop = PID(cfg.POSITION_Z_KP, cfg.POSITION_Z_KI, 0.0,
                          i_max=cfg.POSITION_Z_INT_LIMIT, i_min=-cfg.POSITION_Z_INT_LIMIT)

        self.x_vel_loop = PID(cfg.VELOCITY_XY_KP, 0.0, 0.0,
                              i_max=cfg.VELOCITY_XY_INT_LIMIT, i_min=-cfg.VELOCITY_XY_INT_LIMIT)
        self.y_vel_loop = PID(cfg.VELOCITY_XY_KP, 0.0, 0.0,
                              i_max=cfg.VELOCITY_XY_INT_LIMIT, i_min=-cfg.VELOCITY_XY_INT_LIMIT)

        # ===== 滤波（Wi）=====
        self.x_f = AlphaFilter(alpha=cfg.POSITION_FILTER_ALPHA_POS, init=0.0)
        self.y_f = AlphaFilter(alpha=cfg.POSITION_FILTER_ALPHA_POS, init=0.0)
        self.z_f = AlphaFilter(alpha=cfg.POSITION_FILTER_ALPHA_POS, init=0.0)

        self.vx_lpf = FirstOrderLPF(tau=0.08, init=0.0)
        self.vy_lpf = FirstOrderLPF(tau=0.08, init=0.0)
        self.vz_lpf = FirstOrderLPF(tau=0.12, init=0.0)

        self.vx_sp_lpf = FirstOrderLPF(tau=0.06, init=0.0)
        self.vy_sp_lpf = FirstOrderLPF(tau=0.06, init=0.0)
        self.vz_sp_lpf = FirstOrderLPF(tau=0.10, init=0.0)

        # ===== 角度限制 =====
        self.max_angle = float(getattr(cfg, "POSITION_XY_MAX_ANGLE", 0.35))
        self.max_angle_rate = float(getattr(cfg, "POSITION_XY_MAX_ANGLE_RATE", 3.0))
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

        self.yaw_rad_imu = 0.0
        self.last_yaw_imu_time = 0.0

        self.yaw_rad_mocap_wi = 0.0
        self.last_yaw_mocap_time = 0.0
        self.mocap_yaw_timeout = 0.15

        self.yaw_rad_mocap_hold = 0.0
        self.has_mocap_yaw = False

        self.yaw_hold_sp = None

        self.last_ctrl_time = time.time()
        self.prev_locked = True
        self.last_log_time = 0.0

        self.get_logger().info("✅ PositionController(FRD) 启动：已放大 roll/pitch 输出增益，先压住漂移")

        self.timer = self.create_timer(1.0 / cfg.POSITION_CONTROL_FREQ, self.do_control)

    # Wm -> Wi
    def _wm_to_wi(self, x_wm, y_wm, z_wm):
        return float(y_wm), float(-x_wm), float(z_wm)

    def rc_callback(self, msg):
        self.rc_data = msg

    def path_callback(self, msg):
        self.path_data = msg

    def imu_angle_callback(self, msg: Vector3):
        yaw_imu = wrap_pi(deg2rad(msg.z))
        if self.YAW_POSITIVE_CW:
            yaw_imu = wrap_pi(-yaw_imu)
        self.yaw_rad_imu = yaw_imu
        self.last_yaw_imu_time = time.time()

    def mocap_pose_callback(self, msg: PoseStamped):
        q_wm_t = np.array([
            float(msg.pose.orientation.w),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z)
        ], dtype=float)
        R_wm_t = rotmat_from_quat(q_wm_t)

        R_wm_b = R_wm_t @ self.R_t_b
        R_wi_b = self.R_wi_wm @ R_wm_b

        yaw_wi = wrap_pi(yaw_from_rotmat(R_wi_b))
        if self.YAW_POSITIVE_CW:
            yaw_wi = wrap_pi(-yaw_wi)

        self.yaw_rad_mocap_wi = float(yaw_wi)
        self.last_yaw_mocap_time = time.time()

        self.yaw_rad_mocap_hold = float(yaw_wi)
        self.has_mocap_yaw = True

    def pose_callback(self, msg: PoseStamped):
        self.pose_data = msg

        x_wm = float(msg.pose.position.x)
        y_wm = float(msg.pose.position.y)
        z_wm = float(msg.pose.position.z)

        x_wi, y_wi, z_wi = self._wm_to_wi(x_wm, y_wm, z_wm)

        xf = self.x_f.update(x_wi)
        yf = self.y_f.update(y_wi)
        zf = self.z_f.update(z_wi)

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

    def _slew(self, target, last, dt, max_rate):
        dt = float(max(dt, 1e-4))
        max_step = float(max_rate) * dt
        return float(np.clip(target, last - max_step, last + max_step))

    def _world_to_body_2d(self, vx_w, vy_w, yaw_rad):
        """
        Wi(world): x-front, y-left
        Body(FRD): x-front, y-right
        """
        cy = np.cos(yaw_rad)
        sy = np.sin(yaw_rad)

        vx_b =  cy * vx_w + sy * vy_w
        vy_b_left = -sy * vx_w + cy * vy_w
        vy_b = -vy_b_left
        return float(vx_b), float(vy_b)

    def _update_height_sp_planA(self, current_z, dt_ctrl):
        if self.height_sp is None:
            self.height_sp = float(current_z)
        if not self.takeoff_active:
            return
        target = float(self.height_target)
        step = float(self.max_climb_rate) * float(dt_ctrl)
        if self.height_sp < target:
            self.height_sp = min(self.height_sp + step, target)
        else:
            self.height_sp = max(self.height_sp - step, target)

    def _get_yaw_for_transform(self):
        now = time.time()
        mocap_fresh = (now - self.last_yaw_mocap_time) < self.mocap_yaw_timeout

        if self.YAW_ONLY_MOCAP:
            if mocap_fresh:
                return float(self.yaw_rad_mocap_wi), "MOCAP(Wi+Ext+FRD+CW)"
            if self.has_mocap_yaw:
                return float(self.yaw_rad_mocap_hold), "MOCAP_HOLD(FRD+CW)"
            return 0.0, "MOCAP_NONE"

        if mocap_fresh:
            return float(self.yaw_rad_mocap_wi), "MOCAP(Wi+Ext+FRD+CW)"
        return float(self.yaw_rad_imu), "IMU(rel+CW)"

    def _update_yaw_hold_sp(self, locked_edge=False):
        mode = str(self.YAW_HOLD_MODE).upper()

        if mode == "CONST":
            self.yaw_hold_sp = wrap_pi(deg2rad(self.YAW_HOLD_SP_DEG))
            return

        if mode == "LOCK_ON_ARM":
            if locked_edge or self.yaw_hold_sp is None:
                yaw, src = self._get_yaw_for_transform()
                self.yaw_hold_sp = wrap_pi(yaw)
                self.get_logger().info(
                    f"🧭 Yaw锁定(LOCK_ON_ARM)：{np.rad2deg(self.yaw_hold_sp):.1f}deg (src={src})"
                )
            return

        if locked_edge or self.yaw_hold_sp is None:
            yaw, src = self._get_yaw_for_transform()
            self.yaw_hold_sp = wrap_pi(yaw)
            self.get_logger().info(
                f"🧭 Yaw锁定(DEFAULT)：{np.rad2deg(self.yaw_hold_sp):.1f}deg (src={src})"
            )

    def _publish_yaw_sp(self):
        if self.yaw_hold_sp is None:
            return
        m = Float64()
        m.data = float(self.yaw_hold_sp)
        self.yaw_sp_pub.publish(m)

    def do_control(self):
        if self.rc_data is None or self.pose_data is None:
            return

        locked = (self.rc_data.left_switch == cfg.LOCK_SWITCH_VALUE)

        if locked:
            self.prev_locked = True
            self.height_sp = None
            self.yaw_hold_sp = None

            self.x_loop.reset(); self.y_loop.reset(); self.z_loop.reset()
            self.x_vel_loop.reset(); self.y_vel_loop.reset()
            self.vx_sp_lpf.reset(0.0); self.vy_sp_lpf.reset(0.0); self.vz_sp_lpf.reset(0.0)
            self.last_roll_cmd = 0.0
            self.last_pitch_cmd = 0.0
            return

        locked_edge = False
        if self.prev_locked and not locked:
            locked_edge = True
            current_z = float(self.z_f.filtered_)
            self.height_sp = float(current_z)
            self.height_target = float(self.default_height)
            self.z_loop.reset()
            self.get_logger().info(
                f"🟢 解锁：锁定高度 {self.height_sp:.2f}m -> 平滑爬升到 {self.height_target:.2f}m"
            )
            self.prev_locked = False

        now = time.time()
        dt_ctrl = float(np.clip(now - self.last_ctrl_time, 1e-3, 0.02))
        self.last_ctrl_time = now
        self.prev_locked = False

        current_x = float(self.x_f.filtered_)
        current_y = float(self.y_f.filtered_)
        current_z = float(self.z_f.filtered_)

        self._update_height_sp_planA(current_z, dt_ctrl)

        vx_now_w = float(self.vx_est_w)
        vy_now_w = float(self.vy_est_w)
        vz_now   = float(self.vz_est)

        path = self.path_data.data if len(self.path_data.data) >= 2 else [0.0, 0.0]
        raw_tx = float(path[0])
        raw_ty = float(path[1])

        if self.PATH_IN_WM:
            target_x, target_y, _ = self._wm_to_wi(raw_tx, raw_ty, 0.0)
            path_src = "Wm->Wi"
        else:
            target_x, target_y = raw_tx, raw_ty
            path_src = "Wi"

        target_z = float(self.height_sp if self.height_sp is not None else current_z)

        e_w = np.array([target_x - current_x,
                        target_y - current_y,
                        target_z - current_z], dtype=float)

        ex = float(e_w[0]); ey = float(e_w[1])
        target_x_eff = current_x if abs(ex) < self.pos_dead_xy else target_x
        target_y_eff = current_y if abs(ey) < self.pos_dead_xy else target_y

        vx_sp_w = self.x_loop.step(current_x, target_x_eff, dt_ctrl)
        vy_sp_w = self.y_loop.step(current_y, target_y_eff, dt_ctrl)
        vz_sp   = self.z_loop.step(current_z, target_z, dt_ctrl)

        vx_sp_w = float(np.clip(vx_sp_w, -self.vxy_limit, self.vxy_limit))
        vy_sp_w = float(np.clip(vy_sp_w, -self.vxy_limit, self.vxy_limit))
        vz_sp   = float(np.clip(vz_sp,   -self.vz_limit,  self.vz_limit))

        vx_sp_w_f = self.vx_sp_lpf.update(vx_sp_w, dt_ctrl)
        vy_sp_w_f = self.vy_sp_lpf.update(vy_sp_w, dt_ctrl)
        vz_sp_f   = self.vz_sp_lpf.update(vz_sp,   dt_ctrl)

        self._update_yaw_hold_sp(locked_edge=locked_edge)
        self._publish_yaw_sp()

        yaw, yaw_src = self._get_yaw_for_transform()

        vx_sp_b,  vy_sp_b  = self._world_to_body_2d(vx_sp_w_f, vy_sp_w_f, yaw)
        vx_now_b, vy_now_b = self._world_to_body_2d(vx_now_w,  vy_now_w,  yaw)

        roll_cmd_raw  = self.y_vel_loop.step(vy_now_b, vy_sp_b, dt_ctrl)
        pitch_cmd_raw = self.x_vel_loop.step(vx_now_b, vx_sp_b, dt_ctrl)

        # ✅✅✅ 放大输出（2~3倍）
        roll_cmd_raw  *= self.ATTITUDE_CMD_GAIN
        pitch_cmd_raw *= self.ATTITUDE_CMD_GAIN

        roll_cmd  = float(np.clip(roll_cmd_raw,  -self.max_angle, self.max_angle))

        # ✅ 向前飞必须低头
        pitch_cmd = float(np.clip(-pitch_cmd_raw, -self.max_angle, self.max_angle))

        roll_cmd  = self._slew(roll_cmd,  self.last_roll_cmd,  dt_ctrl, self.max_angle_rate)
        pitch_cmd = self._slew(pitch_cmd, self.last_pitch_cmd, dt_ctrl, self.max_angle_rate)
        self.last_roll_cmd = roll_cmd
        self.last_pitch_cmd = pitch_cmd

        z_out_ratio_increment = float(np.clip(vz_sp_f, -0.08, 0.08))
        final_throttle_ratio = float(self.hover_th_ratio + z_out_ratio_increment)
        final_throttle_ratio = float(np.clip(final_throttle_ratio, cfg.MIN_DESCEND_THROTTLE_RATIO, 1.0))
        final_throttle_pwm = 1000.0 + final_throttle_ratio * 1000.0

        cmd = Vector3()
        cmd.x = float(roll_cmd)
        cmd.y = float(pitch_cmd)
        cmd.z = float(final_throttle_pwm)
        self.pos_cmd_pub.publish(cmd)

        if now - self.last_log_time > 0.8:
            self.last_log_time = now
            self.get_logger().info(
                f"[POS-FRD] path_src={path_src} | yaw({yaw_src})={np.rad2deg(yaw):.1f}deg | "
                f"Wi_now=({current_x:.2f},{current_y:.2f},{current_z:.2f}) | "
                f"Wi_tgt=({target_x:.2f},{target_y:.2f},{target_z:.2f}) | "
                f"e_w=({e_w[0]:.2f},{e_w[1]:.2f},{e_w[2]:.2f}) | "
                f"vx_sp_b={vx_sp_b:.2f} vy_sp_b={vy_sp_b:.2f} | "
                f"roll={np.rad2deg(roll_cmd):.2f}deg pitch={np.rad2deg(pitch_cmd):.2f}deg | "
                f"thr={final_throttle_pwm:.0f} | gain={self.ATTITUDE_CMD_GAIN:.2f}"
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
