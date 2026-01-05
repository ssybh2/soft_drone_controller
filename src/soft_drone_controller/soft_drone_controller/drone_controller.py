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
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro], dtype=float)


def pwm_to_dshot(pwm_val):
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))


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
            if abs(error) < 0.015:
                self.integral = 0.0
                self.last_output = 0.0
                return 0.0
        elif self.axis == "yaw":
            if abs(error) < 0.003:
                error = 0.0
        elif self.axis in ["roll_rate", "pitch_rate", "yaw_rate"]:
            if abs(error) < 0.009:
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
        self.integral = float(np.clip(self.integral, self.i_min, self.i_max))
        i_term = self.ki * self.integral

        if self.use_angular_acc and angular_acc is not None:
            d_term = self.kd * angular_acc * self.d_term_sign
        else:
            d_term = self.kd * measurement_rate * self.d_term_sign

        output = p_term + i_term + d_term
        self.prev_measurement = measurement
        self.last_output = output
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0


class BalanceController(Node):
    def __init__(self):
        super().__init__("drone_controller_node")

        self.ENABLE_FRD_TO_FLU_FIX = True
        self.YAW_POSITIVE_CW = True

        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()

        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)

        self.get_logger().info("✅ 飞控启动完成 - MANUAL不变 / POSITION世界对齐")

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

        self.sub_rc = self.create_subscription(ReadDJIRC, "/ecat/sn2228293/app1/read", self._rc_callback, qos_best_effort)
        self.sub_imu = self.create_subscription(Imu, "/ecat/sn2228293/app2/read", self._imu_callback, qos_best_effort)
        self.sub_filtered_acc = self.create_subscription(Vector3, "/filtered_angular_acceleration", self._filtered_acc_callback, qos_reliable)
        self.sub_pos_cmd = self.create_subscription(Vector3, "/attitude_position_cmd", self._pos_cmd_callback, qos_reliable)
        self.sub_mocap_pose = self.create_subscription(PoseStamped, "/Tracker0/pose", self._mocap_pose_callback, qos_reliable)
        self.sub_yaw_sp = self.create_subscription(Float64, "/yaw_hold_sp", self._yaw_sp_callback, qos_reliable)

        self.lock = threading.Lock()

    def _init_data(self):
        self.rc_data = {
            "left_y": 0.0, "left_x": 0.0,
            "right_x": 0.0, "right_y": 0.0,
            "left_switch": cfg.LOCK_SWITCH_VALUE,
            "right_switch": 2
        }

        self.imu_data = {
            "quat_rel": np.array([1.0, 0.0, 0.0, 0.0]),
            "quat_abs": np.array([1.0, 0.0, 0.0, 0.0]),
            "gyro": np.array([0.0, 0.0, 0.0]),
            "roll_rel": 0.0, "pitch_rel": 0.0, "yaw_rel": 0.0
        }

        self.pos_att = {
            "quat_abs_wi": np.array([1.0, 0.0, 0.0, 0.0]),
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "src": "NONE"
        }

        self.filtered_acc = np.array([0.0, 0.0, 0.0])

        self.pos_cmd_data = {"roll": 0.0, "pitch": 0.0, "throttle": 1000.0}
        self.last_pos_cmd_time = 0.0
        self.pos_cmd_timeout = 0.2

        self.last_rc_time = 0.0
        self.last_imu_time = 0.0

        self.control_mode = "MANUAL"
        self.last_published_dshot = [1200]*4

        self.mocap_quat_wm = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_mocap_time = 0.0
        self.mocap_timeout = 0.15

        self.yaw_hold = None
        self.yaw_sp = None
        self.last_yaw_sp_time = 0.0
        self.yaw_sp_timeout = 0.3

        self.q_align_wi = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_align_valid = False

        self.R_wi_wm = np.array([[0.0, 1.0, 0.0],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0]], dtype=float)
        self.q_wi_wm = quat_from_rotmat(self.R_wi_wm)

        self.R_b_t = np.array([[0.0, 1.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0]], dtype=float)
        self.R_t_b = self.R_b_t.T
        self.q_t_b = quat_from_rotmat(self.R_t_b)

        self.R_frd_to_flu = np.array([[1.0, 0.0, 0.0],
                                      [0.0, -1.0, 0.0],
                                      [0.0, 0.0, -1.0]], dtype=float)
        self.q_frd_to_flu = quat_from_rotmat(self.R_frd_to_flu)

        self.last_mocap_yaw_wi = 0.0
        self.has_mocap_yaw = False

    def _init_controllers(self):
        self.pid_roll_rate = ImprovedPID(cfg.PID_ROLL_RATE["kp"]*3.0, 0.0, cfg.PID_ROLL_RATE["kd"],
                                         i_max=0.5, i_min=-0.5, use_angular_acc=False, node=self, axis="roll_rate")
        self.pid_pitch_rate = ImprovedPID(cfg.PID_PITCH_RATE["kp"]*3.0, 0.0, cfg.PID_PITCH_RATE["kd"],
                                          i_max=0.5, i_min=-0.5, use_angular_acc=False, node=self, axis="pitch_rate")
        self.pid_yaw_angle = ImprovedPID(cfg.PID_YAW_ANGLE["kp"]*7.0, cfg.PID_YAW_ANGLE["ki"]*0.2, 0.0,
                                         i_max=0.2, i_min=-0.2, use_angular_acc=False, node=self, axis="yaw")
        self.pid_yaw_rate = ImprovedPID(cfg.PID_YAW_RATE["kp"]*15.0, 0.0, cfg.PID_YAW_RATE["kd"]*0.6,
                                        i_max=0.1, i_min=-0.1, use_angular_acc=False, node=self, axis="yaw_rate")

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

    # ===================== callbacks =====================
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
        w, x, y, z = quat
        y = -y
        z = -z
        return quat_normalize(np.array([w, x, y, z]))

    def _imu_callback(self, msg: Imu):
        with self.lock:
            q_abs = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z], dtype=float)
            q_abs = self._correct_quat_sign(q_abs)

            if self.state["init_quat"] is None:
                self.state["init_quat"] = q_abs.copy()
                self.state["initialized"] = True
                self.get_logger().info("📡 IMU初始化完成")

            q_rel = quat_mult(q_abs, quat_inv(self.state["init_quat"]))
            roll_rel, pitch_rel, yaw_rel = quat_to_eul(q_rel)
            yaw_rel = wrap_pi(yaw_rel)

            self.imu_data["quat_abs"] = q_abs
            self.imu_data["quat_rel"] = q_rel
            self.imu_data["roll_rel"] = roll_rel
            self.imu_data["pitch_rel"] = pitch_rel
            self.imu_data["yaw_rel"] = yaw_rel

            gyro_rotated = rotate_gyro_data(
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            )
            self.imu_data["gyro"] = gyro_rotated

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
            q_wm = np.array([msg.pose.orientation.w,
                             msg.pose.orientation.x,
                             msg.pose.orientation.y,
                             msg.pose.orientation.z], dtype=float)
            q_wm = quat_normalize(q_wm)

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
            self.pos_cmd_data["pitch"] = msg.y
            self.pos_cmd_data["throttle"] = msg.z
            self.last_pos_cmd_time = self.get_clock().now().nanoseconds / 1e9

    def _yaw_sp_callback(self, msg: Float64):
        with self.lock:
            self.yaw_sp = wrap_pi(float(msg.data))
            self.last_yaw_sp_time = self.get_clock().now().nanoseconds / 1e9

    # ===================== arm/mode =====================
    def _reset_all_controllers(self):
        for pid in [self.pid_yaw_angle, self.pid_yaw_rate, self.pid_roll_rate, self.pid_pitch_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)

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
                self.get_logger().info("🔒 已锁定")
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
                self.yaw_hold = None
                self.get_logger().info("🔓 解锁 - POSITION")
            else:
                self.control_mode = "MANUAL"
                self.yaw_hold = None
                self.get_logger().info("🔓 解锁 - MANUAL")

            self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
            self._reset_all_controllers()
            return

        if not self.state["armed"]:
            return

        if right_switch in {1, 3} and self.control_mode != "POSITION":
            self.control_mode = "POSITION"
            self.yaw_hold = None
            self.get_logger().info("🎯 切换 POSITION")
            self._reset_all_controllers()
        elif right_switch == 2 and self.control_mode != "MANUAL":
            self.control_mode = "MANUAL"
            self.yaw_hold = None
            self.get_logger().info("✈️ 切换 MANUAL")
            self._reset_all_controllers()

    # ===================== safety =====================
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
                self.get_logger().error("🔴 RC/IMU超时，强制上锁")
            return False
        return True

    # ===================== POSITION world attitude =====================
    def _update_position_world_attitude(self):
        now = self.get_clock().now().nanoseconds / 1e9
        mocap_ok = (now - self.last_mocap_time) < self.mocap_timeout

        q_imu_abs = self.imu_data["quat_abs"]

        if mocap_ok:
            q_wm_t = self.mocap_quat_wm
            q_wm_b = quat_mult(q_wm_t, self.q_t_b)
            q_wi_b_mocap = quat_mult(self.q_wi_wm, q_wm_b)

            if self.ENABLE_FRD_TO_FLU_FIX:
                q_wi_b_mocap = quat_mult(q_wi_b_mocap, self.q_frd_to_flu)

            q_wi_b_mocap = quat_normalize(q_wi_b_mocap)

            if np.dot(q_wi_b_mocap, self.pos_att["quat_abs_wi"]) < 0:
                q_wi_b_mocap = -q_wi_b_mocap

            q_align_new = quat_mult(q_wi_b_mocap, quat_inv(q_imu_abs))
            q_align_new = quat_normalize(q_align_new)

            alpha = 0.08
            if not self.q_align_valid:
                self.q_align_wi = q_align_new
                self.q_align_valid = True
            else:
                if np.dot(q_align_new, self.q_align_wi) < 0:
                    q_align_new = -q_align_new
                self.q_align_wi = quat_normalize((1.0 - alpha) * self.q_align_wi + alpha * q_align_new)

            r_m, p_m, y_m = quat_to_eul(q_wi_b_mocap)
            y_m = wrap_pi(float(y_m))
            if self.YAW_POSITIVE_CW:
                y_m = wrap_pi(-y_m)

            self.last_mocap_yaw_wi = float(y_m)
            self.has_mocap_yaw = True

        if self.q_align_valid:
            q_meas_rp = quat_mult(self.q_align_wi, q_imu_abs)
            q_meas_rp = quat_normalize(q_meas_rp)
            src = "IMU_ALIGNED"
        else:
            q_meas_rp = quat_normalize(q_imu_abs)
            src = "IMU_RAW"

        r, p, _ = quat_to_eul(q_meas_rp)
        p = -float(p)

        y = float(self.last_mocap_yaw_wi) if self.has_mocap_yaw else 0.0

        quat_yawfix = eul2quat_matlab([float(p), float(r), float(y)])
        if np.dot(quat_yawfix, self.pos_att["quat_abs_wi"]) < 0:
            quat_yawfix = -quat_yawfix

        self.pos_att["quat_abs_wi"] = quat_yawfix
        self.pos_att["src"] = "IMU_ALIGNED_YAWFIX" if src == "IMU_ALIGNED" else src

        r2, p2, _ = quat_to_eul(quat_yawfix)
        self.pos_att["roll"] = float(r2)
        self.pos_att["pitch"] = float(p2) * (-1.0)
        self.pos_att["yaw"] = float(y)

    # ===================== control =====================
    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        if self.control_mode == "MANUAL":
            roll_measured = float(self.imu_data["roll_rel"])
            pitch_measured = float(self.imu_data["pitch_rel"])
            yaw_measured = float(self.imu_data["yaw_rel"])
            yaw_src = "IMU_REL"
        else:
            self._update_position_world_attitude()
            roll_measured = float(self.pos_att["roll"])
            pitch_measured = float(self.pos_att["pitch"])
            yaw_measured = float(self.pos_att["yaw"])
            yaw_src = str(self.pos_att["src"])

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

        yaw_for_setpoint = yaw_target if self.control_mode == "POSITION" else yaw_measured
        quat_setpoint = eul2quat_matlab([pitch_target, roll_target, yaw_for_setpoint])

        if self.control_mode == "POSITION":
            quat_measured = self.pos_att["quat_abs_wi"]
        else:
            quat_measured = eul2quat_matlab([pitch_measured, roll_measured, yaw_measured])

        if np.dot(quat_setpoint, quat_measured) < 0:
            quat_setpoint = -quat_setpoint

        q_e = calculateErrorQuaternion(quat_setpoint, quat_measured)
        A = 1.0 if q_e[0] >= 0 else -1.0
        TIME_CONSTANT = 0.09
        Omega_sp = A * q_e[1:4] * (2.0 / TIME_CONSTANT) * cfg.Kp_ANGLE
        omega_sp_roll, omega_sp_pitch, _ = Omega_sp

        gyro = self.imu_data["gyro"].copy()
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]

        torque_roll = self.pid_roll_rate.update(omega_sp_roll, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(omega_sp_pitch, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(omega_sp_yaw, gyro[2], dt, self.filtered_acc[2])

        torque_roll = float(np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"]))
        torque_pitch = float(np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"]))
        torque_yaw = float(np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"]))

        if time.time() - self.state["last_debug_time"] > 0.5 and self.control_mode == "POSITION":
            yaw_err_deg = np.rad2deg(wrap_pi(yaw_target - yaw_measured))
            self.get_logger().info(
                f"🧭 POS | src={yaw_src} | "
                f"RPY=({np.rad2deg(roll_measured):.1f},{np.rad2deg(pitch_measured):.1f},{np.rad2deg(yaw_measured):.1f})deg | "
                f"yaw_tgt({yaw_target_src})={np.rad2deg(yaw_target):.1f} err={yaw_err_deg:.1f}deg | "
                f"yaw_used={np.rad2deg(yaw_for_setpoint):.1f}"
            )
            self.state["last_debug_time"] = time.time()

        return torque_roll, torque_pitch, torque_yaw

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
        else:
            dshot = [pwm_to_dshot(p) for p in motor_pwm[:4]]
            msg.channel1 = dshot[2]
            msg.channel2 = dshot[0]
            msg.channel3 = dshot[1]
            msg.channel4 = dshot[3]
            self.last_published_dshot = dshot
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"⚠️ DSHOT publish failed: {e}")

    def _publish_status(self):
        if not self.state["armed"]:
            return
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll_rel"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch_rel"]))
            self.get_logger().info(
                f"📊 状态 | 模式:{self.control_mode} | "
                f"Roll(rel)={roll_error:.2f}° | Pitch(rel)={pitch_error:.2f}° | "
                f"DSHOT:{dshot}"
            )
            self.state["last_debug_time"] = current_time

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
            else:
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()

            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(
                roll_target, pitch_target, yaw_stick, dt
            )

            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm

    def _process_stick(self):
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > cfg.RC_DEAD_ZONE_ROLL else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > cfg.RC_DEAD_ZONE_PITCH else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > cfg.RC_DEAD_ZONE_YAW else 0.0
        throttle_raw = self.rc_data["left_y"] if abs(self.rc_data["left_y"]) > cfg.RC_DEAD_ZONE_THROTTLE else 0.0

        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        throttle = np.clip((throttle_raw + 1.0) / 2.0 * 1000.0, 0.0, 1000.0)
        return throttle, roll_target, pitch_target, yaw_raw


def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()
    executor = MultiThreadedExecutor()
    executor.add_node(controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info("🛑 用户中断，停止飞控")
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
