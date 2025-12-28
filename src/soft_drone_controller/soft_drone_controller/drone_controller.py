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
            kp=cfg.PID_YAW_RATE["kp"] * 4.0,
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
            "stick_deadband": cfg.RC_DEAD_ZONE * 0.3,
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
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > self.state["stick_deadband"] else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > self.state["stick_deadband"] else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > self.state["stick_deadband"]*0.5 else 0.0
        throttle_raw = self.rc_data["left_y"]
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
