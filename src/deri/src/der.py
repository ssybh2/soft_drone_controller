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

# ===================== 四元数核心数学工具函数【新增】 =====================
def quat_mult(q1, q2):
    """四元数乘法：q1 * q2（[w,x,y,z]格式）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_inv(q):
    """四元数求逆（共轭，单位四元数逆=共轭）"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def eul2quat(roll, pitch, yaw):
    """欧拉角转四元数（ZYX顺序，弧度）"""
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

def quat2eul(w, x, y, z):
    """四元数转欧拉角（仅用于日志输出，控制逻辑不依赖）"""
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

def quat_error(q_target, q_current):
    """计算目标四元数与当前四元数的误差（轴角形式，返回[rx, ry, rz]误差向量）"""
    q_err = quat_mult(quat_inv(q_current), q_target, )
    # 转换为轴角误差（小角度近似，误差向量幅值=旋转角度，方向=旋转轴）
    angle = 2 * np.arctan2(np.linalg.norm(q_err[1:]), q_err[0])
    if np.linalg.norm(q_err[1:]) < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis

# 【保留原有】陀螺仪符号修正
def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro])

def pwm_to_dshot(pwm_val):
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))

# ===================== PID控制器（适配四元数误差向量） =====================
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
            self.d_term_sign = -1.0  # 速率环D项阻尼
        
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

# ===================== 主控制器（四元数重构版） =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")
        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        
        self.get_logger().info("✅ 控制器启动完成 - 四元数运算+Yaw随动+电机均衡")
        
    def _init_ros(self):
        qos_best_effort = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
        
        self.pub_imu_angle = self.create_publisher(Vector3, "/imu_angle", qos_reliable)
        self.pub_imu_gyro = self.create_publisher(Vector3, "/imu_gyro", qos_reliable)
        self.pub_dshot = self.create_publisher(WriteDSHOT, "/ecat/sn2228293/app3/write", 10)
        self.pub_torque_output = self.create_publisher(Vector3, "/torque_output", qos_reliable)
        
        self.sub_rc = self.create_subscription(ReadDJIRC, "/ecat/sn2228293/app1/read", self._rc_callback, qos_best_effort)
        self.sub_imu = self.create_subscription(Imu, "/ecat/sn2228293/app2/read", self._imu_callback, qos_best_effort)
        self.sub_filtered_acc = self.create_subscription(Vector3, "/filtered_angular_acceleration", self._filtered_acc_callback, qos_reliable)
        
        self.lock = threading.Lock()
        
    def _init_data(self):
        self.rc_data = {"left_y":0.0, "left_x":0.0, "right_x":0.0, "right_y":0.0, "left_switch":cfg.LOCK_SWITCH_VALUE}
        # 【四元数重构1：保存原始四元数，欧拉角仅用于日志】
        self.imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # 当前姿态四元数 [w,x,y,z]
            "gyro": np.array([0.0,0.0,0.0]),
            "roll":0.0, "pitch":0.0, "yaw":0.0  # 仅日志用
        }
        self.filtered_acc = np.array([0.0,0.0,0.0])
        
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0
        self.last_published_dshot = [1200]*4
        
        # 【四元数重构2：Yaw随动基于四元数的Yaw分量】
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 目标姿态四元数
        self.last_yaw_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 上一时刻Yaw四元数
        
    def _init_controllers(self):
        # 保留原有PID参数（修正力度不变）
        self.pid_roll_angle = ImprovedPID(
            kp=cfg.PID_ROLL_ANGLE["kp"] * 10.0,
            ki=cfg.PID_ROLL_ANGLE["ki"] * 0,
            kd=cfg.PID_ROLL_ANGLE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="roll"
        )
        self.pid_roll_rate = ImprovedPID(
            kp=cfg.PID_ROLL_RATE["kp"] * 10.0,
            ki=cfg.PID_ROLL_RATE["ki"] * 0,
            kd=cfg.PID_ROLL_RATE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="roll_rate"
        )
        
        self.pid_pitch_angle = ImprovedPID(
            kp=cfg.PID_PITCH_ANGLE["kp"] * 10.0,
            ki=cfg.PID_PITCH_ANGLE["ki"] * 0,
            kd=cfg.PID_PITCH_ANGLE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="pitch"
        )
        self.pid_pitch_rate = ImprovedPID(
            kp=cfg.PID_PITCH_RATE["kp"] * 10.0,
            ki=cfg.PID_PITCH_RATE["ki"] * 0,
            kd=cfg.PID_PITCH_RATE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,
            node=self,
            axis="pitch_rate"
        )
        
        self.pid_yaw_angle = ImprovedPID(
            kp=cfg.PID_YAW_ANGLE["kp"] * 4.0,
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
            "init_quat": None,  # 【四元数重构3：初始化姿态四元数】
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.8,
            "last_debug_time": 0.0
        }
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        self.yaw_stick_scale = 0.2
        self.yaw_dshot_gain = 0.6
        
    # ========== 回调函数（四元数重构） ==========
    def _rc_callback(self, msg):
        with self.lock:
            self.rc_data["left_y"] = msg.left_y
            self.rc_data["left_x"] = msg.left_x
            self.rc_data["right_x"] = msg.right_x
            self.rc_data["right_y"] = -msg.right_y
            self.rc_data["left_switch"] = msg.left_switch
            self.last_rc_time = self.get_clock().now().nanoseconds / 1e9
            self._update_arming_status()
            
    def _imu_callback(self, msg):
        with self.lock:
            # 【四元数重构4：直接读取IMU原始四元数，不再依赖欧拉角转换】
            current_quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
            # 修正Pitch/Yaw符号（适配硬件安装方向）
            current_quat = self._correct_quat_sign(current_quat)
            
            # 初始化姿态（仅首次）
            if self.state["init_quat"] is None:
                self.state["init_quat"] = current_quat.copy()
                self.state["initialized"] = True
                # 初始化Yaw随动：以上一时刻四元数为目标
                self.last_yaw_quat = self._extract_yaw_quat(current_quat)
                self.target_quat = current_quat.copy()
            
            # 计算相对初始化姿态的四元数（归零）
            rel_quat = quat_mult(current_quat, quat_inv(self.state["init_quat"]))
            # 仅用于日志：转欧拉角
            roll_zeroed, pitch_zeroed, current_yaw = quat2eul(*rel_quat)
            current_yaw = -current_yaw  # Yaw符号修正（日志用）
            
            # 【四元数重构5：更新Yaw随动目标（以上一时刻Yaw为期望）】
            current_yaw_quat = self._extract_yaw_quat(current_quat)
            self.target_quat = self._set_quat_yaw(rel_quat, self.last_yaw_quat)  # 目标Yaw=上一时刻Yaw
            self.last_yaw_quat = current_yaw_quat  # 更新上一时刻Yaw
            
            # 保存数据（四元数为主，欧拉角仅日志）
            self.imu_data["quat"] = rel_quat
            self.imu_data["roll"] = roll_zeroed
            self.imu_data["pitch"] = pitch_zeroed
            self.imu_data["yaw"] = current_yaw
            
            # 陀螺仪数据修正
            gyro_rotated = rotate_gyro_data(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
            self.imu_data["gyro"] = gyro_rotated
            
            # 日志输出（保留原有格式）
            self.get_logger().info(f"Roll角速率: {gyro_rotated[0]:.3f} | Roll角度: {np.rad2deg(roll_zeroed):.2f}°")

            # 发布IMU数据（欧拉角仅日志用）
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
    
    def _correct_quat_sign(self, quat):
        """修正四元数符号（适配硬件安装方向）"""
        # 对应原代码的pitch = -pitch和current_yaw = -current_yaw
        w, x, y, z = quat
        # Pitch符号修正：反转y分量
        y = -y
        # Yaw符号修正：反转z分量
        z = -z
        return np.array([w, x, y, z])
    
    def _extract_yaw_quat(self, quat):
        """从四元数中提取仅Yaw分量的四元数（Roll/Pitch=0）"""
        roll, pitch, yaw = quat2eul(*quat)
        return eul2quat(0, 0, yaw)
    
    def _set_quat_yaw(self, quat, yaw_quat):
        """保持Roll/Pitch不变，设置Yaw为目标四元数的Yaw"""
        roll, pitch, _ = quat2eul(*quat)
        _, _, yaw = quat2eul(*yaw_quat)
        return eul2quat(roll, pitch, yaw)
            
    def _filtered_acc_callback(self, msg):
        with self.lock:
            self.filtered_acc = np.array([msg.x, msg.y, msg.z])
            
    def _update_arming_status(self):
        current_switch = self.rc_data["left_switch"]
        if current_switch == cfg.LOCK_SWITCH_VALUE and self.state["armed"]:
            self.state["armed"] = False
            self.get_logger().info("🔒 已上锁")
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
            self._reset_all_controllers()
        elif current_switch in cfg.UNLOCK_SWITCH_VALUES and not self.state["armed"]:
            if not self.state["initialized"]:
                self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                return
            self.state["armed"] = True
            self.get_logger().info("🔓 已解锁 - 四元数运算+Yaw随动")
            self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
            self._reset_all_controllers()
            
    def _reset_all_controllers(self):
        for pid in [self.pid_roll_angle, self.pid_pitch_angle, self.pid_yaw_angle, 
                    self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        
    # ========== 核心控制循环（四元数重构） ==========
    def _control_loop(self):
        with self.lock:
            if not self._check_data_validity():
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
            if not self.state["armed"]:
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            dt = 1.0 / cfg.CONTROL_FREQ
            throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            
            # 【四元数重构6：生成目标姿态四元数（Roll/Pitch回平，Yaw随动）】
            # Roll/Pitch目标=0（回平）
            target_eul_roll = np.clip(roll_target, -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
            target_eul_pitch = np.clip(pitch_target, -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
            # Yaw目标=上一时刻值（随动）+摇杆速率指令
            yaw_rate_cmd = 0.0
            if abs(yaw_stick) > self.state["stick_deadband"]:
                yaw_rate_cmd = yaw_stick * self.yaw_stick_scale
            
            # PID计算（基于四元数误差）
            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(target_eul_roll, target_eul_pitch, yaw_rate_cmd, dt)
            # 电机混控（保留原有逻辑，修正力度不变）
            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            # 发布DSHOT
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
    
    def _process_stick(self):
        # 保留原有摇杆处理逻辑
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > self.state["stick_deadband"] else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > self.state["stick_deadband"] else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > self.state["stick_deadband"]*0.5 else 0.0
        throttle_raw = self.rc_data["left_y"]
        
        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG*3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG*3
        throttle = np.clip((throttle_raw + 1.0)/2.0 * 1000.0, 0.0, 1000.0)
        return throttle, roll_target, pitch_target, yaw_raw
    
    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        """基于四元数误差的PID控制（替代原欧拉角PID）"""
        current_quat = self.imu_data["quat"]
        gyro = self.imu_data["gyro"].copy()
        
        # 1. 生成目标姿态四元数（Roll/Pitch=目标值，Yaw=上一时刻值）
        _, _, current_yaw = quat2eul(*current_quat)
        target_quat = eul2quat(roll_target, pitch_target, self.yaw_setpoint if hasattr(self, 'yaw_setpoint') else current_yaw)
        
        # 2. 计算四元数误差（轴角形式）
        quat_err = quat_error(target_quat, current_quat)
        roll_err, pitch_err, yaw_err = quat_err[0], quat_err[1], quat_err[2]
        
        # 陀螺仪死区处理（保留原有）
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]
        
        # 3. 角度外环（输入四元数误差）
        roll_rate_sp = self.pid_roll_angle.update(0.0, roll_err, dt)  # 目标=0（回平）
        pitch_rate_sp = self.pid_pitch_angle.update(0.0, pitch_err, dt)
        yaw_rate_sp = self.pid_yaw_angle.update(0.0, yaw_err, dt) + yaw_rate_cmd  # 叠加摇杆速率
        
        # 4. 速率内环（保留原有逻辑）
        torque_roll = self.pid_roll_rate.update(roll_rate_sp, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(pitch_rate_sp, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(yaw_rate_sp, gyro[2], dt, self.filtered_acc[2])
        torque_roll = -torque_roll 
        torque_pitch = -torque_pitch
        torque_yaw = -torque_yaw
        # 力矩限幅（保留原有）
        torque_roll = np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_pitch = np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_yaw = np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"])
        
        # 发布扭矩数据（保留原有）
        torque_msg = Vector3()
        torque_msg.x = torque_roll
        torque_msg.y = torque_pitch
        torque_msg.z = torque_yaw
        self.pub_torque_output.publish(torque_msg)
        return torque_roll, torque_pitch, torque_yaw
    
    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        # 保留原有混控逻辑（修正力度参数不变）
        base = 1000.0 + throttle
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain
        
        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE

        smoothed_motors = np.array([motor1, motor2, motor3, motor4])
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
        # 保留原有数据有效性检查
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_rc_time > cfg.DATA_TIMEOUT or current_time - self.last_imu_time > cfg.DATA_TIMEOUT:
            if self.state["armed"]:
                self.state["armed"] = False
                self.get_logger().error("🔴 数据超时，强制上锁")
            return False
        return True
    
    def _publish_dshot(self, motor_pwm):
        # 保留原有DSHOT发布逻辑
        msg = WriteDSHOT()
        if isinstance(motor_pwm, (int, float)):
            dshot_val = pwm_to_dshot(motor_pwm)
            msg.channel1 = dshot_val
            msg.channel2 = dshot_val
            msg.channel3 = dshot_val
            msg.channel4 = dshot_val
            self.last_published_dshot = [dshot_val]*4
        elif isinstance(motor_pwm, (list, np.ndarray)):
            dshot = [pwm_to_dshot(p) for p in motor_pwm]
            msg.channel1 = dshot[0]
            msg.channel2 = dshot[1]
            msg.channel3 = dshot[2]
            msg.channel4 = dshot[3]
            self.last_published_dshot = dshot
        else:
            self.get_logger().error(f"❌ 不支持的motor_pwm类型: {type(motor_pwm)}")
            return
        
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ 发布DSHOT失败: {e}")
    
    def _publish_status(self):
        # 保留原有状态发布（欧拉角仅日志用）
        if not self.state["armed"]:
            return
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch"]))
            yaw_error = abs(np.rad2deg(self.yaw_setpoint - self.imu_data["yaw"]) if hasattr(self, 'yaw_setpoint') else 0.0)
            self.get_logger().info(
                f"📊 状态 - Roll误差:{roll_error:.2f}° | Pitch误差:{pitch_error:.2f}° | Yaw误差:{yaw_error:.3f}° | DSHOT:{dshot}"
            )
            self.state["last_debug_time"] = current_time
    
    def destroy_node(self):
        try:
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 销毁时发布上锁指令失败: {e}")
        super().destroy_node()

# ========== 主函数 ==========
def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    try:
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info("🛑 用户中断，停止控制器")
    except Exception as e:
        controller.get_logger().error(f"❌ 控制器运行出错: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()




#!/usr/bin/env python3
"""
无人机飞控主控制器 - 四元数版本 + 位置控制模式
左开关控制模式：
  1(上)：解锁 + 手动模式
  3(中)：解锁 + 位置控制模式
  2(下)：上锁
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

# ===================== 四元数核心数学工具函数 =====================
def quat_mult(q1, q2):
    """四元数乘法：q1 * q2（[w,x,y,z]格式）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_inv(q):
    """四元数求逆（共轭，单位四元数逆=共轭）"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def eul2quat(roll, pitch, yaw):
    """欧拉角转四元数（ZYX顺序，弧度）"""
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

def quat2eul(w, x, y, z):
    """四元数转欧拉角（仅用于日志输出，控制逻辑不依赖）"""
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

def quat_error(q_target, q_current):
    """计算目标四元数与当前四元数的误差（轴角形式，返回[rx, ry, rz]误差向量）"""
    q_err = quat_mult(quat_inv(q_current), q_target)
    # 转换为轴角误差（小角度近似，误差向量幅值=旋转角度，方向=旋转轴）
    angle = 2 * np.arctan2(np.linalg.norm(q_err[1:]), q_err[0])
    if np.linalg.norm(q_err[1:]) < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis

def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    """陀螺仪符号修正"""
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro])

def pwm_to_dshot(pwm_val):
    """PWM值转DSHOT值"""
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))

# ===================== PID控制器 =====================
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
            self.d_term_sign = -1.0  # 速率环D项阻尼
        
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
        """重置PID状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0

# ===================== 主控制器 =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")
        
        # 初始化各个模块
        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        
        # 创建定时器
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        
        self.get_logger().info("✅ 飞控启动完成 - 支持手动/位置控制模式")
        self.get_logger().info("🎮 控制模式：左开关1=手动，3=位置控制，2=上锁")
        
    def _init_ros(self):
        """初始化ROS话题"""
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST, 
            depth=5
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, 
            depth=10
        )
        
        # ========== 发布者 ==========
        self.pub_imu_angle = self.create_publisher(Vector3, "/imu_angle", qos_reliable)
        self.pub_imu_gyro = self.create_publisher(Vector3, "/imu_gyro", qos_reliable)
        self.pub_dshot = self.create_publisher(WriteDSHOT, "/ecat/sn2228293/app3/write", 10)
        self.pub_torque_output = self.create_publisher(Vector3, "/torque_output", qos_reliable)
        self.pub_control_status = self.create_publisher(Vector3, "/fc_control_status", qos_reliable)
        
        # ========== 订阅者 ==========
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
        
        # ========== 新增：位置指令订阅 ==========
        self.sub_pos_cmd = self.create_subscription(
            Vector3,
            "/attitude_position_cmd",  # 位置控制器发布的指令
            self._pos_cmd_callback,
            qos_reliable
        )
        
        # 线程锁
        self.lock = threading.Lock()
        
    def _init_data(self):
        """初始化数据存储"""
        # 遥控器数据
        self.rc_data = {
            "left_y": 0.0,
            "left_x": 0.0, 
            "right_x": 0.0,
            "right_y": 0.0,
            "left_switch": cfg.LOCK_SWITCH_VALUE  # 默认上锁状态
        }
        
        # IMU数据
        self.imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # 当前姿态四元数 [w,x,y,z]
            "gyro": np.array([0.0, 0.0, 0.0]),
            "roll": 0.0, 
            "pitch": 0.0, 
            "yaw": 0.0  # 仅日志用
        }
        
        # 滤波器数据
        self.filtered_acc = np.array([0.0, 0.0, 0.0])
        
        # 位置控制指令数据
        self.pos_cmd_data = {
            "roll": 0.0,
            "pitch": 0.0,
            "throttle": 1000.0
        }
        self.last_pos_cmd_time = 0.0
        self.pos_cmd_timeout = 0.2  # 200ms超时
        
        # 时间记录
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0
        self.last_published_dshot = [1200]*4
        
        # 四元数数据
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 目标姿态四元数
        self.last_yaw_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 上一时刻Yaw四元数
        
        # 控制模式
        self.control_mode = "MANUAL"  # MANUAL, POSITION
        
    def _init_controllers(self):
        """初始化PID控制器"""
        # 横滚角度PID
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
        
        # 横滚速率PID
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
        
        # 俯仰角度PID
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
        
        # 俯仰速率PID
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
        
        # 偏航角度PID
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
        
        # 偏航速率PID
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
        """初始化状态变量"""
        self.state = {
            "armed": False,
            "stick_deadband": cfg.RC_DEAD_ZONE * 0.3,
            "motor_outputs": np.array([1000.0]*4),
            "init_quat": None,  # 初始化姿态四元数
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.8,
            "last_debug_time": 0.0
        }
        
        # 陀螺仪死区
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        
        # Yaw控制参数
        self.yaw_stick_scale = 0.2
        self.yaw_dshot_gain = 0.6
        
    # ========== 回调函数 ==========
    def _rc_callback(self, msg):
        """遥控器数据回调"""
        with self.lock:
            self.rc_data["left_y"] = msg.left_y
            self.rc_data["left_x"] = msg.left_x
            self.rc_data["right_x"] = msg.right_x
            self.rc_data["right_y"] = -msg.right_y
            self.rc_data["left_switch"] = msg.left_switch
            self.last_rc_time = self.get_clock().now().nanoseconds / 1e9
            
            # 更新解锁状态和控制模式
            self._update_arming_and_mode()
            
    def _imu_callback(self, msg):
        """IMU数据回调"""
        with self.lock:
            # 读取IMU原始四元数
            current_quat = np.array([
                msg.orientation.w, 
                msg.orientation.x, 
                msg.orientation.y, 
                msg.orientation.z
            ])
            
            # 修正Pitch/Yaw符号（适配硬件安装方向）
            current_quat = self._correct_quat_sign(current_quat)
            
            # 初始化姿态（仅首次）
            if self.state["init_quat"] is None:
                self.state["init_quat"] = current_quat.copy()
                self.state["initialized"] = True
                # 初始化Yaw随动：以上一时刻四元数为目标
                self.last_yaw_quat = self._extract_yaw_quat(current_quat)
                self.target_quat = current_quat.copy()
                self.get_logger().info("📡 IMU初始化完成")
            
            # 计算相对初始化姿态的四元数（归零）
            rel_quat = quat_mult(current_quat, quat_inv(self.state["init_quat"]))
            
            # 仅用于日志：转欧拉角
            roll_zeroed, pitch_zeroed, current_yaw = quat2eul(*rel_quat)
            current_yaw = -current_yaw  # Yaw符号修正（日志用）
            
            # 更新Yaw随动目标（以上一时刻Yaw为期望）
            current_yaw_quat = self._extract_yaw_quat(current_quat)
            self.target_quat = self._set_quat_yaw(rel_quat, self.last_yaw_quat)  # 目标Yaw=上一时刻Yaw
            self.last_yaw_quat = current_yaw_quat  # 更新上一时刻Yaw
            
            # 保存数据（四元数为主，欧拉角仅日志）
            self.imu_data["quat"] = rel_quat
            self.imu_data["roll"] = roll_zeroed
            self.imu_data["pitch"] = pitch_zeroed
            self.imu_data["yaw"] = current_yaw
            
            # 陀螺仪数据修正
            gyro_rotated = rotate_gyro_data(
                msg.angular_velocity.x, 
                msg.angular_velocity.y, 
                msg.angular_velocity.z
            )
            self.imu_data["gyro"] = gyro_rotated
            
            # 日志输出
            if int(time.time() * 10) % 10 == 0:  # 约1秒一次
                self.get_logger().info(
                    f"📡 IMU数据: Roll={np.rad2deg(roll_zeroed):.3f}°, "
                    f"Pitch={np.rad2deg(pitch_zeroed):.3f}°, "
                    f"Yaw={np.rad2deg(current_yaw):.3f}°"
                )

            # 发布IMU数据（欧拉角仅日志用）
            angle_msg = Vector3()
            angle_msg.x = np.rad2deg(roll_zeroed)
            angle_msg.y = np.rad2deg(pitch_zeroed)
            angle_msg.z = np.rad2deg(current_yaw)
            self.pub_imu_angle.publish(angle_msg)
            
            # 发布陀螺仪数据
            gyro_msg = Vector3()
            gyro_msg.x = gyro_rotated[0]
            gyro_msg.y = gyro_rotated[1]
            gyro_msg.z = gyro_rotated[2]
            self.pub_imu_gyro.publish(gyro_msg)
            
            self.last_imu_time = self.get_clock().now().nanoseconds / 1e9
    
    def _filtered_acc_callback(self, msg):
        """滤波后的角加速度回调"""
        with self.lock:
            self.filtered_acc = np.array([msg.x, msg.y, msg.z])
            
    def _pos_cmd_callback(self, msg):
        """位置控制指令回调"""
        with self.lock:
            self.pos_cmd_data["roll"] = msg.x
            self.pos_cmd_data["pitch"] = msg.y
            self.pos_cmd_data["throttle"] = msg.z
            self.last_pos_cmd_time = self.get_clock().now().nanoseconds / 1e9
            
    def _correct_quat_sign(self, quat):
        """修正四元数符号（适配硬件安装方向）"""
        w, x, y, z = quat
        # Pitch符号修正：反转y分量
        y = -y
        # Yaw符号修正：反转z分量
        z = -z
        return np.array([w, x, y, z])
    
    def _extract_yaw_quat(self, quat):
        """从四元数中提取仅Yaw分量的四元数（Roll/Pitch=0）"""
        roll, pitch, yaw = quat2eul(*quat)
        return eul2quat(0, 0, yaw)
    
    def _set_quat_yaw(self, quat, yaw_quat):
        """保持Roll/Pitch不变，设置Yaw为目标四元数的Yaw"""
        roll, pitch, _ = quat2eul(*quat)
        _, _, yaw = quat2eul(*yaw_quat)
        return eul2quat(roll, pitch, yaw)
            
    def _update_arming_and_mode(self):
        """更新解锁状态和控制模式
        左开关：
          1(上)：解锁 + 手动模式
          3(中)：解锁 + 位置控制模式
          2(下)：上锁
        """
        current_switch = self.rc_data["left_switch"]
        
        # 位置2：上锁（下）
        if current_switch == 2:  # 上锁位置
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                self.get_logger().info("🔒 已上锁")
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                self._reset_all_controllers()
            return
            
        # 位置1：解锁 + 手动模式（上）
        elif current_switch == 1:  
            if not self.state["armed"]:
                if not self.state["initialized"]:
                    self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                    return
                self.state["armed"] = True
                self.control_mode = "MANUAL"
                self.get_logger().info("🔓 已解锁 - 手动模式")
                self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
                self._reset_all_controllers()
            elif self.control_mode != "MANUAL":  # 已解锁，切换到手动模式
                self.control_mode = "MANUAL"
                self.get_logger().info("🔄 切换到手动模式")
                self._reset_all_controllers()
            
        # 位置3：解锁 + 位置控制模式（中）
        elif current_switch == 3:  
            if not self.state["armed"]:
                if not self.state["initialized"]:
                    self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                    return
                self.state["armed"] = True
                self.control_mode = "POSITION"
                self.get_logger().info("🔓 已解锁 - 位置控制模式")
                self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
                self._reset_all_controllers()
            elif self.control_mode != "POSITION":  # 已解锁，切换到位置控制模式
                # 检查位置指令是否有效
                current_time = self.get_clock().now().nanoseconds / 1e9
                pos_cmd_valid = (current_time - self.last_pos_cmd_time) < self.pos_cmd_timeout
                
                if pos_cmd_valid:
                    self.control_mode = "POSITION"
                    self.get_logger().info("🎯 切换到位置控制模式")
                    self._reset_all_controllers()
                else:
                    self.get_logger().warn("⚠️ 位置指令无效，保持在手动模式")
                    self.control_mode = "MANUAL"
            
    def _reset_all_controllers(self):
        """重置所有PID控制器"""
        for pid in [self.pid_roll_angle, self.pid_pitch_angle, self.pid_yaw_angle, 
                    self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        self.get_logger().info("🔄 所有PID控制器已重置")
        
    # ========== 核心控制循环 ==========
    def _control_loop(self):
        """主控制循环（1000Hz）"""
        with self.lock:
            # 检查数据有效性
            if not self._check_data_validity():
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            # 检查是否解锁
            if not self.state["armed"]:
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            dt = 1.0 / cfg.CONTROL_FREQ
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # ========== 根据控制模式选择输入源 ==========
            if self.control_mode == "MANUAL":
                # 手动模式：完全使用遥控器
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                
            elif self.control_mode == "POSITION":
                # 位置控制模式：检查位置指令有效性
                pos_cmd_timeout = (current_time - self.last_pos_cmd_time) > self.pos_cmd_timeout
                
                if pos_cmd_timeout:
                    # 位置指令超时，自动切回手动模式
                    self.control_mode = "MANUAL"
                    self.get_logger().warn("⚠️ 位置指令超时，自动切换回手动模式")
                    throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                else:
                    # 使用位置控制器指令
                    throttle_raw = np.clip(self.pos_cmd_data["throttle"], 1000.0, 2000.0)
                    roll_target = np.clip(self.pos_cmd_data["roll"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    pitch_target = np.clip(self.pos_cmd_data["pitch"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    
                    # 将油门从1000-2000转换为0-1000（与_process_stick一致）
                    throttle = (throttle_raw - 1000.0) / 1000.0 * 1000.0
                    
                    # Yaw仍然使用遥控器控制（允许在位置控制时调整方向）
                    _, _, _, yaw_stick = self._process_stick()
                    
                    # 调试输出
                    if int(current_time * 100) % 100 == 0:  # 约1秒一次
                        self.get_logger().info(
                            f"🎯 位置控制指令: "
                            f"Roll={np.rad2deg(roll_target):.1f}°, "
                            f"Pitch={np.rad2deg(pitch_target):.1f}°, "
                            f"Throttle={throttle_raw:.0f}"
                        )
            else:
                # 默认手动模式
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            
            # ========== PID计算 ==========
            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(
                roll_target, pitch_target, yaw_stick, dt
            )
            
            # ========== 电机混控 ==========
            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            
            # ========== 发布DSHOT指令 ==========
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
            
            # ========== 发布控制状态 ==========
            self._publish_control_status()
    
    def _process_stick(self):
        """处理遥控器摇杆数据"""
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > self.state["stick_deadband"] else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > self.state["stick_deadband"] else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > self.state["stick_deadband"]*0.5 else 0.0
        throttle_raw = self.rc_data["left_y"]
        
        # 计算目标角度（限幅）
        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        
        # 计算油门（-1~1映射到0~1000）
        throttle = np.clip((throttle_raw + 1.0) / 2.0 * 1000.0, 0.0, 1000.0)
        
        return throttle, roll_target, pitch_target, yaw_raw
    
    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        """基于四元数误差的PID控制"""
        current_quat = self.imu_data["quat"]
        gyro = self.imu_data["gyro"].copy()
        
        # 1. 生成目标姿态四元数
        _, _, current_yaw = quat2eul(*current_quat)
        if not hasattr(self, 'yaw_setpoint'):
            self.yaw_setpoint = current_yaw
        target_quat = eul2quat(roll_target, pitch_target, self.yaw_setpoint)
        
        # 2. 计算四元数误差（轴角形式）
        quat_err = quat_error(target_quat, current_quat)
        roll_err, pitch_err, yaw_err = quat_err[0], quat_err[1], quat_err[2]
        
        # 3. 陀螺仪死区处理
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]
        
        # 4. 角度外环（输入四元数误差）
        roll_rate_sp = self.pid_roll_angle.update(0.0, roll_err, dt)  # 目标=0（回平）
        pitch_rate_sp = self.pid_pitch_angle.update(0.0, pitch_err, dt)
        yaw_rate_sp = self.pid_yaw_angle.update(0.0, yaw_err, dt) + yaw_rate_cmd  # 叠加摇杆速率
        
        # 5. 速率内环
        torque_roll = self.pid_roll_rate.update(roll_rate_sp, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(pitch_rate_sp, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(yaw_rate_sp, gyro[2], dt, self.filtered_acc[2])
        
        # 6. 力矩符号修正
        torque_roll = -torque_roll 
        torque_pitch = -torque_pitch
        torque_yaw = -torque_yaw
        
        # 7. 力矩限幅
        torque_roll = np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_pitch = np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_yaw = np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"])
        
        # 8. 发布扭矩数据
        torque_msg = Vector3()
        torque_msg.x = torque_roll
        torque_msg.y = torque_pitch
        torque_msg.z = torque_yaw
        self.pub_torque_output.publish(torque_msg)
        
        return torque_roll, torque_pitch, torque_yaw
    
    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        """电机混控"""
        # 基础油门
        base = 1000.0 + throttle
        
        # Yaw扭矩放大
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain
        
        # 混控计算
        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE

        # 检查电机差值是否过大
        motor_max = np.max([motor1, motor2, motor3, motor4])
        motor_min = np.min([motor1, motor2, motor3, motor4])
        if motor_max - motor_min > 600:
            scale = 600 / (motor_max - motor_min) if motor_max != motor_min else 1.0
            motor_avg = (motor1 + motor2 + motor3 + motor4) / 4
            motor1 = motor_avg + (motor1 - motor_avg) * scale
            motor2 = motor_avg + (motor2 - motor_avg) * scale
            motor3 = motor_avg + (motor3 - motor_avg) * scale
            motor4 = motor_avg + (motor4 - motor_avg) * scale

        # 平滑滤波
        alpha = 0.18
        smoothed_motors = alpha * np.array([motor1, motor2, motor3, motor4]) + (1 - alpha) * self.state["motor_outputs"]
        
        return smoothed_motors
    
    def _check_data_validity(self):
        """检查数据有效性"""
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
        """发布DSHOT指令"""
        msg = WriteDSHOT()
        
        if isinstance(motor_pwm, (int, float)):
            # 单个值：所有电机相同
            dshot_val = pwm_to_dshot(motor_pwm)
            msg.channel1 = dshot_val
            msg.channel2 = dshot_val
            msg.channel3 = dshot_val
            msg.channel4 = dshot_val
            self.last_published_dshot = [dshot_val]*4
            
        elif isinstance(motor_pwm, (list, np.ndarray)):
            # 数组：每个电机单独设置
            if len(motor_pwm) >= 4:
                dshot = [pwm_to_dshot(p) for p in motor_pwm[:4]]
                msg.channel1 = dshot[0]
                msg.channel2 = dshot[1]
                msg.channel3 = dshot[2]
                msg.channel4 = dshot[3]
                self.last_published_dshot = dshot
            else:
                self.get_logger().error(f"❌ 电机PWM数组长度不足: {len(motor_pwm)}")
                return
        else:
            self.get_logger().error(f"❌ 不支持的motor_pwm类型: {type(motor_pwm)}")
            return
        
        # 发布消息
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ 发布DSHOT失败: {e}")
    
    def _publish_control_status(self):
        """发布控制状态"""
        status_msg = Vector3()
        
        # 控制模式编码
        if self.control_mode == "MANUAL":
            status_msg.x = 1.0
        elif self.control_mode == "POSITION":
            status_msg.x = 2.0
        else:
            status_msg.x = 0.0
            
        # 解锁状态
        status_msg.y = 1.0 if self.state["armed"] else 0.0
        
        # 开关位置
        status_msg.z = float(self.rc_data.get("left_switch", 0))
        
        self.pub_control_status.publish(status_msg)
    
    def _publish_status(self):
        """定期发布状态信息"""
        if not self.state["armed"]:
            return
            
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch"]))
            
            # 计算Yaw误差
            if hasattr(self, 'yaw_setpoint'):
                yaw_error = abs(np.rad2deg(self.yaw_setpoint - self.imu_data["yaw"]))
            else:
                yaw_error = 0.0
            
            # 状态日志
            self.get_logger().info(
                f"📊 飞行状态 | "
                f"模式:{self.control_mode} | "
                f"Roll误差:{roll_error:.2f}° | "
                f"Pitch误差:{pitch_error:.2f}° | "
                f"Yaw误差:{yaw_error:.2f}° | "
                f"DSHOT:{dshot}"
            )
            
            # 更新调试时间
            self.state["last_debug_time"] = current_time
    
    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info("🛑 正在关闭飞控...")
        
        # 发布上锁指令
        try:
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
            time.sleep(0.05)  # 确保指令发出
        except Exception as e:
            self.get_logger().warn(f"⚠️ 销毁时发布上锁指令失败: {e}")
            
        # 调用父类销毁方法
        super().destroy_node()

# ========== 主函数 ==========
def main(args=None):
    """主函数：启动飞控节点"""
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


position control


#!/usr/bin/env python3
"""
无人机位置控制器（简化版）
将所有参数移动到配置文件，代码更简洁
"""

import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
import time

class DronePositionController(Node):
    def __init__(self):
        """位置控制器初始化"""
        super().__init__("drone_position_controller")
        
        # 导入配置参数
        from soft_drone_controller.config import controller_params as cfg
        
        # 保存配置引用
        self.cfg = cfg
        
        # 初始化ROS话题
        self._init_ros_topics()
        
        # 初始化数据存储
        self._init_data()
        
        # 初始化PID状态
        self._init_pid_state()
        
        # 创建控制循环定时器
        control_interval = 1.0 / self.cfg.POSITION_CONTROL_FREQ
        self.control_timer = self.create_timer(control_interval, self._control_loop)
        
        # 状态发布定时器（2Hz）
        self.status_timer = self.create_timer(0.5, self._publish_status)
        
        self.get_logger().info("✅ 无人机位置控制器启动完成")
        self.get_logger().info(f"📡 控制频率: {self.cfg.POSITION_CONTROL_FREQ}Hz")
        
    def _init_ros_topics(self):
        """初始化ROS话题"""
        qos_reliable = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            depth=10
        )
        qos_best_effort = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            depth=5
        )
        
        # 订阅动捕系统位置
        self.sub_mocap = self.create_subscription(
            PoseStamped,
            "/Tracker0/pose",
            self._mocap_callback,
            qos_best_effort
        )
        
        # 订阅目标位置
        self.sub_target = self.create_subscription(
            PoseStamped,
            "/drone_target_pose",
            self._target_callback,
            qos_reliable
        )
        
        # 发布姿态指令给飞控
        self.pub_attitude_cmd = self.create_publisher(
            Vector3,
            "/attitude_position_cmd",
            qos_reliable
        )
        
        # 发布调试信息
        self.pub_debug = self.create_publisher(
            Vector3,
            "/position_debug",
            qos_reliable
        )
        
    def _init_data(self):
        """初始化数据存储"""
        self.current_pos = None
        self.target_pos = np.array([1.19, -2.03, 0.4])  # 默认目标
        self.filtered_pos = None
        self.filtered_vel = None
        self.last_mocap_time = 0
        self.last_control_time = 0
        self.mocap_active = False
        self.control_count = 0
        
    def _init_pid_state(self):
        """初始化PID状态"""
        self.error_integral_xy = np.array([0.0, 0.0])
        self.error_integral_z = 0.0
        self.prev_error_xy = np.array([0.0, 0.0])
        self.prev_error_z = 0.0
        self.last_time = None
        
    def _mocap_callback(self, msg):
        """动捕数据回调"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # 提取位置
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # 首次初始化
        if self.current_pos is None:
            self.current_pos = pos
            self.filtered_pos = pos.copy()
            self.filtered_vel = np.zeros(3)
            self.last_time = current_time
            self.mocap_active = True
            self.get_logger().info(f"🎯 动捕数据就绪: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m")
            return
        
        # 计算时间差
        dt = current_time - self.last_time
        if dt < 0.001 or dt > 0.1:
            dt = 1.0 / self.cfg.POSITION_CONTROL_FREQ
        
        # 计算速度
        vel = (pos - self.current_pos) / dt
        
        # 滤波处理
        if self.filtered_pos is not None:
            self.filtered_pos = self.cfg.POSITION_FILTER_ALPHA_POS * pos + (1 - self.cfg.POSITION_FILTER_ALPHA_POS) * self.filtered_pos
            self.filtered_vel = self.cfg.POSITION_FILTER_ALPHA_VEL * vel + (1 - self.cfg.POSITION_FILTER_ALPHA_VEL) * self.filtered_vel
        
        # 更新数据
        self.current_pos = pos
        self.last_mocap_time = current_time
        self.last_time = current_time
        self.mocap_active = True
        
    def _target_callback(self, msg):
        """目标位置回调"""
        old_target = self.target_pos.copy()
        self.target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # 如果目标变化较大，重置积分器
        if np.linalg.norm(self.target_pos - old_target) > 0.5 and self.target_pos[2] > 0.1:
            self.error_integral_xy = np.array([0.0, 0.0])
            self.error_integral_z = 0.0
            self.get_logger().info("🔄 目标位置变化较大，重置PID积分")
            
        self.get_logger().info(
            f"📌 目标位置: X={self.target_pos[0]:.2f}m, Y={self.target_pos[1]:.2f}m, Z={self.target_pos[2]:.2f}m"
        )
        
    def _position_pid_control(self, dt):
        """位置PID控制核心算法"""
        if dt <= 0 or self.current_pos is None or self.filtered_pos is None:
            return 0.0, 0.0, self.cfg.POSITION_BASE_THROTTLE
            
        # 使用滤波后的位置
        current_pos = self.filtered_pos
        
        # 计算位置误差
        error_xy = self.target_pos[:2] - current_pos[:2]
        error_z = self.target_pos[2] - current_pos[2]
        
        # 死区处理
        error_norm_xy = np.linalg.norm(error_xy)
        if error_norm_xy < self.cfg.POSITION_DEADZONE_XY:
            error_xy = np.zeros(2)
        if abs(error_z) < self.cfg.POSITION_DEADZONE_Z:
            error_z = 0.0
            
        # ========== XY轴PID控制 ==========
        self.error_integral_xy += error_xy * dt
        self.error_integral_xy = np.clip(
            self.error_integral_xy, 
            -self.cfg.POSITION_XY_INT_LIMIT, 
            self.cfg.POSITION_XY_INT_LIMIT
        )
        
        # 微分项
        if self.filtered_vel is not None:
            derivative_xy = -self.filtered_vel[:2]
        else:
            derivative_xy = np.zeros(2)
            
        # PID计算
        output_xy = (
            self.cfg.POSITION_XY_KP * error_xy + 
            self.cfg.POSITION_XY_KI * self.error_integral_xy + 
            self.cfg.POSITION_XY_KD * derivative_xy
        )
        
        # 输出限幅
        output_norm = np.linalg.norm(output_xy)
        if output_norm > self.cfg.POSITION_XY_MAX_ANGLE:
            output_xy = output_xy / output_norm * self.cfg.POSITION_XY_MAX_ANGLE
            
        # ========== Z轴PID控制 ==========
        self.error_integral_z += error_z * dt
        self.error_integral_z = np.clip(
            self.error_integral_z, 
            -self.cfg.POSITION_Z_INT_LIMIT, 
            self.cfg.POSITION_Z_INT_LIMIT
        )
        
        # 微分项
        if self.filtered_vel is not None:
            derivative_z = -self.filtered_vel[2]
        else:
            derivative_z = 0.0
            
        # PID计算（油门增量）
        throttle_increment = (
            self.cfg.POSITION_Z_KP * error_z + 
            self.cfg.POSITION_Z_KI * self.error_integral_z + 
            self.cfg.POSITION_Z_KD * derivative_z
        )
        
        # 油门限幅
        throttle_increment = np.clip(
            throttle_increment, 
            -self.cfg.POSITION_Z_THROTTLE_RANGE, 
            self.cfg.POSITION_Z_THROTTLE_RANGE
        )
        
        # 最终油门
        throttle_output = self.cfg.POSITION_BASE_THROTTLE + throttle_increment
        throttle_output = np.clip(throttle_output, 1000.0, 2000.0)
        
        # 保存误差
        self.prev_error_xy = error_xy
        self.prev_error_z = error_z
        
        # 返回控制指令
        # 注意坐标系：动捕X→无人机前向(pitch)，动捕Y→无人机左向(roll)
        roll_cmd = float(output_xy[1])   # Y误差→roll指令
        pitch_cmd = float(output_xy[0])  # X误差→pitch指令
        
        return roll_cmd, pitch_cmd, throttle_output
        
    def _control_loop(self):
        """主控制循环"""
        self.control_count += 1
        
        # 检查动捕数据有效性
        if not self._check_mocap_valid():
            return
            
        # 计算时间间隔
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time is None:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        if dt < 0.001:
            return
            
        # 位置PID计算
        roll_cmd, pitch_cmd, throttle_cmd = self._position_pid_control(dt)
        
        # 发布控制指令
        cmd_msg = Vector3()
        cmd_msg.x = roll_cmd
        cmd_msg.y = pitch_cmd
        cmd_msg.z = throttle_cmd
        self.pub_attitude_cmd.publish(cmd_msg)
        
        # 发布调试信息
        if self.current_pos is not None:
            debug_msg = Vector3()
            if self.target_pos is not None:
                debug_msg.x = self.target_pos[0] - self.current_pos[0]
                debug_msg.y = self.target_pos[1] - self.current_pos[1]
                debug_msg.z = self.target_pos[2] - self.current_pos[2]
            self.pub_debug.publish(debug_msg)
            
        # 更新时间
        self.last_time = current_time
        
    def _check_mocap_valid(self):
        """检查动捕数据有效性"""
        if self.current_pos is None:
            if self.control_count % 100 == 0:
                self.get_logger().warn("⚠️ 等待动捕数据...")
            return False
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # 检查超时
        if current_time - self.last_mocap_time > self.cfg.POSITION_MOCAP_TIMEOUT:
            if self.mocap_active:
                self.mocap_active = False
                self.get_logger().error("❌ 动捕数据超时，停止控制输出")
            return False
            
        # 检查高度合理性
        if self.current_pos[2] < 0:
            self.get_logger().warn(f"⚠️ 检测到异常高度: {self.current_pos[2]:.2f}m")
            return False
            
        return True
        
    def _publish_status(self):
        """发布控制器状态"""
        if self.current_pos is not None:
            self.get_logger().info(
                f"📊 位置控制状态 | "
                f"动捕: {'✅' if self.mocap_active else '❌'} | "
                f"位置: [{self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}, {self.current_pos[2]:.2f}]m"
            )
            
    def reset_controller(self):
        """重置控制器"""
        self.error_integral_xy = np.array([0.0, 0.0])
        self.error_integral_z = 0.0
        self.mocap_active = False
        self.get_logger().info("🔄 位置控制器已重置")
        
    def destroy_node(self):
        """节点销毁"""
        self.get_logger().info("🛑 正在关闭位置控制器...")
        super().destroy_node()

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        controller = DronePositionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("🛑 用户中断位置控制器")
    except Exception as e:
        controller.get_logger().error(f"❌ 位置控制器运行异常: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()



FC
#!/usr/bin/env python3
"""
无人机飞控主控制器 - 四元数版本 + 位置控制模式
左开关控制模式：
  1(上)：解锁 + 手动模式
  3(中)：解锁 + 位置控制模式
  2(下)：上锁
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

# ===================== 四元数核心数学工具函数 =====================
def quat_mult(q1, q2):
    """四元数乘法：q1 * q2（[w,x,y,z]格式）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_inv(q):
    """四元数求逆（共轭，单位四元数逆=共轭）"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def eul2quat(roll, pitch, yaw):
    """欧拉角转四元数（ZYX顺序，弧度）"""
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

def quat2eul(w, x, y, z):
    """四元数转欧拉角（仅用于日志输出，控制逻辑不依赖）"""
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

def quat_error(q_target, q_current):
    """计算目标四元数与当前四元数的误差（轴角形式，返回[rx, ry, rz]误差向量）"""
    q_err = quat_mult(quat_inv(q_current), q_target)
    # 转换为轴角误差（小角度近似，误差向量幅值=旋转角度，方向=旋转轴）
    angle = 2 * np.arctan2(np.linalg.norm(q_err[1:]), q_err[0])
    if np.linalg.norm(q_err[1:]) < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis

def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    """陀螺仪符号修正"""
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro])

def pwm_to_dshot(pwm_val):
    """PWM值转DSHOT值"""
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))

# ===================== PID控制器 =====================
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
            self.d_term_sign = -1.0  # 速率环D项阻尼
        
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
        """重置PID状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0

# ===================== 主控制器 =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")
        
        # 初始化各个模块
        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        
        # 创建定时器
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        
        self.get_logger().info("✅ 飞控启动完成 - 支持手动/位置控制模式")
        self.get_logger().info("🎮 控制模式：左开关1=手动，3=位置控制，2=上锁")
        
    def _init_ros(self):
        """初始化ROS话题"""
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST, 
            depth=5
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, 
            depth=10
        )
        
        # ========== 发布者 ==========
        self.pub_imu_angle = self.create_publisher(Vector3, "/imu_angle", qos_reliable)
        self.pub_imu_gyro = self.create_publisher(Vector3, "/imu_gyro", qos_reliable)
        self.pub_dshot = self.create_publisher(WriteDSHOT, "/ecat/sn2228293/app3/write", 10)
        self.pub_torque_output = self.create_publisher(Vector3, "/torque_output", qos_reliable)
        self.pub_control_status = self.create_publisher(Vector3, "/fc_control_status", qos_reliable)
        
        # ========== 订阅者 ==========
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
        
        # ========== 新增：位置指令订阅 ==========
        self.sub_pos_cmd = self.create_subscription(
            Vector3,
            "/attitude_position_cmd",  # 位置控制器发布的指令
            self._pos_cmd_callback,
            qos_reliable
        )
        
        # 线程锁
        self.lock = threading.Lock()
        
    def _init_data(self):
        """初始化数据存储"""
        # 遥控器数据
        self.rc_data = {
            "left_y": 0.0,
            "left_x": 0.0, 
            "right_x": 0.0,
            "right_y": 0.0,
            "left_switch": cfg.LOCK_SWITCH_VALUE  # 默认上锁状态
        }
        
        # IMU数据
        self.imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # 当前姿态四元数 [w,x,y,z]
            "gyro": np.array([0.0, 0.0, 0.0]),
            "roll": 0.0, 
            "pitch": 0.0, 
            "yaw": 0.0  # 仅日志用
        }
        
        # 滤波器数据
        self.filtered_acc = np.array([0.0, 0.0, 0.0])
        
        # 位置控制指令数据
        self.pos_cmd_data = {
            "roll": 0.0,
            "pitch": 0.0,
            "throttle": 1000.0
        }
        self.last_pos_cmd_time = 0.0
        self.pos_cmd_timeout = 0.2  # 200ms超时
        
        # 时间记录
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0
        self.last_published_dshot = [1200]*4
        
        # 四元数数据
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 目标姿态四元数
        self.last_yaw_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 上一时刻Yaw四元数
        
        # 控制模式
        self.control_mode = "MANUAL"  # MANUAL, POSITION
        
    def _init_controllers(self):
        """初始化PID控制器"""
        # 横滚角度PID
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
        
        # 横滚速率PID
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
        
        # 俯仰角度PID
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
        
        # 俯仰速率PID
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
        
        # 偏航角度PID
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
        
        # 偏航速率PID
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
        """初始化状态变量"""
        self.state = {
            "armed": False,
            "stick_deadband": cfg.RC_DEAD_ZONE * 0.3,
            "motor_outputs": np.array([1000.0]*4),
            "init_quat": None,  # 初始化姿态四元数
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.8,
            "last_debug_time": 0.0
        }
        
        # 陀螺仪死区
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        
        # Yaw控制参数
        self.yaw_stick_scale = 0.2
        self.yaw_dshot_gain = 0.6
        
    # ========== 回调函数 ==========
    def _rc_callback(self, msg):
        """遥控器数据回调"""
        with self.lock:
            self.rc_data["left_y"] = msg.left_y
            self.rc_data["left_x"] = msg.left_x
            self.rc_data["right_x"] = msg.right_x
            self.rc_data["right_y"] = -msg.right_y
            self.rc_data["left_switch"] = msg.left_switch
            self.last_rc_time = self.get_clock().now().nanoseconds / 1e9
            
            # 更新解锁状态和控制模式
            self._update_arming_and_mode()
            
    def _imu_callback(self, msg):
        """IMU数据回调"""
        with self.lock:
            # 读取IMU原始四元数
            current_quat = np.array([
                msg.orientation.w, 
                msg.orientation.x, 
                msg.orientation.y, 
                msg.orientation.z
            ])
            
            # 修正Pitch/Yaw符号（适配硬件安装方向）
            current_quat = self._correct_quat_sign(current_quat)
            
            # 初始化姿态（仅首次）
            if self.state["init_quat"] is None:
                self.state["init_quat"] = current_quat.copy()
                self.state["initialized"] = True
                # 初始化Yaw随动：以上一时刻四元数为目标
                self.last_yaw_quat = self._extract_yaw_quat(current_quat)
                self.target_quat = current_quat.copy()
                self.get_logger().info("📡 IMU初始化完成")
            
            # 计算相对初始化姿态的四元数（归零）
            rel_quat = quat_mult(current_quat, quat_inv(self.state["init_quat"]))
            
            # 仅用于日志：转欧拉角
            roll_zeroed, pitch_zeroed, current_yaw = quat2eul(*rel_quat)
            current_yaw = -current_yaw  # Yaw符号修正（日志用）
            
            # 更新Yaw随动目标（以上一时刻Yaw为期望）
            current_yaw_quat = self._extract_yaw_quat(current_quat)
            self.target_quat = self._set_quat_yaw(rel_quat, self.last_yaw_quat)  # 目标Yaw=上一时刻Yaw
            self.last_yaw_quat = current_yaw_quat  # 更新上一时刻Yaw
            
            # 保存数据（四元数为主，欧拉角仅日志）
            self.imu_data["quat"] = rel_quat
            self.imu_data["roll"] = roll_zeroed
            self.imu_data["pitch"] = pitch_zeroed
            self.imu_data["yaw"] = current_yaw
            
            # 陀螺仪数据修正
            gyro_rotated = rotate_gyro_data(
                msg.angular_velocity.x, 
                msg.angular_velocity.y, 
                msg.angular_velocity.z
            )
            self.imu_data["gyro"] = gyro_rotated
            
            # 日志输出
            if int(time.time() * 10) % 10 == 0:  # 约1秒一次
                self.get_logger().info(
                    f"📡 IMU数据: Roll={np.rad2deg(roll_zeroed):.3f}°, "
                    f"Pitch={np.rad2deg(pitch_zeroed):.3f}°, "
                    f"Yaw={np.rad2deg(current_yaw):.3f}°"
                )

            # 发布IMU数据（欧拉角仅日志用）
            angle_msg = Vector3()
            angle_msg.x = np.rad2deg(roll_zeroed)
            angle_msg.y = np.rad2deg(pitch_zeroed)
            angle_msg.z = np.rad2deg(current_yaw)
            self.pub_imu_angle.publish(angle_msg)
            
            # 发布陀螺仪数据
            gyro_msg = Vector3()
            gyro_msg.x = gyro_rotated[0]
            gyro_msg.y = gyro_rotated[1]
            gyro_msg.z = gyro_rotated[2]
            self.pub_imu_gyro.publish(gyro_msg)
            
            self.last_imu_time = self.get_clock().now().nanoseconds / 1e9
    
    def _filtered_acc_callback(self, msg):
        """滤波后的角加速度回调"""
        with self.lock:
            self.filtered_acc = np.array([msg.x, msg.y, msg.z])
            
    def _pos_cmd_callback(self, msg):
        """位置控制指令回调"""
        with self.lock:
            self.pos_cmd_data["roll"] = msg.x
            self.pos_cmd_data["pitch"] = msg.y
            self.pos_cmd_data["throttle"] = msg.z
            self.last_pos_cmd_time = self.get_clock().now().nanoseconds / 1e9
            
    def _correct_quat_sign(self, quat):
        """修正四元数符号（适配硬件安装方向）"""
        w, x, y, z = quat
        # Pitch符号修正：反转y分量
        y = -y
        # Yaw符号修正：反转z分量
        z = -z
        return np.array([w, x, y, z])
    
    def _extract_yaw_quat(self, quat):
        """从四元数中提取仅Yaw分量的四元数（Roll/Pitch=0）"""
        roll, pitch, yaw = quat2eul(*quat)
        return eul2quat(0, 0, yaw)
    
    def _set_quat_yaw(self, quat, yaw_quat):
        """保持Roll/Pitch不变，设置Yaw为目标四元数的Yaw"""
        roll, pitch, _ = quat2eul(*quat)
        _, _, yaw = quat2eul(*yaw_quat)
        return eul2quat(roll, pitch, yaw)
            
    def _update_arming_and_mode(self):
        """更新解锁状态和控制模式
        左开关：
          1(上)：解锁 + 手动模式
          3(中)：解锁 + 位置控制模式
          2(下)：上锁
        """
        current_switch = self.rc_data["left_switch"]
        
        # 位置2：上锁（下）
        if current_switch == 2:  # 上锁位置
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                self.get_logger().info("🔒 已上锁")
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                self._reset_all_controllers()
            return
            
        # 位置1：解锁 + 手动模式（上）
        elif current_switch == 1:  
            if not self.state["armed"]:
                if not self.state["initialized"]:
                    self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                    return
                self.state["armed"] = True
                self.control_mode = "MANUAL"
                self.get_logger().info("🔓 已解锁 - 手动模式")
                self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
                self._reset_all_controllers()
            elif self.control_mode != "MANUAL":  # 已解锁，切换到手动模式
                self.control_mode = "MANUAL"
                self.get_logger().info("🔄 切换到手动模式")
                self._reset_all_controllers()
            
        # 位置3：解锁 + 位置控制模式（中）
        elif current_switch == 3:  
            if not self.state["armed"]:
                if not self.state["initialized"]:
                    self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                    return
                self.state["armed"] = True
                self.control_mode = "POSITION"
                self.get_logger().info("🔓 已解锁 - 位置控制模式")
                self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
                self._reset_all_controllers()
            elif self.control_mode != "POSITION":  # 已解锁，切换到位置控制模式
                # 检查位置指令是否有效
                current_time = self.get_clock().now().nanoseconds / 1e9
                pos_cmd_valid = (current_time - self.last_pos_cmd_time) < self.pos_cmd_timeout
                
                if pos_cmd_valid:
                    self.control_mode = "POSITION"
                    self.get_logger().info("🎯 切换到位置控制模式")
                    self._reset_all_controllers()
                else:
                    self.get_logger().warn("⚠️ 位置指令无效，保持在手动模式")
                    self.control_mode = "MANUAL"
            
    def _reset_all_controllers(self):
        """重置所有PID控制器"""
        for pid in [self.pid_roll_angle, self.pid_pitch_angle, self.pid_yaw_angle, 
                    self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        self.get_logger().info("🔄 所有PID控制器已重置")
        
    # ========== 核心控制循环 ==========
    def _control_loop(self):
        """主控制循环（1000Hz）"""
        with self.lock:
            # 检查数据有效性
            if not self._check_data_validity():
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            # 检查是否解锁
            if not self.state["armed"]:
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            dt = 1.0 / cfg.CONTROL_FREQ
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # ========== 根据控制模式选择输入源 ==========
            if self.control_mode == "MANUAL":
                # 手动模式：完全使用遥控器
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                
            elif self.control_mode == "POSITION":
                # 位置控制模式：检查位置指令有效性
                pos_cmd_timeout = (current_time - self.last_pos_cmd_time) > self.pos_cmd_timeout
                
                if pos_cmd_timeout:
                    # 位置指令超时，自动切回手动模式
                    self.control_mode = "MANUAL"
                    self.get_logger().warn("⚠️ 位置指令超时，自动切换回手动模式")
                    throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                else:
                    # 使用位置控制器指令
                    throttle_raw = np.clip(self.pos_cmd_data["throttle"], 1000.0, 2000.0)
                    roll_target = np.clip(self.pos_cmd_data["roll"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    pitch_target = np.clip(self.pos_cmd_data["pitch"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    
                    # 将油门从1000-2000转换为0-1000（与_process_stick一致）
                    throttle = (throttle_raw - 1000.0) / 1000.0 * 1000.0
                    
                    # Yaw仍然使用遥控器控制（允许在位置控制时调整方向）
                    _, _, _, yaw_stick = self._process_stick()
                    
                    # 调试输出
                    if int(current_time * 100) % 100 == 0:  # 约1秒一次
                        self.get_logger().info(
                            f"🎯 位置控制指令: "
                            f"Roll={np.rad2deg(roll_target):.1f}°, "
                            f"Pitch={np.rad2deg(pitch_target):.1f}°, "
                            f"Throttle={throttle_raw:.0f}"
                        )
            else:
                # 默认手动模式
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            
            # ========== PID计算 ==========
            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(
                roll_target, pitch_target, yaw_stick, dt
            )
            
            # ========== 电机混控 ==========
            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            
            # ========== 发布DSHOT指令 ==========
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
            
            # ========== 发布控制状态 ==========
            self._publish_control_status()
    
    def _process_stick(self):
        """处理遥控器摇杆数据"""
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > self.state["stick_deadband"] else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > self.state["stick_deadband"] else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > self.state["stick_deadband"]*0.5 else 0.0
        throttle_raw = self.rc_data["left_y"]
        
        # 计算目标角度（限幅）
        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        
        # 计算油门（-1~1映射到0~1000）
        throttle = np.clip((throttle_raw + 1.0) / 2.0 * 1000.0, 0.0, 1000.0)
        
        return throttle, roll_target, pitch_target, yaw_raw
    
    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        """基于四元数误差的PID控制 - 修改为Yaw阻尼模式"""
        current_quat = self.imu_data["quat"]
        gyro = self.imu_data["gyro"].copy()
    
        # 【修改1】生成目标姿态四元数（Yaw使用上一时刻值 + 摇杆指令）
        # 获取当前Yaw
        roll_current, pitch_current, current_yaw = quat2eul(*current_quat)
    
        # 使用上一控制周期的Yaw作为目标（实现阻尼效果）
        if not hasattr(self, 'prev_yaw_for_damping'):
            self.prev_yaw_for_damping = current_yaw
    
        # 【修改2】Yaw目标 = 上一时刻Yaw + 摇杆速率积分
        # 摇杆指令是角速度，需要积分得到角度变化
        yaw_target = self.prev_yaw_for_damping + yaw_rate_cmd * dt
    
        # 构造目标四元数
        target_quat = eul2quat(roll_target, pitch_target, yaw_target)
    
        # 保存当前Yaw作为下一时刻的"上一时刻Yaw"
        self.prev_yaw_for_damping = current_yaw
    
        # 计算四元数误差（轴角形式）
        quat_err = quat_error(target_quat, current_quat)
        roll_err, pitch_err, yaw_err = quat_err[0], quat_err[1], quat_err[2]
    
        # 陀螺仪死区处理
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]
    
        # 角度外环（Roll/Pitch使用回平逻辑，Yaw使用阻尼逻辑）
        roll_rate_sp = self.pid_roll_angle.update(0.0, roll_err, dt)
        pitch_rate_sp = self.pid_pitch_angle.update(0.0, pitch_err, dt)
    
        # 【修改3】Yaw角度环：期望值=0（即不期望有固定位置，只期望跟随上一时刻）
        # 但实际上，由于target_quat使用了上一时刻的Yaw，yaw_err已经反映了这个跟随误差
        yaw_rate_sp = self.pid_yaw_angle.update(0.0, yaw_err, dt) + yaw_rate_cmd
    
        # 速率内环
        torque_roll = self.pid_roll_rate.update(roll_rate_sp, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(pitch_rate_sp, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(yaw_rate_sp, gyro[2], dt, self.filtered_acc[2])
    
        # 力矩符号修正
        torque_roll = -torque_roll 
        torque_pitch = -torque_pitch
        torque_yaw = -torque_yaw
    
        # 力矩限幅
        torque_roll = np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_pitch = np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_yaw = np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"])
    
        # 发布扭矩数据
        torque_msg = Vector3()
        torque_msg.x = torque_roll
        torque_msg.y = torque_pitch
        torque_msg.z = torque_yaw
        self.pub_torque_output.publish(torque_msg)
    
        return torque_roll, torque_pitch, torque_yaw
    
    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        """电机混控"""
        # 基础油门
        base = 1000.0 + throttle
        
        # Yaw扭矩放大
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain
        
        # 混控计算
        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE

        # 检查电机差值是否过大
        motor_max = np.max([motor1, motor2, motor3, motor4])
        motor_min = np.min([motor1, motor2, motor3, motor4])
        if motor_max - motor_min > 600:
            scale = 600 / (motor_max - motor_min) if motor_max != motor_min else 1.0
            motor_avg = (motor1 + motor2 + motor3 + motor4) / 4
            motor1 = motor_avg + (motor1 - motor_avg) * scale
            motor2 = motor_avg + (motor2 - motor_avg) * scale
            motor3 = motor_avg + (motor3 - motor_avg) * scale
            motor4 = motor_avg + (motor4 - motor_avg) * scale

        # 平滑滤波
        alpha = 0.18
        smoothed_motors = alpha * np.array([motor1, motor2, motor3, motor4]) + (1 - alpha) * self.state["motor_outputs"]
        
        return smoothed_motors
    
    def _check_data_validity(self):
        """检查数据有效性"""
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
        """发布DSHOT指令"""
        msg = WriteDSHOT()
        
        if isinstance(motor_pwm, (int, float)):
            # 单个值：所有电机相同
            dshot_val = pwm_to_dshot(motor_pwm)
            msg.channel1 = dshot_val
            msg.channel2 = dshot_val
            msg.channel3 = dshot_val
            msg.channel4 = dshot_val
            self.last_published_dshot = [dshot_val]*4
            
        elif isinstance(motor_pwm, (list, np.ndarray)):
            # 数组：每个电机单独设置
            if len(motor_pwm) >= 4:
                dshot = [pwm_to_dshot(p) for p in motor_pwm[:4]]
                msg.channel1 = dshot[0]
                msg.channel2 = dshot[1]
                msg.channel3 = dshot[2]
                msg.channel4 = dshot[3]
                self.last_published_dshot = dshot
            else:
                self.get_logger().error(f"❌ 电机PWM数组长度不足: {len(motor_pwm)}")
                return
        else:
            self.get_logger().error(f"❌ 不支持的motor_pwm类型: {type(motor_pwm)}")
            return
        
        # 发布消息
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ 发布DSHOT失败: {e}")
    
    def _publish_control_status(self):
        """发布控制状态"""
        status_msg = Vector3()
        
        # 控制模式编码
        if self.control_mode == "MANUAL":
            status_msg.x = 1.0
        elif self.control_mode == "POSITION":
            status_msg.x = 2.0
        else:
            status_msg.x = 0.0
            
        # 解锁状态
        status_msg.y = 1.0 if self.state["armed"] else 0.0
        
        # 开关位置
        status_msg.z = float(self.rc_data.get("left_switch", 0))
        
        self.pub_control_status.publish(status_msg)
    
    def _publish_status(self):
        """定期发布状态信息"""
        if not self.state["armed"]:
            return
            
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch"]))
            
            # 计算Yaw误差
            if hasattr(self, 'yaw_setpoint'):
                yaw_error = abs(np.rad2deg(self.yaw_setpoint - self.imu_data["yaw"]))
            else:
                yaw_error = 0.0
            
            # 状态日志
            self.get_logger().info(
                f"📊 飞行状态 | "
                f"模式:{self.control_mode} | "
                f"Roll误差:{roll_error:.2f}° | "
                f"Pitch误差:{pitch_error:.2f}° | "
                f"Yaw误差:{yaw_error:.2f}° | "
                f"DSHOT:{dshot}"
            )
            
            # 更新调试时间
            self.state["last_debug_time"] = current_time
    
    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info("🛑 正在关闭飞控...")
        
        # 发布上锁指令
        try:
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
            time.sleep(0.05)  # 确保指令发出
        except Exception as e:
            self.get_logger().warn(f"⚠️ 销毁时发布上锁指令失败: {e}")
            
        # 调用父类销毁方法
        super().destroy_node()

# ========== 主函数 ==========
def main(args=None):
    """主函数：启动飞控节点"""
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




#!/usr/bin/env python3
"""
无人机飞控主控制器 - 四元数版本 + 位置控制模式（带详细调试信息）
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

# ===================== 四元数核心数学工具函数 =====================
def quat_mult(q1, q2):
    """四元数乘法：q1 * q2（[w,x,y,z]格式）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_inv(q):
    """四元数求逆（共轭，单位四元数逆=共轭）"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def eul2quat(roll, pitch, yaw):
    """欧拉角转四元数（ZYX顺序，弧度）"""
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

def quat2eul(w, x, y, z):
    """四元数转欧拉角（仅用于日志输出，控制逻辑不依赖）"""
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

def quat_error(q_target, q_current):
    """计算目标四元数与当前四元数的误差（轴角形式，返回[rx, ry, rz]误差向量）"""
    q_err = quat_mult(quat_inv(q_current), q_target)
    # 转换为轴角误差（小角度近似，误差向量幅值=旋转角度，方向=旋转轴）
    angle = 2 * np.arctan2(np.linalg.norm(q_err[1:]), q_err[0])
    if np.linalg.norm(q_err[1:]) < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis

def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    """陀螺仪符号修正"""
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro])

def pwm_to_dshot(pwm_val):
    """PWM值转DSHOT值"""
    PWM_MIN, PWM_MAX = 1000, 2000
    pwm_clipped = np.clip(pwm_val, PWM_MIN, PWM_MAX)
    dshot_val = cfg.DSHOT_MIN + (pwm_clipped - PWM_MIN) * (cfg.DSHOT_MAX - cfg.DSHOT_MIN) / (PWM_MAX - PWM_MIN)
    return int(round(dshot_val))

# ===================== PID控制器 =====================
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
            self.d_term_sign = -1.0  # 速率环D项阻尼
        
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
        """重置PID状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.last_output = 0.0

# ===================== 主控制器 =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")
        
        # 初始化各个模块
        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        
        # 创建定时器
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        
        self.get_logger().info("✅ 飞控启动完成 - 支持手动/位置控制模式")
        self.get_logger().info("🎮 控制模式：左开关1=手动，3=位置控制，2=上锁")
        
    def _init_ros(self):
        """初始化ROS话题"""
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST, 
            depth=5
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, 
            depth=10
        )
        
        # ========== 发布者 ==========
        self.pub_imu_angle = self.create_publisher(Vector3, "/imu_angle", qos_reliable)
        self.pub_imu_gyro = self.create_publisher(Vector3, "/imu_gyro", qos_reliable)
        self.pub_dshot = self.create_publisher(WriteDSHOT, "/ecat/sn2228293/app3/write", 10)
        self.pub_torque_output = self.create_publisher(Vector3, "/torque_output", qos_reliable)
        self.pub_control_status = self.create_publisher(Vector3, "/fc_control_status", qos_reliable)
        
        # ========== 新增调试发布者 ==========
        # 姿态控制调试信息
        self.pub_attitude_debug = self.create_publisher(Vector3, "/attitude_debug", qos_reliable)
        self.pub_attitude_error = self.create_publisher(Vector3, "/attitude_error", qos_reliable)
        self.pub_control_mode_info = self.create_publisher(Vector3, "/control_mode_info", qos_reliable)
        self.pub_position_cmd_status = self.create_publisher(Vector3, "/position_cmd_status", qos_reliable)
        
        # 详细控制调试
        self.pub_control_details = self.create_publisher(Vector3, "/control_details", qos_reliable)
        
        # ========== 订阅者 ==========
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
        
        # ========== 新增：位置指令订阅 ==========
        self.sub_pos_cmd = self.create_subscription(
            Vector3,
            "/attitude_position_cmd",  # 位置控制器发布的指令
            self._pos_cmd_callback,
            qos_reliable
        )
        
        # 线程锁
        self.lock = threading.Lock()
        
    def _init_data(self):
        """初始化数据存储"""
        # 遥控器数据
        self.rc_data = {
            "left_y": 0.0,
            "left_x": 0.0, 
            "right_x": 0.0,
            "right_y": 0.0,
            "left_switch": cfg.LOCK_SWITCH_VALUE  # 默认上锁状态
        }
        
        # IMU数据
        self.imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # 当前姿态四元数 [w,x,y,z]
            "gyro": np.array([0.0, 0.0, 0.0]),
            "roll": 0.0, 
            "pitch": 0.0, 
            "yaw": 0.0  # 仅日志用
        }
        
        # 滤波器数据
        self.filtered_acc = np.array([0.0, 0.0, 0.0])
        
        # 位置控制指令数据
        self.pos_cmd_data = {
            "roll": 0.0,
            "pitch": 0.0,
            "throttle": 1000.0
        }
        self.last_pos_cmd_time = 0.0
        self.pos_cmd_timeout = 0.2  # 200ms超时
        
        # 时间记录
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0
        self.last_published_dshot = [1200]*4
        
        # 四元数数据
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 目标姿态四元数
        self.last_yaw_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 上一时刻Yaw四元数
        
        # 控制模式
        self.control_mode = "MANUAL"  # MANUAL, POSITION
        
        # 控制调试数据
        self.control_debug = {
            "target_roll": 0.0,
            "target_pitch": 0.0,
            "target_yaw": 0.0,
            "position_cmd_valid": False
        }
        
    def _init_controllers(self):
        """初始化PID控制器"""
        # 横滚角度PID
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
        
        # 横滚速率PID
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
        
        # 俯仰角度PID
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
        
        # 俯仰速率PID
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
        
        # 偏航角度PID
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
        
        # 偏航速率PID
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
        """初始化状态变量"""
        self.state = {
            "armed": False,
            "stick_deadband": cfg.RC_DEAD_ZONE * 0.3,
            "motor_outputs": np.array([1000.0]*4),
            "init_quat": None,  # 初始化姿态四元数
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.8,
            "last_debug_time": 0.0,
            "last_control_debug_time": 0.0
        }
        
        # 陀螺仪死区
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        
        # Yaw控制参数
        self.yaw_stick_scale = 0.2
        self.yaw_dshot_gain = 0.6
        
    # ========== 回调函数 ==========
    def _rc_callback(self, msg):
        """遥控器数据回调"""
        with self.lock:
            self.rc_data["left_y"] = msg.left_y
            self.rc_data["left_x"] = msg.left_x
            self.rc_data["right_x"] = msg.right_x
            self.rc_data["right_y"] = -msg.right_y
            self.rc_data["left_switch"] = msg.left_switch
            self.last_rc_time = self.get_clock().now().nanoseconds / 1e9
            
            # 更新解锁状态和控制模式
            self._update_arming_and_mode()
            
    def _imu_callback(self, msg):
        """IMU数据回调"""
        with self.lock:
            # 读取IMU原始四元数
            current_quat = np.array([
                msg.orientation.w, 
                msg.orientation.x, 
                msg.orientation.y, 
                msg.orientation.z
            ])
            
            # 修正Pitch/Yaw符号（适配硬件安装方向）
            current_quat = self._correct_quat_sign(current_quat)
            
            # 初始化姿态（仅首次）
            if self.state["init_quat"] is None:
                self.state["init_quat"] = current_quat.copy()
                self.state["initialized"] = True
                # 初始化Yaw随动：以上一时刻四元数为目标
                self.last_yaw_quat = self._extract_yaw_quat(current_quat)
                self.target_quat = current_quat.copy()
                self.get_logger().info("📡 IMU初始化完成")
            
            # 计算相对初始化姿态的四元数（归零）
            rel_quat = quat_mult(current_quat, quat_inv(self.state["init_quat"]))
            
            # 仅用于日志：转欧拉角
            roll_zeroed, pitch_zeroed, current_yaw = quat2eul(*rel_quat)
            current_yaw = current_yaw  # Yaw符号修正（日志用）
            
            # 更新Yaw随动目标（以上一时刻Yaw为期望）
            current_yaw_quat = self._extract_yaw_quat(current_quat)
            self.target_quat = self._set_quat_yaw(rel_quat, self.last_yaw_quat)  # 目标Yaw=上一时刻Yaw
            self.last_yaw_quat = current_yaw_quat  # 更新上一时刻Yaw
            
            # 保存数据（四元数为主，欧拉角仅日志）
            self.imu_data["quat"] = rel_quat
            self.imu_data["roll"] = roll_zeroed
            self.imu_data["pitch"] = pitch_zeroed
            self.imu_data["yaw"] = current_yaw
            
            # 陀螺仪数据修正
            gyro_rotated = rotate_gyro_data(
                msg.angular_velocity.x, 
                msg.angular_velocity.y, 
                msg.angular_velocity.z
            )
            self.imu_data["gyro"] = gyro_rotated
            
            # 日志输出
            if int(time.time() * 10) % 10 == 0:  # 约1秒一次
                self.get_logger().info(
                    f"📡 IMU数据: Roll={np.rad2deg(roll_zeroed):.3f}°, "
                    f"Pitch={np.rad2deg(pitch_zeroed):.3f}°, "
                    f"Yaw={np.rad2deg(current_yaw):.3f}°"
                )

            # 发布IMU数据（欧拉角仅日志用）
            angle_msg = Vector3()
            angle_msg.x = np.rad2deg(roll_zeroed)
            angle_msg.y = np.rad2deg(pitch_zeroed)
            angle_msg.z = np.rad2deg(current_yaw)
            self.pub_imu_angle.publish(angle_msg)
            
            # 发布陀螺仪数据
            gyro_msg = Vector3()
            gyro_msg.x = gyro_rotated[0]
            gyro_msg.y = gyro_rotated[1]
            gyro_msg.z = gyro_rotated[2]
            self.pub_imu_gyro.publish(gyro_msg)
            
            self.last_imu_time = self.get_clock().now().nanoseconds / 1e9
    
    def _filtered_acc_callback(self, msg):
        """滤波后的角加速度回调"""
        with self.lock:
            self.filtered_acc = np.array([msg.x, msg.y, msg.z])
            
    def _pos_cmd_callback(self, msg):
        """位置控制指令回调"""
        with self.lock:
            self.pos_cmd_data["roll"] = msg.x
            self.pos_cmd_data["pitch"] = msg.y
            self.pos_cmd_data["throttle"] = msg.z
            self.last_pos_cmd_time = self.get_clock().now().nanoseconds / 1e9
            
            # 记录位置指令调试信息
            self.control_debug["target_roll"] = msg.x
            self.control_debug["target_pitch"] = msg.y
            
    def _correct_quat_sign(self, quat):
        """修正四元数符号（适配硬件安装方向）"""
        w, x, y, z = quat
        # Pitch符号修正：反转y分量
        y = -y
        # Yaw符号修正：反转z分量
        z = -z
        return np.array([w, x, y, z])
    
    def _extract_yaw_quat(self, quat):
        """从四元数中提取仅Yaw分量的四元数（Roll/Pitch=0）"""
        roll, pitch, yaw = quat2eul(*quat)
        return eul2quat(0, 0, yaw)
    
    def _set_quat_yaw(self, quat, yaw_quat):
        """保持Roll/Pitch不变，设置Yaw为目标四元数的Yaw"""
        roll, pitch, _ = quat2eul(*quat)
        _, _, yaw = quat2eul(*yaw_quat)
        return eul2quat(roll, pitch, yaw)
            
    def _update_arming_and_mode(self):
        """更新解锁状态和控制模式
        左开关：
          1(上)：解锁 + 手动模式
          3(中)：解锁 + 位置控制模式
          2(下)：上锁
        """
        current_switch = self.rc_data["left_switch"]
        
        # 位置2：上锁（下）
        if current_switch == 2:  # 上锁位置
            if self.state["armed"]:
                self.state["armed"] = False
                self.control_mode = "MANUAL"
                self.get_logger().info("🔒 已上锁")
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                self._reset_all_controllers()
            return
            
        # 位置1：解锁 + 手动模式（上）
        elif current_switch == 1:  
            if not self.state["armed"]:
                if not self.state["initialized"]:
                    self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                    return
                self.state["armed"] = True
                self.control_mode = "MANUAL"
                self.get_logger().info("🔓 已解锁 - 手动模式")
                self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
                self._reset_all_controllers()
            elif self.control_mode != "MANUAL":  # 已解锁，切换到手动模式
                self.control_mode = "MANUAL"
                self.get_logger().info("🔄 切换到手动模式")
                self._reset_all_controllers()
            
        # 位置3：解锁 + 位置控制模式（中）
        elif current_switch == 3:  
            if not self.state["armed"]:
                if not self.state["initialized"]:
                    self.get_logger().warn("⚠️ IMU未初始化，禁止解锁")
                    return
                self.state["armed"] = True
                self.control_mode = "POSITION"
                self.get_logger().info("🔓 已解锁 - 位置控制模式")
                self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
                self._reset_all_controllers()
            elif self.control_mode != "POSITION":  # 已解锁，切换到位置控制模式
                # 检查位置指令是否有效
                current_time = self.get_clock().now().nanoseconds / 1e9
                pos_cmd_valid = (current_time - self.last_pos_cmd_time) < self.pos_cmd_timeout
                
                if pos_cmd_valid:
                    self.control_mode = "POSITION"
                    self.get_logger().info("🎯 切换到位置控制模式")
                    self._reset_all_controllers()
                else:
                    self.get_logger().warn("⚠️ 位置指令无效，保持在手动模式")
                    self.control_mode = "MANUAL"
            
    def _reset_all_controllers(self):
        """重置所有PID控制器"""
        for pid in [self.pid_roll_angle, self.pid_pitch_angle, self.pid_yaw_angle, 
                    self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        self.get_logger().info("🔄 所有PID控制器已重置")
        
    # ========== 核心控制循环 ==========
    def _control_loop(self):
        """主控制循环（1000Hz）"""
        with self.lock:
            # 检查数据有效性
            if not self._check_data_validity():
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            # 检查是否解锁
            if not self.state["armed"]:
                self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
                return
                
            dt = 1.0 / cfg.CONTROL_FREQ
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # ========== 根据控制模式选择输入源 ==========
            if self.control_mode == "MANUAL":
                # 手动模式：完全使用遥控器
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                
            elif self.control_mode == "POSITION":
                # 位置控制模式：检查位置指令有效性
                pos_cmd_timeout = (current_time - self.last_pos_cmd_time) > self.pos_cmd_timeout
                
                if pos_cmd_timeout:
                    # 位置指令超时，自动切回手动模式
                    self.control_mode = "MANUAL"
                    self.get_logger().warn("⚠️ 位置指令超时，自动切换回手动模式")
                    throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
                else:
                    # 使用位置控制器指令
                    throttle_raw = np.clip(self.pos_cmd_data["throttle"], 1000.0, 2000.0)
                    roll_target = np.clip(self.pos_cmd_data["roll"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    pitch_target = np.clip(self.pos_cmd_data["pitch"], -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
                    
                    # 将油门从1000-2000转换为0-1000（与_process_stick一致）
                    throttle = (throttle_raw - 1000.0) / 1000.0 * 1000.0
                    
                    # Yaw仍然使用遥控器控制（允许在位置控制时调整方向）
                    _, _, _, yaw_stick = self._process_stick()
                    
                    # 记录调试信息
                    self.control_debug["position_cmd_valid"] = True
                    
                    # 调试输出
                    if int(current_time * 100) % 100 == 0:  # 约1秒一次
                        self.get_logger().info(
                            f"🎯 位置控制指令: "
                            f"Roll={np.rad2deg(roll_target):.1f}°, "
                            f"Pitch={np.rad2deg(pitch_target):.1f}°, "
                            f"Throttle={throttle_raw:.0f}"
                        )
            else:
                # 默认手动模式
                throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            
            # ========== PID计算 ==========
            torque_roll, torque_pitch, torque_yaw = self._pid_control_quat(
                roll_target, pitch_target, yaw_stick, dt
            )
            
            # ========== 发布调试信息 ==========
            self._publish_control_debug(roll_target, pitch_target, dt)
            
            # ========== 电机混控 ==========
            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            
            # ========== 发布DSHOT指令 ==========
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
            
            # ========== 发布控制状态 ==========
            self._publish_control_status()
    
    def _process_stick(self):
        """处理遥控器摇杆数据"""
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > self.state["stick_deadband"] else 0.0
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > self.state["stick_deadband"] else 0.0
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > self.state["stick_deadband"]*0.5 else 0.0
        throttle_raw = self.rc_data["left_y"]
        
        # 计算目标角度（限幅）
        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG * 3
        
        # 计算油门（-1~1映射到0~1000）
        throttle = np.clip((throttle_raw + 1.0) / 2.0 * 1000.0, 0.0, 1000.0)
        
        return throttle, roll_target, pitch_target, yaw_raw
    
    def _pid_control_quat(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        """基于四元数误差的PID控制 - 修改为Yaw阻尼模式"""
        current_quat = self.imu_data["quat"]
        gyro = self.imu_data["gyro"].copy()
    
        # 【修改1】生成目标姿态四元数（Yaw使用上一时刻值 + 摇杆指令）
        # 获取当前Yaw
        roll_current, pitch_current, current_yaw = quat2eul(*current_quat)
    
        # 使用上一控制周期的Yaw作为目标（实现阻尼效果）
        if not hasattr(self, 'prev_yaw_for_damping'):
            self.prev_yaw_for_damping = current_yaw
    
        # 【修改2】Yaw目标 = 上一时刻Yaw + 摇杆速率积分
        # 摇杆指令是角速度，需要积分得到角度变化
        yaw_target = self.prev_yaw_for_damping + yaw_rate_cmd * dt
    
        # 构造目标四元数
        target_quat = eul2quat(roll_target, pitch_target, yaw_target)
    
        # 保存当前Yaw作为下一时刻的"上一时刻Yaw"
        self.prev_yaw_for_damping = current_yaw
    
        # 计算四元数误差（轴角形式）
        quat_err = quat_error(target_quat, current_quat)
        roll_err, pitch_err, yaw_err = quat_err[0], quat_err[1], quat_err[2]
    
        # 陀螺仪死区处理
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]
    
        # 角度外环（Roll/Pitch使用回平逻辑，Yaw使用阻尼逻辑）
        roll_rate_sp = self.pid_roll_angle.update(0.0, roll_err, dt)
        pitch_rate_sp = self.pid_pitch_angle.update(0.0, pitch_err, dt)
    
        # 【修改3】Yaw角度环：期望值=0（即不期望有固定位置，只期望跟随上一时刻）
        # 但实际上，由于target_quat使用了上一时刻的Yaw，yaw_err已经反映了这个跟随误差
        yaw_rate_sp = self.pid_yaw_angle.update(0.0, yaw_err, dt) + yaw_rate_cmd
    
        # 速率内环
        torque_roll = self.pid_roll_rate.update(roll_rate_sp, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(pitch_rate_sp, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(yaw_rate_sp, gyro[2], dt, self.filtered_acc[2])
    
        # 力矩符号修正
        torque_roll = -torque_roll 
        torque_pitch = -torque_pitch
        torque_yaw = -torque_yaw
    
        # 力矩限幅
        torque_roll = np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_pitch = np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_yaw = np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"])
        
        # 发布姿态调试信息
        self._publish_attitude_debug(roll_target, pitch_target, roll_current, pitch_current, roll_err, pitch_err, yaw_err)
    
        # 发布扭矩数据
        torque_msg = Vector3()
        torque_msg.x = torque_roll
        torque_msg.y = torque_pitch
        torque_msg.z = torque_yaw
        self.pub_torque_output.publish(torque_msg)
    
        return torque_roll, torque_pitch, torque_yaw
    
    def _publish_attitude_debug(self, roll_target, pitch_target, roll_current, pitch_current, roll_err, pitch_err, yaw_err):
        """发布姿态调试信息"""
        # 目标 vs 当前姿态
        attitude_debug = Vector3()
        attitude_debug.x = np.rad2deg(roll_target)   # 目标滚转角（度）
        attitude_debug.y = np.rad2deg(pitch_target)  # 目标俯仰角（度）
        attitude_debug.z = np.rad2deg(roll_current)  # 当前滚转角（度）
        self.pub_attitude_debug.publish(attitude_debug)
        
        # 姿态误差
        attitude_error = Vector3()
        attitude_error.x = np.rad2deg(roll_err)    # 滚转误差（度）
        attitude_error.y = np.rad2deg(pitch_err)   # 俯仰误差（度）
        attitude_error.z = np.rad2deg(yaw_err)     # 偏航误差（度）
        self.pub_attitude_error.publish(attitude_error)
    
    def _publish_control_debug(self, roll_target, pitch_target, dt):
        """发布控制调试信息"""
        current_time = time.time()
        
        # 控制模式信息
        mode_info = Vector3()
        if self.control_mode == "MANUAL":
            mode_info.x = 1.0
        elif self.control_mode == "POSITION":
            mode_info.x = 2.0
        else:
            mode_info.x = 0.0
        
        # 位置指令有效性
        cmd_timeout = time.time() - self.last_pos_cmd_time > self.pos_cmd_timeout
        mode_info.y = 1.0 if (self.control_mode == "POSITION" and not cmd_timeout) else 0.0
        
        # 电机输出范围
        motor_outputs = self.state["motor_outputs"]
        mode_info.z = np.max(motor_outputs) - np.min(motor_outputs)  # 电机最大差值
        self.pub_control_mode_info.publish(mode_info)
        
        # 位置指令状态
        pos_cmd_status = Vector3()
        pos_cmd_status.x = self.pos_cmd_data["roll"]
        pos_cmd_status.y = self.pos_cmd_data["pitch"]
        pos_cmd_status.z = self.pos_cmd_data["throttle"]
        self.pub_position_cmd_status.publish(pos_cmd_status)
        
        # 详细控制信息（每0.2秒发布一次）
        if current_time - self.state["last_control_debug_time"] > 0.2:
            control_details = Vector3()
            control_details.x = np.rad2deg(self.imu_data["roll"])  # 当前滚转角
            control_details.y = np.rad2deg(self.imu_data["pitch"]) # 当前俯仰角
            control_details.z = np.rad2deg(self.imu_data["yaw"])   # 当前偏航角
            self.pub_control_details.publish(control_details)
            
            self.state["last_control_debug_time"] = current_time
            
    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        """电机混控"""
        # 基础油门
        base = 1000.0 + throttle
        
        # Yaw扭矩放大
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain
        
        # 混控计算
        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified) * cfg.DSHOT_SCALE

        # 检查电机差值是否过大
        motor_max = np.max([motor1, motor2, motor3, motor4])
        motor_min = np.min([motor1, motor2, motor3, motor4])
        if motor_max - motor_min > 600:
            scale = 600 / (motor_max - motor_min) if motor_max != motor_min else 1.0
            motor_avg = (motor1 + motor2 + motor3 + motor4) / 4
            motor1 = motor_avg + (motor1 - motor_avg) * scale
            motor2 = motor_avg + (motor2 - motor_avg) * scale
            motor3 = motor_avg + (motor3 - motor_avg) * scale
            motor4 = motor_avg + (motor4 - motor_avg) * scale

        # 平滑滤波
        alpha = 0.18
        smoothed_motors = alpha * np.array([motor1, motor2, motor3, motor4]) + (1 - alpha) * self.state["motor_outputs"]
        
        return smoothed_motors
    
    def _check_data_validity(self):
        """检查数据有效性"""
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
        """发布DSHOT指令"""
        msg = WriteDSHOT()
        
        if isinstance(motor_pwm, (int, float)):
            # 单个值：所有电机相同
            dshot_val = pwm_to_dshot(motor_pwm)
            msg.channel1 = dshot_val
            msg.channel2 = dshot_val
            msg.channel3 = dshot_val
            msg.channel4 = dshot_val
            self.last_published_dshot = [dshot_val]*4
            
        elif isinstance(motor_pwm, (list, np.ndarray)):
            # 数组：每个电机单独设置
            if len(motor_pwm) >= 4:
                dshot = [pwm_to_dshot(p) for p in motor_pwm[:4]]
                msg.channel1 = dshot[0]
                msg.channel2 = dshot[1]
                msg.channel3 = dshot[2]
                msg.channel4 = dshot[3]
                self.last_published_dshot = dshot
            else:
                self.get_logger().error(f"❌ 电机PWM数组长度不足: {len(motor_pwm)}")
                return
        else:
            self.get_logger().error(f"❌ 不支持的motor_pwm类型: {type(motor_pwm)}")
            return
        
        # 发布消息
        try:
            self.pub_dshot.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ 发布DSHOT失败: {e}")
    
    def _publish_control_status(self):
        """发布控制状态"""
        status_msg = Vector3()
        
        # 控制模式编码
        if self.control_mode == "MANUAL":
            status_msg.x = 1.0
        elif self.control_mode == "POSITION":
            status_msg.x = 2.0
        else:
            status_msg.x = 0.0
            
        # 解锁状态
        status_msg.y = 1.0 if self.state["armed"] else 0.0
        
        # 开关位置
        status_msg.z = float(self.rc_data.get("left_switch", 0))
        
        self.pub_control_status.publish(status_msg)
    
    def _publish_status(self):
        """定期发布状态信息"""
        if not self.state["armed"]:
            return
            
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch"]))
            
            # 计算Yaw误差
            if hasattr(self, 'yaw_setpoint'):
                yaw_error = abs(np.rad2deg(self.yaw_setpoint - self.imu_data["yaw"]))
            else:
                yaw_error = 0.0
            
            # 状态日志
            self.get_logger().info(
                f"📊 飞行状态 | "
                f"模式:{self.control_mode} | "
                f"Roll误差:{roll_error:.2f}° | "
                f"Pitch误差:{pitch_error:.2f}° | "
                f"Yaw误差:{yaw_error:.2f}° | "
                f"DSHOT:{dshot}"
            )
            
            # 更新调试时间
            self.state["last_debug_time"] = current_time
    
    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info("🛑 正在关闭飞控...")
        
        # 发布上锁指令
        try:
            self._publish_dshot(cfg.DSHOT_IDLE_LOCK)
            time.sleep(0.05)  # 确保指令发出
        except Exception as e:
            self.get_logger().warn(f"⚠️ 销毁时发布上锁指令失败: {e}")
            
        # 调用父类销毁方法
        super().destroy_node()

# ========== 主函数 ==========
def main(args=None):
    """主函数：启动飞控节点"""
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

