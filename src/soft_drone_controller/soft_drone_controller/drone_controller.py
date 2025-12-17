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
