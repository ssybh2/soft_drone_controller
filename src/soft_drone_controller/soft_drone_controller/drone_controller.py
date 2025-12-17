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

# ===================== 数学工具函数 =====================
def quat2eul(w, x, y, z):
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

# 【修改1：修正陀螺仪X/Z轴符号（解决IMU gyro X/Z反向）】
# 原：[-roll_gyro, -pitch_gyro, yaw_gyro] → 现：恢复X轴，反转Z轴，调整Pitch gyro符号
def rotate_gyro_data(roll_gyro, pitch_gyro, yaw_gyro):
    return np.array([roll_gyro, -pitch_gyro, -yaw_gyro])  # X轴恢复、Z轴反转、Pitch gyro恢复

def pwm_to_dshot(pwm_val):
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
            self.d_term_sign = -1.0  # 对于速率环，D项应该为负（阻尼项）
        
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
            #self.prev_error = 0.0
            #self.prev_error = 0.0
            # 【修改2：删除prev_measurement重置（解决符号断档）】
            # self.prev_measurement = measurement  # 删掉这行！
            self.last_output = 0.0
            return 0.0
            
        # Yaw外环误差强制趋近0，仅随动无修正
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
        
        # 【修改3：D项符号适配修正后的陀螺仪（去掉负号，避免反向）】
        if self.use_angular_acc and angular_acc is not None:
            d_term = self.kd * angular_acc * self.d_term_sign # 去掉-号
        else:
            d_term = self.kd * measurement_rate * self.d_term_sign  # 去掉-号

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

# ===================== 主控制器（还原Yaw随动+电机均衡） =====================
class BalanceController(Node):
    def __init__(self):
        super().__init__("balance_controller")
        self._init_ros()
        self._init_data()
        self._init_controllers()
        self._init_state()
        
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        
        self.get_logger().info("✅ 控制器启动完成 - Yaw随动+电机均衡+姿态稳定")
        
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
        self.imu_data = {"roll":0.0, "pitch":0.0, "yaw":0.0, "gyro":np.array([0.0,0.0,0.0])}
        self.filtered_acc = np.array([0.0,0.0,0.0])
        
        self.last_rc_time = 0.0
        self.last_imu_time = 0.0
        self.last_published_dshot = [1200]*4
        
        # Yaw核心：期望=上一时刻实际值，严格随动
        self.yaw_setpoint = 0.0
        
    def _init_controllers(self):
        # ---------------------- Roll/Pitch（稳定修正） ----------------------
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
        # 【修改4：Roll速率环关闭角加速度（保持三轴一致）】
        self.pid_roll_rate = ImprovedPID(
            kp=cfg.PID_ROLL_RATE["kp"] * 10.0,
            ki=cfg.PID_ROLL_RATE["ki"] * 0,
            kd=cfg.PID_ROLL_RATE["kd"],
            i_max=0.5,
            i_min=-0.5,
            use_angular_acc=False,  # 从True改为False
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
        
        # ---------------------- Yaw（严格随动+均衡输出） ----------------------
        self.pid_yaw_angle = ImprovedPID(
            kp=cfg.PID_YAW_ANGLE["kp"] * 4.0,  # 适度增益，随动无偏移
            ki=cfg.PID_YAW_ANGLE["ki"] * 0.0,  # 无积分，避免累积偏差
            kd=cfg.PID_YAW_ANGLE["kd"] * 0.4,
            i_max=0.05,
            i_min=-0.05,
            use_angular_acc=False,
            node=self,
            axis="yaw"
        )
        self.pid_yaw_rate = ImprovedPID(
            kp=cfg.PID_YAW_RATE["kp"] * 4.0,  # 速率环增益适配，旋转平稳
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
            "angle_zero_point": None,
            "initialized": False,
            "torque_limit_roll_pitch": 1.3,
            "torque_limit_yaw": 1.9,  # Yaw扭矩适中，电机均衡
            "last_debug_time": 0.0
        }
        self.gyro_deadband_roll_pitch = cfg.GYRO_DEADBAND_ROLL_PITCH
        self.gyro_deadband_yaw = cfg.GYRO_DEADBAND_YAW
        self.yaw_stick_scale = 0.2  # 摇杆灵敏度适配，旋转有力不突兀
        self.yaw_dshot_gain = 1.0  # 温和放大，兼顾力度与均衡
        
    # ========== 回调函数 ==========
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
            w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
            roll, pitch, current_yaw = quat2eul(w, x, y, z)
            
            # 【修改5：修正Pitch角度符号（解决Pitch飞行方向反向）】
            # 原：pitch = -pitch → 现：恢复pitch原始符号（删掉-号）
            pitch = -pitch  # 删掉这行！
            current_yaw = -current_yaw  # Yaw符号保留（按需调整）
             
            if self.state["angle_zero_point"] is None:
                self.state["angle_zero_point"] = np.array([roll, pitch])
                self.state["initialized"] = True
                self.yaw_setpoint = current_yaw  # 初始化Yaw期望=当前值
            
            # Roll/Pitch归零，Yaw期望=当前实际值（严格随动）
            roll_zeroed = roll - self.state["angle_zero_point"][0]
            pitch_zeroed = pitch - self.state["angle_zero_point"][1]
            self.yaw_setpoint = current_yaw  # 关键：每帧更新期望=当前实际Yaw
            
            self.imu_data["roll"] = roll_zeroed
            self.imu_data["pitch"] = pitch_zeroed
            self.imu_data["yaw"] = current_yaw
            ######################################################################
            gyro_rotated = rotate_gyro_data(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
            self.imu_data["gyro"] = gyro_rotated
            self.get_logger().info(f"Roll角速率: {gyro_rotated[0]:.3f} | Roll角度: {np.rad2deg(roll_zeroed):.2f}°")

            # 发布IMU数据
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
            self.get_logger().info("🔓 已解锁 - Yaw随动+电机均衡")
            self._publish_dshot(cfg.DSHOT_IDLE_UNLOCK)
            self._reset_all_controllers()
            
    def _reset_all_controllers(self):
        for pid in [self.pid_roll_angle, self.pid_pitch_angle, self.pid_yaw_angle, 
                    self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
            pid.reset()
        self.state["motor_outputs"] = np.array([1000.0]*4)
        
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
            throttle, roll_target, pitch_target, yaw_stick = self._process_stick()
            
            # Roll/Pitch期望=0°（回平）
            roll_target = np.clip(roll_target, -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
            pitch_target = np.clip(pitch_target, -cfg.MAX_ROLL_PITCH_ANG, cfg.MAX_ROLL_PITCH_ANG)
            # Yaw：期望=当前实际值+摇杆转速指令（仅控旋转速度，不偏移期望）
            yaw_rate_cmd = 0.0
            if abs(yaw_stick) > self.state["stick_deadband"]:
                yaw_rate_cmd = yaw_stick * self.yaw_stick_scale  # 摇杆直接控Yaw速率
            
            # PID计算
            torque_roll, torque_pitch, torque_yaw = self._pid_control(roll_target, pitch_target, yaw_rate_cmd, dt)
            # 电机混控（均衡分配扭矩）
            motor_pwm = self._motor_mix(throttle, torque_roll, torque_pitch, torque_yaw)
            # 发布DSHOT
            self._publish_dshot(motor_pwm)
            self.state["motor_outputs"] = motor_pwm
    
    def _process_stick(self):
        roll_raw = self.rc_data["right_x"] if abs(self.rc_data["right_x"]) > self.state["stick_deadband"] else 0.0
        # 【修改6：Pitch摇杆符号适配（可选，若仍反向则加-号）】
        pitch_raw = self.rc_data["right_y"] if abs(self.rc_data["right_y"]) > self.state["stick_deadband"] else 0.0
        # 若Pitch仍反向，改为：pitch_raw = -self.rc_data["right_y"] ...
        yaw_raw = self.rc_data["left_x"] if abs(self.rc_data["left_x"]) > self.state["stick_deadband"]*0.5 else 0.0
        throttle_raw = self.rc_data["left_y"]
        
        roll_target = roll_raw * cfg.MAX_ROLL_PITCH_ANG*3
        pitch_target = pitch_raw * cfg.MAX_ROLL_PITCH_ANG*3
        throttle = np.clip((throttle_raw + 1.0)/2.0 * 1000.0, 0.0, 1000.0)
        return throttle, roll_target, pitch_target, yaw_raw
    
    def _pid_control(self, roll_target, pitch_target, yaw_rate_cmd, dt):
        roll_current = self.imu_data["roll"]
        pitch_current = self.imu_data["pitch"]
        yaw_current = self.imu_data["yaw"]
        gyro = self.imu_data["gyro"].copy()
        
        # 陀螺仪死区处理
        gyro[0] = 0.0 if abs(gyro[0]) < self.gyro_deadband_roll_pitch else gyro[0]
        gyro[1] = 0.0 if abs(gyro[1]) < self.gyro_deadband_roll_pitch else gyro[1]
        gyro[2] = 0.0 if abs(gyro[2]) < self.gyro_deadband_yaw else gyro[2]
        
        # 角度外环（Yaw期望=当前实际值，仅修正微小偏移）
        roll_rate_sp = self.pid_roll_angle.update(roll_target, roll_current, dt)
        pitch_rate_sp = self.pid_pitch_angle.update(pitch_target, pitch_current, dt)
        yaw_rate_sp = self.pid_yaw_angle.update(self.yaw_setpoint, yaw_current, dt) + yaw_rate_cmd  # 叠加摇杆速率指令
        
        # 速率内环（平稳输出扭矩）
        torque_roll = self.pid_roll_rate.update(roll_rate_sp, gyro[0], dt, self.filtered_acc[0])
        torque_pitch = self.pid_pitch_rate.update(pitch_rate_sp, gyro[1], dt, self.filtered_acc[1])
        torque_yaw = self.pid_yaw_rate.update(yaw_rate_sp, gyro[2], dt, self.filtered_acc[2])
        
        # 力矩限幅（均衡输出）
        torque_roll = np.clip(torque_roll, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_pitch = np.clip(torque_pitch, -self.state["torque_limit_roll_pitch"], self.state["torque_limit_roll_pitch"])
        torque_yaw = np.clip(torque_yaw, -self.state["torque_limit_yaw"], self.state["torque_limit_yaw"])
        
        torque_msg = Vector3()
        torque_msg.x = torque_roll
        torque_msg.y = torque_pitch
        torque_msg.z = torque_yaw
        self.pub_torque_output.publish(torque_msg)
        return torque_roll, torque_pitch, torque_yaw
    
    def _motor_mix(self, throttle, torque_roll, torque_pitch, torque_yaw):
        base = 1000.0 + throttle
        # Yaw扭矩温和放大，均衡分配到电机
        yaw_torque_amplified = torque_yaw * self.yaw_dshot_gain
        # 标准X型混控，扭矩分配均衡无突兀
        motor1 = base + (cfg.MIX_MATRIX[0][0]*torque_roll + cfg.MIX_MATRIX[0][1]*torque_pitch + cfg.MIX_MATRIX[0][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE
        motor2 = base + (cfg.MIX_MATRIX[1][0]*torque_roll + cfg.MIX_MATRIX[1][1]*torque_pitch + cfg.MIX_MATRIX[1][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE
        motor3 = base + (cfg.MIX_MATRIX[2][0]*torque_roll + cfg.MIX_MATRIX[2][1]*torque_pitch + cfg.MIX_MATRIX[2][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE
        motor4 = base + (cfg.MIX_MATRIX[3][0]*torque_roll + cfg.MIX_MATRIX[3][1]*torque_pitch + cfg.MIX_MATRIX[3][2]*yaw_torque_amplified)*cfg.DSHOT_SCALE

        smoothed_motors = np.array([motor1, motor2, motor3, motor4])
        # 限制电机输出差值，确保均衡
        motor_max = np.max([motor1, motor2, motor3, motor4])
        motor_min = np.min([motor1, motor2, motor3, motor4])
        if motor_max - motor_min > 600:  # 最大差值控制，避免悬殊
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
        if current_time - self.last_rc_time > cfg.DATA_TIMEOUT or current_time - self.last_imu_time > cfg.DATA_TIMEOUT:
            if self.state["armed"]:
                self.state["armed"] = False
                self.get_logger().error("🔴 数据超时，强制上锁")
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
        if not self.state["armed"]:
            return
        current_time = time.time()
        if current_time - self.state["last_debug_time"] > 0.5:
            dshot = self.last_published_dshot
            roll_error = abs(np.rad2deg(self.imu_data["roll"]))
            pitch_error = abs(np.rad2deg(self.imu_data["pitch"]))
            yaw_error = abs(np.rad2deg(self.yaw_setpoint - self.imu_data["yaw"]))
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
