#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, Vector3
# 导入你的配置文件，保持参数体系一致
from soft_drone_controller.config import controller_params as cfg

# 位置PID控制器（复用飞控的PID核心逻辑，适配位置控制场景）
class PositionPID:
    def __init__(self, kp, ki, kd, max_output, axis=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output  # 输出限幅（防止姿态角/油门过大）
        self.axis = axis
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = 0.0

    def update(self, setpoint, measured, dt):
        if dt <= 0:
            dt = 1.0 / cfg.CONTROL_FREQ  # 与飞控控制频率同步
        
        # 核心PID计算
        error = setpoint - measured
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_output/2, self.max_output/2)  # 积分限幅
        derivative = (error - self.prev_error) / dt if self.last_time != 0 else 0.0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.max_output, self.max_output)  # 输出限幅
        
        # 更新状态
        self.prev_error = error
        self.last_time = dt
        return output

# 位置控制器主节点（核心：动捕位置反馈→飞控姿态/油门目标）
class DronePositionController(Node):
    def __init__(self):
        super().__init__("drone_position_controller")
        self._init_qos()          # 初始化QoS配置（匹配飞控）
        self._init_pid()          # 初始化位置PID参数
        self._init_ros_topics()   # 初始化订阅/发布话题
        self._init_data()         # 初始化数据存储
        
        # 控制定时器（频率与飞控一致，默认cfg.CONTROL_FREQ）
        self.control_timer = self.create_timer(1.0 / cfg.CONTROL_FREQ, self._control_loop)
        self.get_logger().info("✅ 位置控制器启动完成 - 适配BalanceController飞控")

    def _init_qos(self):
        """匹配飞控的QoS配置，保证数据传输稳定性"""
        self.qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )

    def _init_pid(self):
        """初始化位置PID参数（需根据无人机实测调试）"""
        # X/Y轴：位置误差→姿态角（max_output=最大倾斜角，单位rad）
        self.pid_x = PositionPID(kp=0.8, ki=0.02, kd=0.1, max_output=np.deg2rad(15), axis="x")  # X→pitch
        self.pid_y = PositionPID(kp=0.8, ki=0.02, kd=0.1, max_output=np.deg2rad(15), axis="y")  # Y→roll
        # Z轴：位置误差→油门（max_output=油门增量，0-1000）
        self.pid_z = PositionPID(kp=50.0, ki=1.0, kd=5.0, max_output=200.0, axis="z")

    def _init_ros_topics(self):
        """定义所有订阅/发布话题"""
        # 1. 订阅动捕的无人机实际位置（核心反馈）
        self.sub_mocap_pose = self.create_subscription(
            PoseStamped,
            "/Tracker0/pose",  # 动捕发布的刚体位置话题（需与你的动捕配置一致）
            self._mocap_pose_callback,
            self.qos_best_effort
        )
        # 2. 订阅外部目标位置（可通过ros2 topic pub手动发布测试）
        self.sub_target_pose = self.create_subscription(
            PoseStamped,
            "/drone_target_pose",
            self._target_pose_callback,
            self.qos_reliable
        )
        # 3. 发布位置控制指令给飞控（核心输出）
        self.pub_pos_cmd = self.create_publisher(
            Vector3,
            "/drone_pos_cmd",  # 飞控订阅的位置指令话题
            self.qos_reliable
        )

    def _init_data(self):
        """初始化数据存储变量"""
        self.current_pose = None  # 动捕反馈的实际位置
        # 默认目标位置（可通过/drone_target_pose话题覆盖）
        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = 0.0  # 初始原点
        self.target_pose.pose.position.y = 0.0
        self.target_pose.pose.position.z = 1.0  # 默认目标高度1m
        self.last_mocap_time = 0.0  # 最后接收动捕数据的时间（用于超时判断）

    def _mocap_pose_callback(self, msg):
        """接收动捕的无人机实际位置"""
        self.current_pose = msg
        self.last_mocap_time = self.get_clock().now().nanoseconds / 1e9

    def _target_pose_callback(self, msg):
        """接收外部设置的目标位置（如上位机/脚本发布）"""
        self.target_pose = msg
        self.get_logger().info(
            f"📌 更新目标位置：X={msg.pose.position.x:.2f}m | Y={msg.pose.position.y:.2f}m | Z={msg.pose.position.z:.2f}m"
        )

    def _control_loop(self):
        """核心控制逻辑：位置误差→飞控的姿态/油门目标"""
        # 1. 检查动捕数据有效性
        if self.current_pose is None:
            self.get_logger().warn("⚠️ 未收到动捕位置数据，暂停位置控制")
            return
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_mocap_time if self.last_mocap_time != 0 else 1.0/cfg.CONTROL_FREQ

        # 2. 提取实际位置和目标位置
        actual_x = self.current_pose.pose.position.x
        actual_y = self.current_pose.pose.position.y
        actual_z = self.current_pose.pose.position.z
        target_x = self.target_pose.pose.position.x
        target_y = self.target_pose.pose.position.y
        target_z = self.target_pose.pose.position.z

        # 3. PID计算：位置误差→姿态/油门目标
        pitch_target = self.pid_x.update(target_x, actual_x, dt)  # X位置误差→pitch角（rad）
        roll_target = self.pid_y.update(target_y, actual_y, dt)   # Y位置误差→roll角（rad）
        throttle_target = self.pid_z.update(target_z, actual_z, dt) + 500.0  # Z误差→油门（基础油门500）
        throttle_target = np.clip(throttle_target, 0.0, 1000.0)  # 油门限幅0-1000

        # 4. 构造并发布控制指令给飞控
        pos_cmd_msg = Vector3()
        pos_cmd_msg.x = roll_target    # 飞控的roll目标（rad）
        pos_cmd_msg.y = pitch_target   # 飞控的pitch目标（rad）
        pos_cmd_msg.z = throttle_target  # 飞控的throttle目标（0-1000）
        self.pub_pos_cmd.publish(pos_cmd_msg)

        # 5. 调试日志（可选）
        self.get_logger().debug(
            f"位置误差：X={target_x-actual_x:.2f}m | Y={target_y-actual_y:.2f}m | Z={target_z-actual_z:.2f}m | "
            f"输出目标：Roll={np.rad2deg(roll_target):.1f}° | Pitch={np.rad2deg(pitch_target):.1f}° | Throttle={throttle_target:.0f}"
        )

def main(args=None):
    """主函数：启动位置控制器节点"""
    rclpy.init(args=args)
    position_controller = DronePositionController()
    try:
        rclpy.spin(position_controller)
    except KeyboardInterrupt:
        position_controller.get_logger().info("🛑 用户中断，停止位置控制器")
    finally:
        position_controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
