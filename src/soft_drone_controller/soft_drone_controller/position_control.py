#!/usr/bin/env python3
"""
无人机位置控制器（简化版） - 带详细调试信息
修正了动捕坐标系到无人机机体坐标系的转换
增加角度（度）输出话题
修复了高度控制问题
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
        
        # 详细调试信息发布定时器（10Hz）
        self.debug_timer = self.create_timer(0.1, self._publish_detailed_debug)
        
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
        
        # 发布姿态指令给飞控（弧度）
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
        
        # ========== 新增：详细调试信息发布者 ==========
        self.pub_position_details = self.create_publisher(
            Vector3,
            "/position_control_details",
            qos_reliable
        )
        
        self.pub_pid_debug = self.create_publisher(
            Vector3,
            "/position_pid_debug",
            qos_reliable
        )
        
        self.pub_control_output = self.create_publisher(
            Vector3,
            "/position_control_output",
            qos_reliable
        )
        
        # ========== 新增：角度输出话题 ==========
        # 用于在Foxglove中直接查看角度值（度）
        self.pub_attitude_cmd_deg = self.create_publisher(
            Vector3,
            "/attitude_position_cmd_deg",
            qos_reliable
        )
        
        self.pub_control_output_deg = self.create_publisher(
            Vector3,
            "/position_control_output_deg",
            qos_reliable
        )
        
    def _init_data(self):
        """初始化数据存储"""
        self.current_pos = None
        self.target_pos = np.array([0.0,0.0,1.0])  # 默认目标
        self.filtered_pos = None
        self.filtered_vel = None
        self.last_mocap_time = 0
        self.last_control_time = 0
        self.mocap_active = False
        self.control_count = 0
        
        # 调试数据
        self.debug_data = {
            "position_error": np.zeros(3),
            "control_output": np.zeros(3),  # 弧度
            "control_output_deg": np.zeros(3),  # 度（新增）
            "pid_terms_xy": np.zeros(3),  # P, I, D
            "pid_terms_z": np.zeros(3),   # P, I, D
            "last_debug_time": 0
        }
        
    def _init_pid_state(self):
        """初始化PID状态"""
        self.error_integral_xy = np.array([0.0, 0.0])
        self.error_integral_z = 0.0
        self.prev_error_xy = np.array([0.0, 0.0])
        self.prev_error_z = 0.0
        self.last_time = None
        
        # PID项记录
        self.p_term_xy = np.array([0.0, 0.0])
        self.i_term_xy = np.array([0.0, 0.0])
        self.d_term_xy = np.array([0.0, 0.0])
        
        self.p_term_z = 0.0
        self.i_term_z = 0.0
        self.d_term_z = 0.0
        
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
        
    def _mocap_to_body_coordinates(self, mocap_pos):
        """将动捕坐标系转换到无人机机体坐标系
        
        动捕坐标系(Nokov): 
          - Y轴: 向前
          - X轴: 向右  
          - Z轴: 向上
        
        无人机机体坐标系(FRD):
          - X轴: 向前
          - Y轴: 向右
          - Z轴: 向下
          
        修正：高度使用动捕坐标系（Z向上为正），但为了保持一致性，我们转换Z但不改变符号
        """
        body_x = -mocap_pos[1]    # 动捕Y → 机体X（前）
        body_y = mocap_pos[0]     # 动捕X → 机体Y（右）
        body_z = mocap_pos[2]     # 动捕Z → 保持原值（动捕坐标系）
        return np.array([body_x, body_y, body_z])
        
    def _position_pid_control(self, dt):
        """位置PID控制核心算法（修正坐标转换和高度控制）"""
        if dt <= 0 or self.current_pos is None or self.filtered_pos is None:
            return 0.0, 0.0, self.cfg.POSITION_BASE_THROTTLE
            
        # 使用滤波后的位置
        current_pos = self.filtered_pos
        
        # ========== XY坐标转换 ==========
        # 将动捕坐标转换为机体坐标（仅XY）
        current_body = self._mocap_to_body_coordinates(current_pos)
        target_body = self._mocap_to_body_coordinates(self.target_pos)
        
        # 计算XY位置误差（在机体坐标系中）
        error_body = target_body - current_body
        
        # 分解XY误差
        error_x = error_body[0]  # 前向误差（机体X）
        error_y = error_body[1]  # 右向误差（机体Y）
        
        # ========== 修正：高度误差计算 ==========
        # 高度直接使用动捕坐标系：Z向上为正
        # 目标在上方时，error_z应该为正
        error_z = self.target_pos[2] - current_pos[2]
        
        # 在机体坐标系中计算XY误差
        error_xy = np.array([error_x, error_y])

        # 添加高度误差详细调试
        if self.control_count % 20 == 0:  # 每20个控制周期输出一次，避免日志过多
            self.get_logger().info(
                f"📊 高度误差调试 | "
                f"动捕高度={current_pos[2]:.2f}m | "
                f"动捕目标高度={self.target_pos[2]:.2f}m | "
                f"高度误差={error_z:.2f}m | "
                f"基础油门={self.cfg.POSITION_BASE_THROTTLE:.0f}"
            )
        
        # 保存误差用于调试（动捕坐标系）
        self.debug_data["position_error"] = np.array([
            self.target_pos[0] - current_pos[0],
            self.target_pos[1] - current_pos[1],
            error_z  # 使用修正后的高度误差
        ])
        
        # 死区处理（在机体坐标系中）
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
            # 将动捕速度也转换到机体坐标系
            mocap_vel = self.filtered_vel
            body_vel = self._mocap_to_body_coordinates(mocap_vel) - self._mocap_to_body_coordinates(np.zeros(3))
            derivative_xy = -body_vel[:2]  # 取负号，因为速度与误差方向相反
        else:
            derivative_xy = np.zeros(2)
            
        # PID计算
        self.p_term_xy = self.cfg.POSITION_XY_KP * error_xy
        self.i_term_xy = self.cfg.POSITION_XY_KI * self.error_integral_xy
        self.d_term_xy = self.cfg.POSITION_XY_KD * derivative_xy
        
        output_xy = self.p_term_xy + self.i_term_xy + self.d_term_xy
        
        # 输出限幅
        output_norm = np.linalg.norm(output_xy)
        if output_norm > self.cfg.POSITION_XY_MAX_ANGLE:
            output_xy = output_xy / output_norm * self.cfg.POSITION_XY_MAX_ANGLE
            
        # ========== Z轴PID控制（高度控制） ==========
        # 重置高度积分项（避免之前的错误累积）
        if abs(error_z) < 0.01:  # 如果高度误差很小，重置积分项
            self.error_integral_z = 0.0
            
        self.error_integral_z += error_z * dt
        self.error_integral_z = np.clip(
            self.error_integral_z, 
            -self.cfg.POSITION_Z_INT_LIMIT, 
            self.cfg.POSITION_Z_INT_LIMIT
        )
        
        # 微分项（使用动捕坐标系的速度）
        if self.filtered_vel is not None:
            derivative_z = -self.filtered_vel[2]  # 使用动捕Z速度，取负号
        else:
            derivative_z = 0.0
            
        # PID计算（油门增量）
        # 确保PID参数是正数：正误差（目标在上方）应该增加油门
        self.p_term_z = self.cfg.POSITION_Z_KP * error_z
        self.i_term_z = self.cfg.POSITION_Z_KI * self.error_integral_z
        self.d_term_z = self.cfg.POSITION_Z_KD * derivative_z
        
        throttle_increment = self.p_term_z + self.i_term_z + self.d_term_z

        # ========== 新增：油门计算调试 ==========
        if self.control_count % 20 == 0:
            self.get_logger().info(
                f"📊 油门计算调试 | "
                f"KP={self.cfg.POSITION_Z_KP} | "
                f"P项={self.p_term_z:.0f} | "
                f"I项={self.i_term_z:.0f} | "
                f"D项={self.d_term_z:.0f} | "
                f"油门增量={throttle_increment:.0f} | "
                f"范围限制=[{-self.cfg.POSITION_Z_THROTTLE_RANGE}, {self.cfg.POSITION_Z_THROTTLE_RANGE}]"
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
        
        # ========== 新增：最终油门调试 ==========
        if self.control_count % 20 == 0:
            self.get_logger().info(
                f"📊 最终油门调试 | "
                f"基础油门={self.cfg.POSITION_BASE_THROTTLE:.0f} | "
                f"油门增量={throttle_increment:.0f} | "
                f"最终油门={throttle_output:.0f}"
            )
        
        # ========== 生成控制指令 ==========
        # 注意：机体坐标系中：
        #   - error_x（前向误差） → pitch指令（俯仰）
        #   - error_y（右向误差） → roll指令（横滚）
        #   
        # 如果error_x为正（目标在前方） → 需要负pitch（前倾）来向前飞
        # 如果error_y为正（目标在右方） → 需要正roll（右倾）来向右飞
        
        # 使用PID输出生成指令（output_xy已经考虑了误差方向）
        roll_cmd = float(output_xy[1])   # Y分量 → roll指令
        pitch_cmd = float(output_xy[0])  # X分量 → pitch指令
        
        # 保存控制输出用于调试（弧度）
        self.debug_data["control_output"] = np.array([roll_cmd, pitch_cmd, throttle_output])
        
        # 保存角度输出（度）
        self.debug_data["control_output_deg"] = np.array([
            np.rad2deg(roll_cmd),
            np.rad2deg(pitch_cmd),
            throttle_output  # 油门值保持不变
        ])
        
        # 保存PID项用于调试
        self.debug_data["pid_terms_xy"] = np.array([
            np.linalg.norm(self.p_term_xy),
            np.linalg.norm(self.i_term_xy),
            np.linalg.norm(self.d_term_xy)
        ])
        
        self.debug_data["pid_terms_z"] = np.array([
            abs(self.p_term_z),
            abs(self.i_term_z),
            abs(self.d_term_z)
        ])
        
        # 保存误差
        self.prev_error_xy = error_xy
        self.prev_error_z = error_z
        
        # 返回控制指令
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
        
        # 发布控制指令给飞控（弧度）
        cmd_msg = Vector3()
        cmd_msg.x = roll_cmd
        cmd_msg.y = pitch_cmd
        cmd_msg.z = throttle_cmd
        self.pub_attitude_cmd.publish(cmd_msg)
        
        # 发布角度指令用于调试（度）
        cmd_deg_msg = Vector3()
        cmd_deg_msg.x = np.rad2deg(roll_cmd)
        cmd_deg_msg.y = np.rad2deg(pitch_cmd)
        cmd_deg_msg.z = throttle_cmd  # 油门保持不变
        self.pub_attitude_cmd_deg.publish(cmd_deg_msg)
        
        # 发布控制输出调试信息（弧度）
        output_msg = Vector3()
        output_msg.x = roll_cmd
        output_msg.y = pitch_cmd
        output_msg.z = throttle_cmd
        self.pub_control_output.publish(output_msg)
        
        # 发布控制输出调试信息（度）
        output_deg_msg = Vector3()
        output_deg_msg.x = np.rad2deg(roll_cmd)
        output_deg_msg.y = np.rad2deg(pitch_cmd)
        output_deg_msg.z = throttle_cmd
        self.pub_control_output_deg.publish(output_deg_msg)
        
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
        
    def _publish_detailed_debug(self):
        """发布详细调试信息"""
        if self.current_pos is None or self.target_pos is None:
            return
            
        # 位置详细信息
        details_msg = Vector3()
        details_msg.x = self.target_pos[0]  # 目标X（动捕坐标系）
        details_msg.y = self.current_pos[0] # 当前X（动捕坐标系）
        details_msg.z = self.target_pos[0] - self.current_pos[0]  # X误差（动捕坐标系）
        self.pub_position_details.publish(details_msg)
        
        # PID调试信息（每0.5秒发布一次）
        current_time = time.time()
        if current_time - self.debug_data["last_debug_time"] > 0.5:
            pid_debug_msg = Vector3()
            # XY PID项
            pid_debug_msg.x = self.debug_data["pid_terms_xy"][0]  # P项
            pid_debug_msg.y = self.debug_data["pid_terms_xy"][1]  # I项
            pid_debug_msg.z = self.debug_data["pid_terms_xy"][2]  # D项
            self.pub_pid_debug.publish(pid_debug_msg)
            
            # 日志输出
            if self.mocap_active:
                # 计算机体坐标系下的误差用于调试
                if self.current_pos is not None and self.target_pos is not None:
                    current_body = self._mocap_to_body_coordinates(self.current_pos)
                    target_body = self._mocap_to_body_coordinates(self.target_pos)
                    error_body = target_body - current_body
                    
                    self.get_logger().info(
                        f"🎯 位置控制详情 | "
                        f"动捕坐标: [{self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}, {self.current_pos[2]:.2f}] | "
                        f"机体坐标XY: [{current_body[0]:.2f}, {current_body[1]:.2f}] | "
                        f"机体误差XY: [{error_body[0]:.2f}, {error_body[1]:.2f}] | "
                        f"高度误差: {self.debug_data['position_error'][2]:.2f}m | "
                        f"控制输出(度): R={self.debug_data['control_output_deg'][0]:.1f}°, P={self.debug_data['control_output_deg'][1]:.1f}°, T={self.debug_data['control_output_deg'][2]:.0f}"
                    )
            
            self.debug_data["last_debug_time"] = current_time
        
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
            
        # 检查高度合理性（动捕坐标系中Z向上）
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
                f"动捕位置: [{self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}, {self.current_pos[2]:.2f}]m"
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