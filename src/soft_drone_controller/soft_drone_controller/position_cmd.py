import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import ReadDJIRC
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def rad2deg(r: float) -> float:
    return float(r) * 180.0 / math.pi


def deg2rad(d: float) -> float:
    return float(d) * math.pi / 180.0


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


class FirstOrderYawLPF:
    """一阶低通：y_dot = (x - y) / tau"""

    def __init__(self, tau: float = 0.20):
        self.tau = float(tau)
        self.y = 0.0
        self.inited = False

    def reset(self, yaw0: float = 0.0):
        self.y = wrap_pi(yaw0)
        self.inited = True

    def update(self, x: float, dt: float) -> float:
        x = wrap_pi(x)
        if (not self.inited) or dt <= 1e-6:
            self.y = x
            self.inited = True
            return self.y
        a = clamp(dt / max(self.tau, 1e-3), 0.0, 1.0)
        e = wrap_pi(x - self.y)
        self.y = wrap_pi(self.y + a * e)
        return self.y


class SlewRateLimiter:
    def __init__(self, rate_limit: float):
        # rad/s
        self.rate = float(rate_limit)
        self.y = 0.0
        self.inited = False

    def reset(self, y0: float = 0.0):
        self.y = wrap_pi(y0)
        self.inited = True

    def update(self, x: float, dt: float) -> float:
        x = wrap_pi(x)
        if (not self.inited) or dt <= 1e-6:
            self.y = x
            self.inited = True
            return self.y
        e = wrap_pi(x - self.y)
        max_step = self.rate * dt
        e = clamp(e, -max_step, max_step)
        self.y = wrap_pi(self.y + e)
        return self.y


class PositionCmdNode(Node):
    """
    只负责 PATH(1) 航线发布，不影响：
      - MANUAL(2)：手动模式
      - HOLD(3)：定点悬停（由 position_control 的 HOLD 参数/逻辑负责）

    RC 右开关：1=PATH，3=HOLD，2=MANUAL
    发布：/pos_path = [x_wm, y_wm, z_wm, yaw_sp(rad)]
    """

    # PATH 子状态
    ST_IDLE = "IDLE"
    ST_APPROACH = "APPROACH"
    ST_CIRCLE = "CIRCLE"
    ST_RETURN = "RETURN"
    ST_LAND = "LAND"
    ST_DONE = "DONE"
    ST_ABORTED = "ABORTED"

    def __init__(self):
        super().__init__("position_cmd_node")

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_rc = self.create_subscription(ReadDJIRC, "/ecat/sn2228293/app1/read", self.rc_callback, qos_best_effort)
        self.sub_pose = self.create_subscription(PoseStamped, "/Tracker0/pose", self.pose_callback, qos_reliable)
        self.pub_path = self.create_publisher(Float64MultiArray, "/pos_path", 10)

        # ========== PATH 参数（只影响 PATH）==========
        self.declare_parameter("circle_radius", 0.80)          # m
        self.declare_parameter("circle_speed", 0.22)           # m/s（切向速度）
        self.declare_parameter("circle_laps", 1)               # N 圈
        self.declare_parameter("cmd_rate_hz", 300.0)            # 发布频率

        # 圆心模式：
        #   ENTRY：进入 PATH 的那一刻，当前位置作为圆心
        #   FIXED：使用固定圆心（center_x, center_y）
        self.declare_parameter("center_mode", "FIXED")
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)

        # 高度模式（避免一切“突然上冲”）
        #   KEEP_CURRENT：PATH 时 z 目标 = 当前动捕 z（不主动改变高度）
        #   FIXED：PATH 时 z 目标 = fixed_z
        self.declare_parameter("z_mode", "FIXED")
        self.declare_parameter("fixed_z", 0.70)

        # 安全/平滑：距离太大先 APPROACH，不要一上来就“硬追圆”
        self.declare_parameter("approach_tol", 1.5)           # 到达圆起点阈值(m)
        self.declare_parameter("return_tol", 0.25)             # 回到起点阈值(m)
        self.declare_parameter("land_z", 0.10)                 # 降落高度(m)
        self.declare_parameter("land_rate", 0.15)              # m/s

        # 轨迹推进逻辑：误差大就先别推进相位
        self.declare_parameter("stop_adv_dist", 1.60)          # e_xy > stop_adv_dist => 相位暂停
        self.declare_parameter("hard_abort_dist", 4.00)        # e_xy > hard_abort_dist => ABORT
        self.declare_parameter("min_speed", 0.03)              # 防止 v=0 导致 omega=0

        # Yaw 控制
        # yaw_mode:
        #   CONST：固定航向（始终面向同一方向）
        #   TANGENT：切向朝向（机头沿圆切线）
        #   POINT：始终指向目标点（用于 APPROACH）
        self.declare_parameter("yaw_mode", "CONST")
        self.declare_parameter("yaw_const_deg", 0.0)           # deg
        self.declare_parameter("yaw_rate_limit_deg_s", 20.0)   # deg/s
        self.declare_parameter("yaw_tau", 0.25)                # s（低通）

        # ========== 运行时状态 ==========
        self.right_switch = 3
        self.pose_ok = False
        self.now_wm = (0.0, 0.0, 0.0)
        self.last_pose_stamp = self.get_clock().now()

        self.state = self.ST_IDLE
        self.last_state = None

        self.center_wm = (0.0, 0.0)
        self.start_wm = (0.0, 0.0, 0.0)
        self.phase = 0.0
        self.lap_count = 0

        # yaw 平滑器
        yaw_rate = deg2rad(self.get_parameter("yaw_rate_limit_deg_s").value)
        self.yaw_rl = SlewRateLimiter(yaw_rate)
        self.yaw_lpf = FirstOrderYawLPF(self.get_parameter("yaw_tau").value)

        self.last_t = self.get_clock().now()

        hz = float(self.get_parameter("cmd_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self.timer_cb)

        self.get_logger().info("✅ position_cmd 启动：仅负责 PATH(1)，不会影响 MANUAL/HOLD。")

    # ---------------- callbacks ----------------
    def rc_callback(self, msg: ReadDJIRC):
        self.right_switch = int(msg.right_switch)

    def pose_callback(self, msg: PoseStamped):
        self.pose_ok = True
        p = msg.pose.position
        self.now_wm = (float(p.x), float(p.y), float(p.z))
        self.last_pose_stamp = self.get_clock().now()

    # ---------------- helpers ----------------
    def _transition(self, new_state: str, reason: str = ""):
        if new_state != self.state:
            self.last_state = self.state
            self.state = new_state
            if reason:
                self.get_logger().info(f"🔁 PATH 状态切换: {self.last_state} -> {self.state} | {reason}")
            else:
                self.get_logger().info(f"🔁 PATH 状态切换: {self.last_state} -> {self.state}")

    def _compute_circle_point(self, phase: float) -> tuple[float, float]:
        cx, cy = self.center_wm
        R = float(self.get_parameter("circle_radius").value)
        return (cx + R * math.cos(phase), cy + R * math.sin(phase))

    def _compute_tangent_yaw(self, phase: float) -> float:
        # 切向：沿着相位增加方向；切向方向是 (-sin, cos)
        return wrap_pi(math.atan2(math.cos(phase), -math.sin(phase)))

    def _yaw_target(self, yaw_mode: str, tgt_xy: tuple[float, float], now_xy: tuple[float, float], phase: float) -> float:
        if yaw_mode == "CONST":
            return wrap_pi(deg2rad(float(self.get_parameter("yaw_const_deg").value)))
        if yaw_mode == "TANGENT":
            return self._compute_tangent_yaw(phase)
        # POINT：指向目标点
        dx = tgt_xy[0] - now_xy[0]
        dy = tgt_xy[1] - now_xy[1]
        return wrap_pi(math.atan2(dy, dx))

    def _select_z_sp(self, now_z: float) -> float:
        z_mode = str(self.get_parameter("z_mode").value).upper()
        if z_mode == "FIXED":
            return float(self.get_parameter("fixed_z").value)
        return float(now_z)  # KEEP_CURRENT

    def _publish_path(self, tx: float, ty: float, tz: float, yaw_sp: float):
        msg = Float64MultiArray()
        msg.data = [float(tx), float(ty), float(tz), float(yaw_sp)]
        self.pub_path.publish(msg)

    # ---------------- main loop ----------------
    def timer_cb(self):
        now_t = self.get_clock().now()
        dt = (now_t - self.last_t).nanoseconds * 1e-9
        dt = clamp(dt, 0.0, 0.1)
        self.last_t = now_t

        # RC log（低频）
        if not hasattr(self, "_rc_log_t"):
            self._rc_log_t = now_t
        if (now_t - self._rc_log_t).nanoseconds * 1e-9 > 0.25:
            self.get_logger().info(f"🎛️ RC right_switch={self.right_switch} (1=PATH,3=HOLD,2=MANUAL) state={self.state}")
            self._rc_log_t = now_t

        # 只在 PATH(1) 发布轨迹
        if self.right_switch != 1:
            if self.state not in (self.ST_IDLE, self.ST_DONE):
                self.get_logger().warn("⏹️ 离开 PATH(1)：停止发布 /pos_path")
            self._transition(self.ST_IDLE)
            return

        if not self.pose_ok:
            self.get_logger().warn("⚠️ PATH(1) 但未收到动捕 Pose，等待…")
            return

        pose_age = (now_t - self.last_pose_stamp).nanoseconds * 1e-9
        if pose_age > 0.5:
            self.get_logger().error(f"🛑 动捕超时 {pose_age:.2f}s，停止发布 /pos_path")
            self._transition(self.ST_ABORTED, "mocap timeout")
            return

        x, y, z = self.now_wm
        now_xy = (x, y)

        # 第一次进入 PATH：初始化
        if self.state == self.ST_IDLE:
            cmode = str(self.get_parameter("center_mode").value).upper()
            if cmode == "FIXED":
                cx = float(self.get_parameter("center_x").value)
                cy = float(self.get_parameter("center_y").value)
                self.center_wm = (cx, cy)
                self.get_logger().warn(f"🎯 使用固定圆心 Wm=({cx:.2f}, {cy:.2f})")
            else:
                self.center_wm = (x, y)
                self.get_logger().info(f"🎯 以进入 PATH 时刻位置作为圆心 Wm=({x:.2f}, {y:.2f})")

            self.start_wm = (x, y, z)
            self.phase = 0.0
            self.lap_count = 0

            yaw0 = self._yaw_target("CONST", now_xy, now_xy, self.phase)
            self.yaw_rl.reset(yaw0)
            self.yaw_lpf.reset(yaw0)

            self._transition(self.ST_APPROACH, "enter PATH")

            R = float(self.get_parameter("circle_radius").value)
            v = max(float(self.get_parameter("circle_speed").value), float(self.get_parameter("min_speed").value))
            omega = v / max(R, 1e-3)
            self.get_logger().info(
                f"🚀 进入 PATH(1)：圆心 Wm=({self.center_wm[0]:.2f},{self.center_wm[1]:.2f}) | "
                f"R={R:.2f} v={v:.2f} omega={omega:.2f}rad/s | "
                f"yaw_mode={self.get_parameter('yaw_mode').value} | "
                f"z_mode={self.get_parameter('z_mode').value}"
            )

        R = float(self.get_parameter("circle_radius").value)
        v = max(float(self.get_parameter("circle_speed").value), float(self.get_parameter("min_speed").value))
        omega = v / max(R, 1e-3)
        laps_target = int(self.get_parameter("circle_laps").value)

        tz = self._select_z_sp(z)

        if self.state == self.ST_APPROACH:
            tgt_xy = self._compute_circle_point(0.0)
            e_xy = math.hypot(tgt_xy[0] - x, tgt_xy[1] - y)

            if e_xy > float(self.get_parameter("hard_abort_dist").value):
                self._transition(self.ST_ABORTED, f"too far e_xy={e_xy:.2f}m")
            else:
                if e_xy < float(self.get_parameter("approach_tol").value):
                    self._transition(self.ST_CIRCLE, "approach done")

                yaw_mode = str(self.get_parameter("yaw_mode").value).upper()
                yaw_tgt = self._yaw_target("POINT" if yaw_mode != "CONST" else "CONST", tgt_xy, now_xy, self.phase)
                yaw_rl = self.yaw_rl.update(yaw_tgt, dt)
                yaw_out = self.yaw_lpf.update(yaw_rl, dt)
                self._publish_path(tgt_xy[0], tgt_xy[1], tz, yaw_out)

        elif self.state == self.ST_CIRCLE:
            tgt_xy = self._compute_circle_point(self.phase)
            e_xy = math.hypot(tgt_xy[0] - x, tgt_xy[1] - y)

            stop_adv = float(self.get_parameter("stop_adv_dist").value)
            adv = 1 if e_xy <= stop_adv else 0

            if e_xy > float(self.get_parameter("hard_abort_dist").value):
                self._transition(self.ST_ABORTED, f"too far e_xy={e_xy:.2f}m")
            else:
                if adv:
                    self.phase += omega * dt
                    if self.phase >= 2.0 * math.pi:
                        self.phase -= 2.0 * math.pi
                        self.lap_count += 1
                        self.get_logger().info(f"🏁 完成第 {self.lap_count}/{laps_target} 圈")
                        if self.lap_count >= laps_target:
                            self._transition(self.ST_RETURN, "laps done")

                yaw_mode = str(self.get_parameter("yaw_mode").value).upper()
                yaw_tgt = self._yaw_target(yaw_mode, tgt_xy, now_xy, self.phase)
                yaw_rl = self.yaw_rl.update(yaw_tgt, dt)
                yaw_out = self.yaw_lpf.update(yaw_rl, dt)
                self._publish_path(tgt_xy[0], tgt_xy[1], tz, yaw_out)

        elif self.state == self.ST_RETURN:
            sx, sy, _ = self.start_wm
            tgt_xy = (sx, sy)
            e_xy = math.hypot(tgt_xy[0] - x, tgt_xy[1] - y)

            yaw_tgt = self._yaw_target("CONST", tgt_xy, now_xy, self.phase)
            yaw_rl = self.yaw_rl.update(yaw_tgt, dt)
            yaw_out = self.yaw_lpf.update(yaw_rl, dt)

            tz_ret = self._select_z_sp(z) if str(self.get_parameter("z_mode").value).upper() != "FIXED" else float(self.get_parameter("fixed_z").value)
            self._publish_path(tgt_xy[0], tgt_xy[1], tz_ret, yaw_out)

            if e_xy < float(self.get_parameter("return_tol").value):
                self._transition(self.ST_LAND, "back to start")

        elif self.state == self.ST_LAND:
            sx, sy, _ = self.start_wm
            tgt_xy = (sx, sy)

            yaw_tgt = self._yaw_target("CONST", tgt_xy, now_xy, self.phase)
            yaw_rl = self.yaw_rl.update(yaw_tgt, dt)
            yaw_out = self.yaw_lpf.update(yaw_rl, dt)

            if not hasattr(self, "_land_z"):
                self._land_z = float(z)
            land_rate = float(self.get_parameter("land_rate").value)
            self._land_z = max(self._land_z - land_rate * dt, float(self.get_parameter("land_z").value))

            self._publish_path(tgt_xy[0], tgt_xy[1], self._land_z, yaw_out)

            if self._land_z <= float(self.get_parameter("land_z").value) + 1e-3:
                self._transition(self.ST_DONE, "land done")
                self.get_logger().warn("✅ PATH 完成：已返回起点并下降到 land_z。建议切回 MANUAL 并锁桨。")

        elif self.state == self.ST_ABORTED:
            if not hasattr(self, "_abort_log_once"):
                self.get_logger().error("🛑 PATH 已 ABORT：停止发布 /pos_path。请切回 MANUAL/HOLD，再重新进入 PATH。")
                self._abort_log_once = True
            return

        elif self.state in (self.ST_DONE, self.ST_IDLE):
            return


def main(args=None):
    rclpy.init(args=args)
    node = PositionCmdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
