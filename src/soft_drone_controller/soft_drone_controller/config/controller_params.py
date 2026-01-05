#!/usr/bin/env python3
"""
无人机飞控配置文件
将所有参数集中在此处，便于统一管理和调试
"""

import numpy as np

# ========== 1. 系统基本参数 ==========
CONTROL_FREQ = 1000.0  # 控制频率（1000Hz）
DATA_TIMEOUT = 0.3     # 数据超时时间（0.3s）
Kp_ANGLE = 0.5

# ========== 2. 遥控器参数 ==========
#RC_DEAD_ZONE = 0.1    # 摇杆死区（5%）
RC_DEAD_ZONE_ROLL = 0.03
RC_DEAD_ZONE_PITCH = 0.03
RC_DEAD_ZONE_YAW = 0.03
RC_DEAD_ZONE_THROTTLE = 0.05
THRUST_MID = 0.5       # 油门中位
IDLE_THROTTLE = 0.0    # 怠速油门
ATTITUDE_TIME_CONSTANT = 0.1

# ========== 3. 姿态控制PID参数 ==========
# 角度外环PID参数
PID_ROLL_ANGLE = {"kp": 0.00002, "ki": 0.0, "kd": 0.001, "i_max": 0.03, "i_min": -0.03}
PID_PITCH_ANGLE = {"kp": 0.00002, "ki": 0.0, "kd": 0.001, "i_max": 0.03, "i_min": -0.03}
PID_YAW_ANGLE = {"kp": 0.12, "ki": 0.0, "kd": 0.00, "i_max": 0.02, "i_min": -0.02}

# 速率内环PID参数
PID_ROLL_RATE = {"kp": 0.0405, "ki": 0.01, "kd": 0.001, "i_max": 0.3, "i_min": -0.3}
PID_PITCH_RATE = {"kp": 0.0405, "ki": 0.01, "kd": 0.001, "i_max": 0.3, "i_min": -0.3}
PID_YAW_RATE = {"kp": 0.05, "ki": 0.0, "kd": 0.01*0.08, "i_max": 0.2, "i_min": -0.2}

# ========== 4. 位置控制PID参数 ==========
DSHOT_MIN = 48
DSHOT_MAX = 2047
TAKEOFF_CLIMB_RATE = 0.15  # 起飞最大爬升速度（m/s）
POSITION_DEFAULT_HEIGHT = 0.7
HOVER_DSHOT_VALUE = 760
HOVER_THROTTLE_RATIO = (HOVER_DSHOT_VALUE - DSHOT_MIN) / (DSHOT_MAX - DSHOT_MIN)
MIN_DESCEND_DSHOT_VALUE = 720
MIN_DESCEND_THROTTLE_RATIO = (MIN_DESCEND_DSHOT_VALUE - DSHOT_MIN) / (DSHOT_MAX - DSHOT_MIN)

# XY位置控制参数（位置误差→姿态角）
POSITION_XY_KP = 0.2     # 比例增益（rad/m）
POSITION_XY_KI = 0.0     # 积分增益
POSITION_XY_KD = 0.0   # 微分增益
POSITION_XY_INT_LIMIT = 0.2   # XY积分限幅
POSITION_XY_MAX_ANGLE = 0.2   # 最大倾角指令（rad）

# ======== ✅ 新增：PATH 模式专用“更紧”的跟踪增益（不影响 HOLD/MANUAL）========
# 说明：
# - PATH 的目标点在移动（动态目标），同样的 KP 往往会“追不上”，所以给 PATH 单独更大 KP 更合理
# - 你可以先用 0.35；如果仍慢，可以试 0.45；如果开始抖，再往回退
PATH_POSITION_XY_KP = 0.65

# Z高度控制参数（高度误差→油门）
POSITION_Z_KP = 0.023      # 比例增益（throttle/m）
POSITION_Z_KI = 0.0       # 积分增益
POSITION_Z_KD = 0.0      # 微分增益
POSITION_Z_INT_LIMIT = 0.1   # Z积分限幅
#POSITION_Z_THROTTLE_RANGE = 300  # 油门增量范围

# 速度环 (输出期望加速度)
VELOCITY_XY_KP = 0.17
VELOCITY_XY_KI = 0.0
VELOCITY_XY_KD = 0.04
VELOCITY_XY_INT_LIMIT = 0.25 # 加速度上限
VELOCITY_Z_KP = 0.07
VELOCITY_Z_KI = 0.0
VELOCITY_Z_KD = 0.05
VELOCITY_Z_INT_LIMIT = 1.0 # 加速度上限

# 位置控制滤波器参数
POSITION_FILTER_ALPHA_POS = 0.3  # 位置滤波系数
POSITION_FILTER_ALPHA_VEL = 0.2  # 速度滤波系数

# 位置控制死区参数
POSITION_DEADZONE_XY = 0.02  # XY平面死区（m）
POSITION_DEADZONE_Z = 0.05   # Z高度死区（m）

# 位置控制基础油门（悬停油门）
POSITION_BASE_THROTTLE = 1000.0

# ========== 5. 陀螺仪参数 ==========
GYRO_DEADBAND_ROLL_PITCH = 0.0  # 横滚/俯仰陀螺仪死区（rad/s）
GYRO_DEADBAND_YAW = 0.0         # 偏航陀螺仪死区（rad/s）

# ========== 6. 角度限制 ==========
MAX_ROLL_PITCH_ANG = np.deg2rad(30)  # 最大横滚/俯仰角（70°）
MAX_YAW_ANGLE = np.deg2rad(45)       # 最大偏航角（45°）
MAX_YAW_RATE = np.deg2rad(200)       # 最大偏航速率（200°/s）

# ========== 7. DSHOT电机参数 ==========
DSHOT_IDLE_LOCK = 48        # 上锁时DSHOT值
DSHOT_IDLE_UNLOCK = 120     # 解锁怠速DSHOT值
DSHOT_MIN = 48              # DSHOT最小值
DSHOT_MAX = 2047            # DSHOT最大值
DSHOT_SCALE = 500           # DSHOT缩放因子

# ========== 8. 电机混控矩阵（X型布局） ==========
MIX_MATRIX = [
    [-1, 1, 1],   # 电机1
    [ 1,-1, 1],   # 电机2
    [ 1, 1,-1],   # 电机3
    [-1,-1,-1]  # 电机4
]

# ========== 9. 电机补偿参数 ==========
MOTOR_GAIN = [1.0, 1.0, 1.0, 1.0]  # 电机增益补偿
PITCH_COMP = [0, 0, 0, 0]          # 俯仰补偿
ROLL_COMP = [0, 0, 0, 0]           # 横滚补偿
YAW_COMP = [0, 0, 0, 0]            # 偏航补偿
AXIS_THRESHOLD = 0.05              # 轴阈值

# ========== 10. 轴力矩缩放 ==========
ROLL_SCALE = 1.9    # 横滚力矩缩放
PITCH_SCALE = 1.9   # 俯仰力矩缩放
YAW_SCALE = 1.9     # 偏航力矩缩放

# ========== 11. 控制模式配置 ==========
# 左开关位置定义：
#   1(上)：解锁 + 手动模式
#   3(中)：解锁 + 位置控制模式
#   2(下)：上锁
#UNLOCK_SWITCH_VALUES = [1, 3]  # 解锁位置列表
UNLOCK_SWITCH_CHANNEL = "left_switch"
LOCK_SWITCH_VALUE = 2           # 上锁位置
MANUAL_MODE_VALUE = 1           # 手动模式位置
POSITION_MODE_VALUE = 3         # 位置控制模式位置

# ========== 12. 位置控制其他参数 ==========
POSITION_CMD_TIMEOUT = 0.2      # 位置指令超时时间（s）
POSITION_CONTROL_FREQ = 1000.0   # 位置控制频率（Hz）
POSITION_MOCAP_TIMEOUT = 0.5    # 动捕数据超时时间（s）

# ======== ✅ 新增：PATH 模式专用姿态输出增益（只给 PATH 更“用力”，不影响 HOLD）========
# 你当前 position_control 里 ATTITUDE_CMD_GAIN=2.6，这里给 PATH 用 3.3
# 如果你觉得还慢，可以 3.8；如果开始抖或者过冲，就往回收
PATH_ATTITUDE_CMD_GAIN = 3.8
