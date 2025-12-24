#!/usr/bin/env python3
"""
无人机飞控配置文件
将所有参数集中在此处，便于统一管理和调试
"""

import numpy as np

# ========== 1. 系统基本参数 ==========
CONTROL_FREQ = 1000.0  # 控制频率（1000Hz）
DATA_TIMEOUT = 0.3     # 数据超时时间（0.3s）

# ========== 2. 遥控器参数 ==========
RC_DEAD_ZONE = 0.05    # 摇杆死区（5%）
THRUST_MID = 0.5       # 油门中位
IDLE_THROTTLE = 0.0    # 怠速油门

# ========== 3. 姿态控制PID参数 ==========
# 角度外环PID参数
PID_ROLL_ANGLE = {"kp": 0.25, "ki": 0.0, "kd": 0.00, "i_max": 0.03, "i_min": -0.03}
PID_PITCH_ANGLE = {"kp": 0.25, "ki": 0.0, "kd": 0.00, "i_max": 0.03, "i_min": -0.03}
PID_YAW_ANGLE = {"kp": 0.08, "ki": 0.0, "kd": 0.00, "i_max": 0.02, "i_min": -0.02}

# 速率内环PID参数
PID_ROLL_RATE = {"kp": 0.09, "ki": 0.00, "kd": 0.0008, "i_max": 0.01, "i_min": -0.01}
PID_PITCH_RATE = {"kp": 0.09, "ki": 0.00, "kd": 0.0008, "i_max": 0.01, "i_min": -0.01}
PID_YAW_RATE = {"kp": 0.09, "ki": 0.0, "kd": 0.01, "i_max": 0.01, "i_min": -0.01}

# ========== 4. 位置控制PID参数 ==========
# XY位置控制参数（位置误差→姿态角）
POSITION_XY_KP = 0.15     # 比例增益（rad/m）
POSITION_XY_KI = 0.02     # 积分增益
POSITION_XY_KD = 0.05     # 微分增益
POSITION_XY_INT_LIMIT = 0.5   # XY积分限幅
POSITION_XY_MAX_ANGLE = 0.8   # 最大倾角指令（rad）

# Z高度控制参数（高度误差→油门）
POSITION_Z_KP = 20.0      # 比例增益（throttle/m）
POSITION_Z_KI = 5.0       # 积分增益
POSITION_Z_KD = 10.0      # 微分增益
POSITION_Z_INT_LIMIT = 50.0   # Z积分限幅
POSITION_Z_THROTTLE_RANGE = 300  # 油门增量范围

# 位置控制滤波器参数
POSITION_FILTER_ALPHA_POS = 0.3  # 位置滤波系数
POSITION_FILTER_ALPHA_VEL = 0.2  # 速度滤波系数

# 位置控制死区参数
POSITION_DEADZONE_XY = 0.02  # XY平面死区（m）
POSITION_DEADZONE_Z = 0.05   # Z高度死区（m）

# 位置控制基础油门（悬停油门）
POSITION_BASE_THROTTLE = 1000.0

# ========== 5. 陀螺仪参数 ==========
GYRO_DEADBAND_ROLL_PITCH = 200  # 横滚/俯仰陀螺仪死区（rad/s）
GYRO_DEADBAND_YAW = 1.3         # 偏航陀螺仪死区（rad/s）

# ========== 6. 角度限制 ==========
MAX_ROLL_PITCH_ANG = np.deg2rad(70)  # 最大横滚/俯仰角（70°）
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
    [-1, 1, 1],   # 电机1：前左
    [1, -1, 1],   # 电机2：后左
    [1, 1, -1],   # 电机3：前右
    [-1, -1, -1]  # 电机4：后右
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
POSITION_CONTROL_FREQ = 100.0   # 位置控制频率（Hz）
POSITION_MOCAP_TIMEOUT = 0.5    # 动捕数据超时时间（s）