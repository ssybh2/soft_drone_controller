import numpy as np

# ========== 1. 角度外环PID参数（积分分离+小限幅） ==========
# 横滚/俯仰：降低KP，无积分，极小D项（减少噪声）
PID_ROLL_ANGLE = {"kp": 0.025, "ki": 0.0, "kd": 0.00, "i_max": 0.03, "i_min": -0.03}
PID_PITCH_ANGLE = {"kp": 0.025, "ki": 0.0, "kd": 0.00, "i_max": 0.03, "i_min": -0.03}
# 偏航：适度KP，无积分，极小D项
PID_YAW_ANGLE = {"kp": 0.02, "ki": 0.0, "kd": 0.00, "i_max": 0.02, "i_min": -0.02}

# ========== 2. 速率内环PID参数（柔和输出，无积分） ==========
# 横滚/俯仰：低KP，无积分，极小D项（避免抖动）
PID_ROLL_RATE = {"kp": 0.09, "ki": 0.00, "kd": 0.0008, "i_max": 0.01, "i_min": -0.01}
PID_PITCH_RATE = {"kp": 0.09, "ki": 0.00, "kd": 0.0008, "i_max": 0.01, "i_min": -0.01}
# 偏航：适度KP，无积分，极小D项
PID_YAW_RATE = {"kp": 0.8, "ki": 0.0, "kd": 0.01, "i_max": 0.01, "i_min": -0.01}

# ========== 3. 轴力矩缩放（默认1.0，无需调整） ==========
ROLL_SCALE = 1.9
PITCH_SCALE = 1.9
YAW_SCALE = 1.9

# ========== 12. 陀螺仪死区（单位：rad/s） ==========
GYRO_DEADBAND_ROLL_PITCH = 20
GYRO_DEADBAND_YAW = 15
# ========== 4. 电机补偿（无硬件偏差则全0） ==========
MOTOR_GAIN = [1.0, 1.0, 1.0, 1.0]  # 电机增益补偿
PITCH_COMP = [0, 0, 0, 0]          # 俯仰补偿
ROLL_COMP = [0, 0, 0, 0]           # 横滚补偿
YAW_COMP = [0, 0, 0, 0]            # 偏航补偿
AXIS_THRESHOLD = 0.05              # 轴阈值

# ========== 5. 速率缩放（默认1.0） ==========
K_ROLL_RATE = 1.0
K_PITCH_RATE = 1.0
K_YAW_RATE = 1.0

# ========== 6. 角度限制（适度缩小，更安全） ==========
MAX_ROLL_PITCH_ANG = np.deg2rad(70)  # 原70°→45°（减少大角度偏差）
MAX_YAW_ANGLE = np.deg2rad(45)       # 原45°→30°（偏航更柔和）

# ========== 7. 遥控器参数 ==========
RC_DEAD_ZONE = 0.05          # 摇杆死区（0.05=5%）
MAX_YAW_RATE = np.deg2rad(200)# 偏航最大速率（原400→200）
THRUST_MID = 0.5             # 油门中位
IDLE_THROTTLE = 0.0          # 怠速油门

# ========== 8. DSHOT参数（核心优化） ==========
DSHOT_IDLE_LOCK = 48         # 上锁时DSHOT值
DSHOT_IDLE_UNLOCK = 120      # 解锁怠速DSHOT值
DSHOT_MIN = 48               # DSHOT最小值
DSHOT_MAX = 2047             # DSHOT最大值
DSHOT_SCALE = 500          # 原350→80（大幅降低力矩放大效应）

# ========== 9. X型混控矩阵（验证过的正确矩阵） ==========
MIX_MATRIX = [
    [-1, 1, 1],  # 电机1：前左
    [1, -1, 1],  # 电机2：后左
    [1, 1, -1],   # 电机3：前右（修正原矩阵错误）
    [-1, -1,-1]    # 电机4：后右（修正原矩阵错误）
]

# ========== 10. 系统参数 ==========
CONTROL_FREQ = 1000.0  # 控制频率（400Hz）
DATA_TIMEOUT = 0.3    # 数据超时时间（0.6s）

# ========== 11. 解锁配置 ==========
UNLOCK_SWITCH_CHANNEL = "left_switch"
LOCK_SWITCH_VALUE = 2
UNLOCK_SWITCH_VALUES = [1, 3]



