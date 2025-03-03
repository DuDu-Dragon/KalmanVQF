import numpy as np

# Load data (assuming you have a .mat file and use scipy.io)
from pexpect import EOF
from scipy.io import loadmat

path = 'Simulations/data_mat/02_undisturbed_slow_rotation_B.mat'

def EKF (path):

    data = loadmat(path)
    imu_acc = data['imu_acc']
    imu_gyr = data['imu_gyr']
    imu_mag = data['imu_mag']
    opt_quat = data['opt_quat']
    sampling_rate = 7.0/2000

    ax = imu_acc[:, 0]
    ay = imu_acc[:, 1]
    az = imu_acc[:, 2]
    gx = imu_gyr[:, 0]
    gy = imu_gyr[:, 1]
    gz = imu_gyr[:, 2]
    mx = imu_mag[:, 0]
    my = imu_mag[:, 1]
    mz = imu_mag[:, 2]

    # Initialization
    e0, e1, e2, e3 = 1.0, 0.0, 0.0, 0.0
    pb, qb, rb = 0.0, 0.0, 0.0

    # Covariance matrix
    P = np.zeros((7, 7))
    # Process noise matrix
    #Q = np.diag(np.array([[1, 1, 1, 1] * 0.0005, [1, 1, 1] * 0.0001]) ** 2)

    Q1 = np.diag([0.0005**2] * 4)
    Q2 = np.diag([0.0001**2] * 3)
    #Q = np.block([Q1,np.zeros((4,3))],[np.zeros((3,4)),Q2])
    Q = np.block([[Q1, np.zeros((4, 3))], [np.zeros((3, 4)), Q2]])

    # Measurement noise matrix
    R = np.diag([0.0045] * 3 + [2.5] * 3)

    print("Q:", Q)
    print("R:", R)
    # State space initialization
    x = np.array([e0, e1, e2, e3, pb, qb, rb])

    # Sampling rate (you may need to set this value)
    roll, pitch, yaw = np.zeros(len(imu_acc)), np.zeros(len(imu_acc)), np.zeros(len(imu_acc))
    gt_roll, gt_pitch, gt_yaw = np.zeros(len(imu_acc)), np.zeros(len(imu_acc)), np.zeros(len(imu_acc))
    angle_errors = np.zeros(len(imu_acc))

    def euler_to_rotation_matrix(roll, pitch, yaw):
        """将欧拉角 (roll, pitch, yaw) 转换为旋转矩阵"""
        # 绕X轴旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        # 绕Y轴旋转矩阵
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        # 绕Z轴旋转矩阵
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # 按ZYX顺序组合旋转矩阵
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R


    def rotation_matrix_to_angle(R):
        """从旋转矩阵 R 提取旋转角度"""
        # 防止数值误差导致的问题，限制范围在 [-1, 1] 内
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        return angle

    for i in range(1, len(imu_acc)):
        # Sample time
        dt = 7.0 / 2000.0

        #################
        # PREDICTION STEP
        #################

        # Read data from gyro
        p, q, r = gx[i], gy[i], gz[i]

        # Input vector
        u = np.array([p, q, r, pb, qb, rb])

        # Transition matrix
        F = 0.5 * np.array([
            [-e1, -e2, -e3, e1, e2, e3],
            [e0, -e3, e2, -e0, e3, -e2],
            [e3, e0, -e1, -e3, -e0, e1],
            [-e2, e1, e0, e2, -e1, -e0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # State space estimate
        x = x + dt * (F @ u)

        # Update quaternions value
        e0, e1, e2, e3 = x[0], x[1], x[2], x[3]

        # Normalize quaternions
        norm = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
        e0, e1, e2, e3 = e0 / norm, e1 / norm, e2 / norm, e3 / norm
        x[0:4] = e0, e1, e2, e3

        # Jacobian matrix A - partial derivatives dF/du
        A = 0.5 * np.array([
            [0, -(p - pb), -(q - qb), -(r - rb), e1, e2, e3],
            [(p - pb), 0, (r - rb), -(q - qb), -e0, e3, -e2],
            [(q - qb), -(r - rb), 0, (p - pb), -e3, -e0, e1],
            [(r - rb), (q - qb), -(p - pb), 0, e2, -e1, -e0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        # Covariance matrix estimate
        P += dt * (A @ P + P @ A.T + Q)

        # Magnetometer model estimation
        dm = 0  # magnetic declination angle
        msin = np.sin(np.radians(dm))
        mcos = np.cos(np.radians(dm))

        m = np.array([
            msin * (2 * e0 * e3 + 2 * e1 * e2) - mcos * (2 * e2**2 + 2 * e3**2 - 1),
            -mcos * (2 * e0 * e3 - 2 * e1 * e2) - msin * (2 * e1**2 + 2 * e3**2 - 1),
            mcos * (2 * e0 * e2 + 2 * e1 * e3) - msin * (2 * e0 * e1 - 2 * e2 * e3)
        ])

        # Accelerometer model estimation
        a = -np.array([2 * (e1 * e3 - e0 * e2), 2 * (e0 * e1 + e2 * e3), 1 - 2 * (e1**2 + e2**2)])

        # Models matrix
        z = np.hstack([a, m])

        # Measure from acc and mag
        y = np.array([-ax[i], -ay[i], -az[i], mx[i], my[i], mz[i]])

        # Normalize measurements
        y[:3] /= np.linalg.norm(y[:3])
        y[3:] /= np.linalg.norm(y[3:])

        #################
        # UPDATE STEP
        #################

        # Jacobian matrix H - partial derivatives dy/dx
        H = 2 * np.array([
            [e2, -e3, e0, -e1, 0, 0, 0],
            [-e1, -e0, -e3, -e2, 0, 0, 0],
            [0, 2 * e1, 2 * e2, 0, 0, 0, 0],
            [e3 * msin, e2 * msin, e1 * msin - 2 * e2 * mcos, e0 * msin - 2 * e3 * mcos, 0, 0, 0],
            [-e3 * mcos, e2 * mcos - 2 * e1 * msin, e1 * mcos, -e0 * mcos - 2 * e3 * msin, 0, 0, 0],
            [e2 * mcos - e1 * msin, e3 * mcos - e0 * msin, e0 * mcos + e3 * msin, e1 * mcos + e2 * msin, 0, 0, 0]
        ])

        # Gain [7, 6]
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        # Covariance matrix [7, 7]
        P = (np.eye(7) - K @ H) @ P
        # State space 7
        x += K @ (y - z)

        # Update quaternions and biases
        e0, e1, e2, e3 = x[0], x[1], x[2], x[3]
        pb, qb, rb = x[4], x[5], x[6]

        # Normalize quaternions
        norm = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
        e0, e1, e2, e3 = e0 / norm, e1 / norm, e2 / norm, e3 / norm
        x[0:4] = e0, e1, e2, e3

        # Euler angles
        r = np.arctan2(2 * (e0 * e1 + e3 * e2), 1 - 2 * (e1**2 + e2**2)) 
        p = np.arcsin(2 * (e0 * e2 - e3 * e1)) 
        y = np.arctan2(2 * (e0 * e3 + e1 * e2), 1 - 2 * (e2**2 + e3**2)) 

        gtq = opt_quat[i, :]
        gtq0, gtq1, gtq2, gtq3 = gtq[0], gtq[1], gtq[2], gtq[3]
        gt_r = np.arctan2(2 * (gtq0 * gtq1 + gtq3 * gtq2), 1 - 2 * (gtq1**2 + gtq2**2)) 
        gt_p = np.arcsin(2 * (gtq0 * gtq2 - gtq3 * gtq1))
        gt_y = np.arctan2(2 * (gtq0 * gtq3 + gtq1 * gtq2), 1 - 2 * (gtq2**2 + gtq3**2)) 

        # Euler angles
        roll[i] = r * 180 / np.pi
        pitch[i] = p * 180 / np.pi
        yaw[i] = y * 180 / np.pi

        gt_roll[i] = gt_r * 180 / np.pi
        gt_pitch[i] = gt_p * 180 / np.pi
        gt_yaw[i] = gt_y * 180 / np.pi

        # 将欧拉角转换为旋转矩阵
        R1 = euler_to_rotation_matrix(r, p, y)
        R2 = euler_to_rotation_matrix(gt_r, gt_p, gt_y)
        
        # 计算相对旋转矩阵 R_rel
        R_rel = np.dot(R2, R1.T)  # R1.T 是 R1 的转置，相当于 R1 的逆矩阵
        
        # 从旋转矩阵中提取旋转角度
        angle_error = rotation_matrix_to_angle(R_rel)
        
        # 将弧度转换为度数
        angle_errors[i] = np.degrees(angle_error)

    # Optionally, plot the results
    import matplotlib.pyplot as plt

    # 绘制欧拉角对比图
    plt.figure(figsize=(10, 8))

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(roll, label='Estimated Roll')
    plt.plot(gt_roll, label='Ground Truth Roll')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(pitch, label='Estimated Pitch')
    plt.plot(gt_pitch, label='Ground Truth Pitch')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(yaw, label='Estimated Yaw')
    plt.plot(gt_yaw, label='Ground Truth Yaw')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出角度差的均值
    mean_error = np.nanmean(angle_errors)  # 使用 np.nanmean 来忽略 NaN 值
    print(f"Mean Rotation Angle Error: {mean_error:.2f} degrees")

    # 绘制角度差的折线图
    plt.plot(range(len(angle_errors)), angle_errors)
    plt.xlabel('Sample Index')
    plt.ylabel('Rotation Angle Error (degrees)')
    plt.title('Rotation Angle Errors for Each Sample')
    plt.show()

EKF(path)