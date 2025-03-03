import math
from scipy.io import loadmat
import numpy as np

def calculateErrorQuatEarth(imu_quat, opt_quat):
    """
    Calculates quaternion that represents the orientation estimation error in the global coordinate system.

    :param imu_quat: IMU orientation, shape (N, 4)
    :param opt_quat: OMC orientation, shape (N, 4)
    :return: error quaternion, shape (N, 4)
    """
    # normalize the input quaternions just in case
    imu_quat = imu_quat / np.linalg.norm(imu_quat, axis=1)[:, None]
    opt_quat = opt_quat / np.linalg.norm(opt_quat, axis=1)[:, None]
    # calculate the relative orientation expressed in the global coordinate system
    # imu_quat * (inv(opt_quat) * imu_quat) * inv(imu_quat) = imu_quat * inv(opt_quat)
    out = quatmult(imu_quat, invquat(opt_quat))
    # normalize the output quaternion
    out = out / np.linalg.norm(out, axis=1)[:, None]
    return out


def calculateTotalError(q_diff):
    """
    Calculates the total error, i.e. the total absolute rotation angle of the quaternion.

    :param q_diff: error quaternion, shape (N, 4)
    :return: error in rad, shape (N,)
    """
    return 2 * np.arccos(np.clip(np.abs(q_diff[:, 0]), 0, 1))


def calculateHeadingError(q_diff_earth):
    """
    Calculates the heading error.

    :param q_diff_earth: error quaternion in global coordinates (c.f. calculateErrorQuatEarth), shape (N, 4)
    :return: error in rad, shape (N,)
    """
    return 2 * np.arctan(np.abs(q_diff_earth[:, 3] / q_diff_earth[:, 0]))


def calculateInclinationError(q_diff_earth):
    """
    Calculates the inclination error.

    :param q_diff_earth: error quaternion in global coordinates (c.f. calculateErrorQuatEarth), shape (N, 4)
    :return: error in rad, shape (N,)
    """
    return 2 * np.arccos(np.clip(np.sqrt(q_diff_earth[:, 0] ** 2 + q_diff_earth[:, 3] ** 2), 0, 1))


def calculateRMSE(imu_quat, opt_quat, movement):
    """
    Calculates total/heading/inclination errors in degrees (only considering movement phases).

    :param imu_quat: IMU orientation, shape (N, 4)
    :param opt_quat: OMC orientation, shape (N, 4)
    :param movement: boolean indexing array that denotes motion phases, shape (N,)
    :return: dict containing total, heading and inclination errors in degrees
    """
    assert movement.dtype == bool

    q_diff_earth = calculateErrorQuatEarth(imu_quat, opt_quat)

    totalError = calculateTotalError(q_diff_earth)[movement]
    headingError = calculateHeadingError(q_diff_earth)[movement]
    inclError = calculateInclinationError(q_diff_earth)[movement]

    return dict(
        total_rmse_deg=np.rad2deg(rmse(totalError)),
        heading_rmse_deg=np.rad2deg(rmse(headingError)),
        inclination_rmse_deg=np.rad2deg(rmse(inclError))
    )


def rmse(diff):
    """Calculates the RMS of the input signal."""
    return np.sqrt(np.nanmean(diff**2))


def quatmult(q1, q2):
    """
    Quaternion multiplication.

    If two Nx4 arrays are given, they are multiplied row-wise. Alternative one of the inputs can be a single
    quaternion which is then multiplied to all rows of the other input array.
    """

    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)

    # if both input quaternions are 1D arrays, we also want to return a 1D output
    is1D = max(len(q1.shape), len(q2.shape)) < 2

    # but to be able to use the same indexing in all cases, make sure everything is in 2D arrays
    if q1.shape == (4,):
        q1 = q1.reshape((1, 4))
    if q2.shape == (4,):
        q2 = q2.reshape((1, 4))

    # check the dimensions
    N = max(q1.shape[0], q2.shape[0])
    assert q1.shape == (N, 4) or q1.shape == (1, 4)
    assert q2.shape == (N, 4) or q2.shape == (1, 4)

    # actual quaternion multiplication
    q3 = np.zeros((N, 4), np.float)
    q3[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q3[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q3[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q3[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    if is1D:
        q3 = q3.reshape((4,))

    return q3


def invquat(q):
    """Calculates the inverse of unit quaternions."""

    q = np.asarray(q, np.float)
    if len(q.shape) != 2:
        assert q.shape == (4,)
        qConj = q.copy()
        qConj[1:] *= -1
        return qConj
    else:
        assert q.shape[1] == 4
        qConj = q.copy()
        qConj[:, 1:] *= -1
        return qConj


class MahonyAHRS:
    def __init__(self, Kp=0.5, Ki=0.0, sampleFreq=512.0):
        self.twoKp = 2 * Kp  # 2 * proportional gain
        self.twoKi = 2 * Ki  # 2 * integral gain
        self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0  # quaternion
        self.integralFBx, self.integralFBy, self.integralFBz = 0.0, 0.0, 0.0  # integral error terms
        self.sampleFreq = sampleFreq  # sample frequency in Hz

    def update(self, gx, gy, gz, ax, ay, az, mx, my, mz):
        if mx == my == mz == 0.0:
            self.updateIMU(gx, gy, gz, ax, ay, az)
            return

        if not (ax == ay == az == 0.0):
            ax /= math.sqrt(ax * ax + ay * ay + az * az)  # normalize accelerometer
            mx /= math.sqrt(mx * mx + my * my + mz * mz)  # normalize magnetometer

            q0q0, q0q1, q0q2, q0q3 = self.q0 * self.q0, self.q0 * self.q1, self.q0 * self.q2, self.q0 * self.q3
            q1q1, q1q2, q1q3 = self.q1 * self.q1, self.q1 * self.q2, self.q1 * self.q3
            q2q2, q2q3, q3q3 = self.q2 * self.q2, self.q2 * self.q3, self.q3 * self.q3

            hx = 2 * (mx * (0.5 - q2q2 - q3q3) + my * (q1q2 - q0q3) + mz * (q1q3 + q0q2))
            hy = 2 * (mx * (q1q2 + q0q3) + my * (0.5 - q1q1 - q3q3) + mz * (q2q3 - q0q1))
            bx = math.sqrt(hx * hx + hy * hy)
            bz = 2 * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1) + mz * (0.5 - q1q1 - q2q2))

            halfvx = self.q1 * self.q3 - self.q0 * self.q2
            halfvy = self.q0 * self.q1 + self.q2 * self.q3
            halfvz = q0q0 - 0.5 + q3q3
            halfwx = bx * (0.5 - q2q2 - q3q3) + bz * (self.q1 * self.q3 - self.q0 * self.q2)
            halfwy = bx * (self.q1 * self.q2 - self.q0 * self.q3) + bz * (self.q0 * self.q1 + self.q2 * self.q3)
            halfwz = bx * (self.q0 * self.q2 + self.q1 * self.q3) + bz * (0.5 - q1q1 - q2q2)

            halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy)
            halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz)
            halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx)

            if self.twoKi > 0.0:
                self.integralFBx += self.twoKi * halfex / self.sampleFreq
                self.integralFBy += self.twoKi * halfey / self.sampleFreq
                self.integralFBz += self.twoKi * halfez / self.sampleFreq
                gx += self.integralFBx
                gy += self.integralFBy
                gz += self.integralFBz
            else:
                self.integralFBx = self.integralFBy = self.integralFBz = 0.0

            gx += self.twoKp * halfex
            gy += self.twoKp * halfey
            gz += self.twoKp * halfez

        self._integrate_rate_of_change(gx, gy, gz)

    def updateIMU(self, gx, gy, gz, ax, ay, az):
        if not (ax == ay == az == 0.0):
            ax /= math.sqrt(ax * ax + ay * ay + az * az)  # normalize accelerometer

            halfvx = self.q1 * self.q3 - self.q0 * self.q2
            halfvy = self.q0 * self.q1 + self.q2 * self.q3
            halfvz = self.q0 * self.q0 - 0.5 + self.q3 * self.q3

            halfex = (ay * halfvz - az * halfvy)
            halfey = (az * halfvx - ax * halfvz)
            halfez = (ax * halfvy - ay * halfvx)

            if self.twoKi > 0.0:
                self.integralFBx += self.twoKi * halfex / self.sampleFreq
                self.integralFBy += self.twoKi * halfey / self.sampleFreq
                self.integralFBz += self.twoKi * halfez / self.sampleFreq
                gx += self.integralFBx
                gy += self.integralFBy
                gz += self.integralFBz
            else:
                self.integralFBx = self.integralFBy = self.integralFBz = 0.0

            gx += self.twoKp * halfex
            gy += self.twoKp * halfey
            gz += self.twoKp * halfez

        self._integrate_rate_of_change(gx, gy, gz)

    def _integrate_rate_of_change(self, gx, gy, gz):
        gx *= 0.5 / self.sampleFreq
        gy *= 0.5 / self.sampleFreq
        gz *= 0.5 / self.sampleFreq
        qa, qb, qc = self.q0, self.q1, self.q2
        self.q0 += (-qb * gx - qc * gy - self.q3 * gz)
        self.q1 += (qa * gx + qc * gz - self.q3 * gy)
        self.q2 += (qa * gy - qb * gz + self.q3 * gx)
        self.q3 += (qa * gz + qb * gy - qc * gx)

        norm = math.sqrt(self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)
        self.q0 /= norm
        self.q1 /= norm
        self.q2 /= norm
        self.q3 /= norm

        quaternion = (self.q0 / norm, self.q1 / norm, self.q2 / norm, self.q3 / norm)
        return quaternion
    

# 加载数据
path = 'Simulations/data_mat/02_undisturbed_slow_rotation_B.mat'
data = loadmat(path)

# 提取数据
imu_acc = data['imu_acc']  # 加速度计数据 (ax, ay, az)
imu_gyr = data['imu_gyr']  # 陀螺仪数据 (gx, gy, gz)
imu_mag = data['imu_mag']  # 磁力计数据 (mx, my, mz)
opt_quat = data['opt_quat']  # 真实四元数 (q0, q1, q2, q3)
sampling_rate =  2000  # 采样频率

# 创建 MahonyAHRS 实例
ahrs = MahonyAHRS(Kp=0.5, Ki=0.0, sampleFreq=7.0 / sampling_rate)

# 初始化存储结果的列表
estimated_quat = []

# 遍历数据并调用 update 方法
for i in range(len(imu_acc)):
    gx, gy, gz = imu_gyr[i]
    ax, ay, az = imu_acc[i]
    mx, my, mz = imu_mag[i]
    
    # 调用 update 方法
    quat = ahrs.update(gx, gy, gz, ax, ay, az, mx, my, mz)
    
    # 将估计的四元数存储到列表中
    estimated_quat.append(quat)

data['movement'] = data['movement'].squeeze().astype(bool)

# 将列表转换为 NumPy 数组
estimated_quat = np.array(estimated_quat)

total,heading,inclina = calculateRMSE(estimated_quat,opt_quat, data['movement'])

# 打印结果
print("Estimated Quaternion Shape:", estimated_quat.shape)
print("Optimal Quaternion Shape:", opt_quat.shape)

# 可选：计算估计四元数与真实四元数之间的误差
#error = np.linalg.norm(estimated_quat - opt_quat, axis=1)
print("Average Quaternion Error:", total,heading,inclina)