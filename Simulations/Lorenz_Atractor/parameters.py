"""This file contains the parameters for the Lorenz Atractor simulation.

Update 2023-02-06: f and h support batch size speed up

"""

import torch
import math
import os
import scipy.io
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 4
n = 3
variance = 0
#m1x_0 = torch.ones(m, 1) # 创建大小为（m,1）的张量，其中所有元素为1
batch_size = 1000
#m1x_0 = torch.full((batch_size, 4), torch.tensor([1.0, 0.0, 0.0, 0.0]), dtype=torch.double)
m1x_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
# m2x_0 = 0 * 0 * torch.eye(m) # 创造大小为（m,m）的全零矩阵
m2x_0 = torch.eye(m) # 创造大小为（m,m）的全零矩阵

### Taylor expansion order
J = 5
J_mod = 2

### Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)

### input_nan fix
def slerp(q0, q1, taus):
    """批量进行四元数插值操作"""
    dot = (q0 * q1).sum(dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)  # 使用 torch.clamp 代替手动限制

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta = theta_0.unsqueeze(-1) * taus
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta_0.unsqueeze(-1) - theta) / (sin_theta_0.unsqueeze(-1) + 1e-7)
    s1 = sin_theta / (sin_theta_0.unsqueeze(-1) + 1e-7)

    q_interp = s0 * q0.unsqueeze(1) + s1 * q1.unsqueeze(1)
    q_interp = q_interp / q_interp.norm(dim=-1, keepdim=True)  # 单位化

    return q_interp

def interpolate_quaternions(input_quat):
    """快速插值四元数，使用批量处理代替循环"""
    nan_mask = torch.isnan(input_quat[:, 0])
    valid_indices = torch.nonzero(~nan_mask, as_tuple=False).squeeze()
    
    if len(valid_indices) < 2:
        raise ValueError("Not enough valid quaternions to interpolate.")
    
    output_quat = input_quat.clone()
    
    # 计算所有插值间隔
    start_indices = valid_indices[:-1]
    end_indices = valid_indices[1:]

    for start_idx, end_idx in zip(start_indices, end_indices):
        if start_idx >= end_idx - 1:
            continue

        q0 = input_quat[start_idx]
        q1 = input_quat[end_idx]

        # 批量生成 tau (0 到 1)
        nan_indices = torch.arange(start_idx + 1, end_idx)
        taus = (nan_indices - start_idx).float() / (end_idx - start_idx)
        
        # 批量插值四元数
        interpolated_quats = slerp(q0.unsqueeze(0), q1.unsqueeze(0), taus.unsqueeze(1))
        output_quat[nan_indices] = interpolated_quats.squeeze(1)

    return output_quat

def clean_fix_data(input_acc, input_gyr, input_mag, input_quat_nan):
    """清理并修复加速度计、陀螺仪、磁力计和四元数数据"""
    quat_nan_mask = torch.isnan(input_quat_nan[:, 0])

    first_valid_idx = torch.nonzero(~quat_nan_mask, as_tuple=False)[0].item()
    last_valid_idx = torch.nonzero(~quat_nan_mask, as_tuple=False)[-1].item()

    cleaned_quat = input_quat_nan[first_valid_idx:last_valid_idx + 1]
    fixed_quat = interpolate_quaternions(cleaned_quat)
    cleaned_acc = input_acc[first_valid_idx:last_valid_idx + 1]
    cleaned_gyr = input_gyr[first_valid_idx:last_valid_idx + 1]
    cleaned_mag = input_mag[first_valid_idx:last_valid_idx + 1]

    if torch.any(torch.isnan(fixed_quat)):
        print("opt_quat 中仍存在 NaN 值")
    # else:
    #     print("opt_quat 中已无 NaN 值")

    return cleaned_acc, cleaned_gyr, cleaned_mag, fixed_quat


def load_data(path_data: str, validation_files: list, test_files: list, batch_size: int = 1000):
    def process_files(file_list, path_data, batch_size):
        """辅助函数，用于处理指定文件列表，将数据切分成批次"""
        imu_acc_batches_list = []
        imu_gyr_batches_list = []
        imu_mag_batches_list = []
        opt_quat_batches_list = []
        # ToDo 1 ：修改
        # batch_sizes = []
        batch_nums = []

        for file_name in file_list:
            file_path = os.path.join(path_data, file_name)
            if not file_name.endswith('.mat'):
                continue

            mat_data = scipy.io.loadmat(file_path)
            print(f"Processing {file_name}")

            raw_imu_acc = torch.tensor(mat_data['imu_acc'], dtype=torch.float32)
            raw_imu_gyr = torch.tensor(mat_data['imu_gyr'], dtype=torch.float32)
            raw_imu_mag = torch.tensor(mat_data['imu_mag'], dtype=torch.float32)
            raw_opt_quat = torch.tensor(mat_data['opt_quat'], dtype=torch.float32)  # 默认改为 float32


            imu_acc, imu_gyr, imu_mag, opt_quat = clean_fix_data(
                raw_imu_acc, raw_imu_gyr, raw_imu_mag, raw_opt_quat
            )

            if imu_acc is None or imu_gyr is None or imu_mag is None or opt_quat is None:
                print(f"Skipping {file_name} due to missing data")
                continue

            N = imu_acc.shape[0]
            if N != imu_gyr.shape[0] or N != imu_mag.shape[0] or N != opt_quat.shape[0]:
                print(f"Skipping {file_name} due to inconsistent data lengths")
                continue

            num_batches = N // batch_size
            batch_nums.append(num_batches)

            imu_acc_batches = imu_acc[:num_batches * batch_size].reshape(num_batches, batch_size, 3)
            imu_gyr_batches = imu_gyr[:num_batches * batch_size].reshape(num_batches, batch_size, 3)
            imu_mag_batches = imu_mag[:num_batches * batch_size].reshape(num_batches, batch_size, 3)
            opt_quat_batches = opt_quat[:num_batches * batch_size].reshape(num_batches, batch_size, 4)

            imu_acc_batches_list.append(torch.tensor(imu_acc_batches))
            imu_gyr_batches_list.append(torch.tensor(imu_gyr_batches))
            imu_mag_batches_list.append(torch.tensor(imu_mag_batches))
            opt_quat_batches_list.append(torch.tensor(opt_quat_batches))


        if imu_acc_batches_list:
            imu_acc_batches = torch.cat(imu_acc_batches_list, dim=0)
            imu_gyr_batches = torch.cat(imu_gyr_batches_list, dim=0)
            imu_mag_batches = torch.cat(imu_mag_batches_list, dim=0)
            opt_quat_batches = torch.cat(opt_quat_batches_list, dim=0)
        else:
            imu_acc_batches = torch.tensor([])
            imu_gyr_batches = torch.tensor([])
            imu_mag_batches = torch.tensor([])
            opt_quat_batches = torch.tensor([])

        return imu_acc_batches, imu_gyr_batches, imu_mag_batches, opt_quat_batches, batch_nums

    # 获取所有训练文件（排除验证集和测试集文件）
    all_files = sorted(os.listdir(path_data))
    train_files = [f for f in all_files if f not in validation_files and f not in test_files]

    # 处理训练集
    train_imu_acc_batches, train_imu_gyr_batches, train_imu_mag_batches, train_opt_quat_batches, train_batch_nums = process_files(train_files, path_data, batch_size)
    print(f"Training set processed: {train_imu_acc_batches.shape}, {train_imu_gyr_batches.shape},{train_opt_quat_batches.shape},{train_batch_nums}")

    # 处理验证集
    val_imu_acc_batches, val_imu_gyr_batches, val_imu_mag_batches, val_opt_quat_batches, val_batch_nums = process_files(validation_files, path_data, batch_size)
    print(f"Validation set processed: {val_imu_acc_batches.shape}, {val_imu_gyr_batches.shape},{val_opt_quat_batches.shape},{val_batch_nums}")

    # 处理测试集
    test_imu_acc_batches, test_imu_gyr_batches, test_imu_mag_batches, test_opt_quat_batches, test_batch_nums = process_files(test_files, path_data, batch_size)
    print(f"Test set processed: {test_imu_acc_batches.shape}, {test_imu_gyr_batches.shape},{test_opt_quat_batches.shape},{test_batch_nums}")

    return (
        (train_imu_acc_batches, train_imu_gyr_batches, train_imu_mag_batches, train_opt_quat_batches, train_batch_nums),
        (val_imu_acc_batches, val_imu_gyr_batches, val_imu_mag_batches, val_opt_quat_batches, val_batch_nums),
        (test_imu_acc_batches, test_imu_gyr_batches, test_imu_mag_batches, test_opt_quat_batches, test_batch_nums)
    )


######################################################
### State evolution function f for Lorenz Atractor ###
######################################################

def normalize_quaternion(q):
    """
    对形状为 (batch_size, 4, 1) 的四元数张量进行归一化。

    参数:
    q: 形状为 (batch_size, 4, 1) 的四元数张量

    返回:
    归一化后的四元数张量，形状与输入相同
    """
    # 计算每个四元数的模长
    norm = torch.sqrt(torch.sum(q**2, dim=1, keepdim=True))  # 在四元数的维度(第二维度)上进行求和
    
    # 对四元数进行归一化，并返回结果
    return q / (norm + 1e-7)  # 添加一个小值以防止除以零的情况

def f(pre_v, gyro, dt):
    """
        利用当前四元数和陀螺仪数据更新下一时刻的四元数

        参数:
        pre_v (torch.Tensor): 当前时刻的四元数，形状为 [batch_size, 4]
        gyro (torch.Tensor): 陀螺仪的角速度矢量，形状为 [batch_size, 3]
        dt (float): 时间步长

        返回:
        p_v (torch.Tensor): 更新后的四元数，形状为 [batch_size, 4]
    """
    # 将四元数和陀螺仪数据扩展第三维以便后续计算
    pre_v = pre_v.unsqueeze(2)  # [batch_size, 4, 1]
    gyro = gyro.unsqueeze(2)  # [batch_size, 3, 1]

    # 拆分四元数分量
    e0 = pre_v[:, 0, 0]  # [batch_size]
    e1 = pre_v[:, 1, 0]
    e2 = pre_v[:, 2, 0]
    e3 = pre_v[:, 3, 0]

    # 构建四元数旋转矩阵 O_nb，形状为 [batch_size, 4, 4]
    O_nb = torch.stack([
        torch.stack([-e1, -e2, -e3], dim=-1),  # 第一行
        torch.stack([e0, -e3, e2], dim=-1),  # 第二行
        torch.stack([e3, e0, -e1], dim=-1),  # 第三行
        torch.stack([-e2, e1, e0], dim=-1)  # 第四行
    ], dim=1)

    # 更新四元数公式 p_v = pre_v + 0.5 * dt * O_nb * gyro
    p_v = pre_v + dt * 0.5 * torch.bmm(O_nb, gyro)  # [batch_size, 4, 1]

    # 归一化四元数，确保其模为1
    p_v = normalize_quaternion(p_v).squeeze(-1)  # [batch_size, 4]

    return p_v


###############################################
### Observation function h for magnetometer ###
###############################################
H_design = torch.eye(n)
H_Rotate = torch.mm(RotMatrix,H_design)
H_Rotate_inv = torch.inverse(H_Rotate)


def h(q_t, dm):
    """
    根据当前四元数预测磁力计和加速度计的输出

    参数:
    q_t (torch.Tensor): 当前时刻四元数，形状为 [batch_size, 4]
    dm (float): 地磁场的磁偏角（单位：弧度）

    返回:
    z (torch.Tensor): 磁力计与加速度计的预测值，形状为 [batch_size, 6]
    """
    # 确保输入的四元数形状为 [batch_size, 4]
    assert q_t.shape[1] == 4, "四元数的形状应为 (batch_size, 4)"

    # 提取四元数分量
    e0, e1, e2, e3 = q_t[:, 0], q_t[:, 1], q_t[:, 2], q_t[:, 3]

    # 计算磁偏角的正弦和余弦值
    msin = torch.sin(torch.tensor(dm))  # 磁偏角的正弦
    mcos = torch.cos(torch.tensor(dm))  # 磁偏角的余弦

    # 磁力计预测模型
    m_x = msin * (2 * e0 * e3 + 2 * e1 * e2) - mcos * (2 * e2**2 + 2 * e3**2 - 1)
    m_y = -mcos * (2 * e0 * e3 - 2 * e1 * e2) - msin * (2 * e1**2 + 2 * e3**2 - 1)
    m_z = mcos * (2 * e0 * e2 + 2 * e1 * e3) - msin * (2 * e0 * e1 - 2 * e2 * e3)
    m = torch.stack([m_x, m_y, m_z], dim=1)  # 合并磁力计预测值

    # 加速度计预测模型
    a_x = -2 * (e1 * e3 - e0 * e2)
    a_y = 2 * (e0 * e1 + e2 * e3)
    a_z = 1 - 2 * (e1**2 + e2**2)
    a = torch.stack([a_x, a_y, a_z], dim=1)  # 合并加速度计预测值

    # 合并磁力计和加速度计预测值
    z = torch.cat([a, m], dim=1)  # [batch_size, 6]

    return z
    

def hRotate(x, jacobian=False):
    """
    线性观测模型，应用单位矩阵变换

    参数:
    x (torch.Tensor): 输入数据，形状为 [batch_size, n, 1]
    jacobian (bool): 是否返回雅可比矩阵

    返回:
    y (torch.Tensor): 观测值，形状为 [batch_size, n, 1]
    """
    H = H_design.to(x.device).reshape((1, n, n)).repeat(x.shape[0], 1, 1)  # 单位矩阵
    y = torch.bmm(H, x)  # 矩阵乘法计算输出
    if jacobian:
        return y, H
    else:
        return y

def h_nonlinear(x):
    return toSpherical(x)


def h_nobatch(x, jacobian=False):
    H = H_design.to(x.device)
    y = torch.matmul(H,x)
    if jacobian:
        return y, H
    else:
        return y
###############################################
### process noise Q and observation noise R ###
###############################################
Q_non_diag = False
R_non_diag = False

Q_structure = torch.eye(m)
R_structure = torch.eye(n)

if(Q_non_diag):
    q_d = 1
    q_nd = 1/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

if(R_non_diag):
    r_d = 1
    r_nd = 1/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])

##################################
### Utils for non-linear cases ###
##################################
def getJacobian(x, g):
    """
    Currently, pytorch does not have a built-in function to compute Jacobian matrix
    in a batched manner, so we have to iterate over the batch dimension.

    input x (torch.tensor): [batch_size, m/n, 1]
    input g (function): function to be differentiated
    output Jac (torch.tensor): [batch_size, m, m] for f, [batch_size, n, m] for h
    """
    # Method 1: using autograd.functional.jacobian
    # batch_size = x.shape[0]
    # Jac_x0 = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[0,:,:],0)))
    # Jac = torch.zeros([batch_size, Jac_x0.shape[0], Jac_x0.shape[1]])
    # Jac[0,:,:] = Jac_x0
    # for i in range(1,batch_size):
    #     Jac[i,:,:] = torch.squeeze(autograd.functional.jacobian(g, torch.unsqueeze(x[i,:,:],0)))
    # Method 2: using F, H directly
    _,Jac = g(x, jacobian=True)
    return Jac

def toSpherical(cart):
    """
    将笛卡尔坐标转换为球坐标

    参数:
    cart (torch.Tensor): 笛卡尔坐标，形状为 [batch_size, m, 1] 或 [batch_size, m]

    返回:
    spher (torch.Tensor): 球坐标，形状为 [batch_size, n, 1]
    """
    rho = torch.linalg.norm(cart, dim=1).reshape(cart.shape[0], 1)  # 距离
    phi = torch.atan2(cart[:, 1, ...], cart[:, 0, ...]).reshape(cart.shape[0], 1)  # 方位角
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)  # 将角度调整为 [0, 2π]

    theta = torch.div(cart[:, 2, ...], rho.squeeze()).reshape(cart.shape[0], 1)  # 仰角
    theta = torch.acos(theta).reshape(cart.shape[0], 1)

    spher = torch.cat([rho, theta, phi], dim=1).reshape(cart.shape[0], 3, 1)  # 合并结果
    return spher

def toCartesian(sphe, cart):
    """
    input sphe (torch.tensor): [batch_size, n, 1] or [batch_size, n]
    output cart (torch.tensor): [batch_size, n]
    """
    rho = sphe[:, 0, ...]
    theta = sphe[:, 1, ...]
    phi = sphe[:, 2, ...]

    x = (rho * torch.sin(theta) * torch.cos(phi)).reshape(sphe.shape[0],1)
    y = (rho * torch.sin(theta) * torch.sin(phi)).reshape(sphe.shape[0],1)
    z = (rho * torch.cos(theta)).reshape(sphe.shape[0],1)

    cart = torch.cat([x,y,z],dim=1).reshape(cart.shape[0],3,1) # [batch_size, n, 1]

    return cart