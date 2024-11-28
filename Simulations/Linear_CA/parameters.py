"""
This file contains the parameters for the simulations with linear kinematic model
* Constant Acceleration Model (CA)
    # full state P, V, A
    # only postion P
* Constant Velocity Model (CV)
"""

import torch

m = 3 # dim of state for CA model
m_cv = 2 # dim of state for CV model

delta_t_gen =  1e-2

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
#状态演化矩阵，用于描述一个匀加速模型（CA, Constant Acceleration Model）。
#该模型假设物体以恒定加速度运动，系统的状态包括位置、速度和加速度。维度为 [3X3]
#表示状态的三维结构（位置、速度、加速度）
F_gen = torch.tensor([[1, delta_t_gen,0.5*delta_t_gen**2],
                  [0,       1,       delta_t_gen],
                  [0,       0,         1]]).float()
#状态演化矩阵，用于描述匀速运动模型（CV, Constant Velocity Model）
#系统状态仅包括位置和速度。
F_CV = torch.tensor([[1, delta_t_gen],
                     [0,           1]]).float()              

# Full observation
# 系统的所有状态（位置、速度、加速度）
H_identity = torch.eye(3)
# Observe only the postion
#只观测到系统的位置信息
H_onlyPos = torch.tensor([[1, 0, 0]]).float()

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise Parameters
r2 = torch.tensor([1]).float()
q2 = torch.tensor([1]).float()

# 匀加速模型下的过程噪声协方差矩阵，过程噪声反映了系统在状态转移过程中未建模的随机扰动
# q2 是过程噪声的强度，它控制了系统在状态转移过程中的噪声水平
Q_gen = q2 * torch.tensor([[1/20*delta_t_gen**5, 1/8*delta_t_gen**4,1/6*delta_t_gen**3],
                           [ 1/8*delta_t_gen**4, 1/3*delta_t_gen**3,1/2*delta_t_gen**2],
                           [ 1/6*delta_t_gen**3, 1/2*delta_t_gen**2,       delta_t_gen]]).float()

Q_CV = q2 * torch.tensor([[1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
                          [1/2*delta_t_gen**2,        delta_t_gen]]).float()  

R_3 = r2 * torch.eye(3)
R_2 = r2 * torch.eye(2)

# 仅观测位置的噪声协方差，标量形式，表示观测位置时的噪声强度。
R_onlyPos = r2

