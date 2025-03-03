"""# **Class: System Model for Non-linear Cases**

1 Store system model parameters: 
    state transition function f, 
    observation function h, 
    process noise Q, 
    observation noise R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test,
    state dimension m,
    observation dimension n, etc.

2 Generate datasets for non-linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:
    # 定义系统模型类，包含运动模型、观测模型、序列生成、批量生成等功能
    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f  # 状态转移函数，定义系统的动态演化
        self.m = m  # 状态变量维度
        self.Q = Q  # 过程噪声协方差矩阵
        #########################
        ### Observation Model ###
        #########################
        self.h = h  # 观测函数，定义状态到观测的映射
        self.n = n  # 观测变量维度
        self.R = R  # 观测噪声协方差矩阵
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m) # 过程噪声协方差的先验（默认单位矩阵）
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m)) # 状态协方差先验（默认全零矩阵）
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n*2) # 自定义先验（观测相关）
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0  # 初始状态均值
        self.m2x_0 = m2x_0  # 初始状态协方差

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

