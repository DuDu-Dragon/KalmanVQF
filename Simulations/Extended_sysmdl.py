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

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # 根据给定的噪声参数和序列长度生成状态和观测序列
        self.x = torch.zeros(size=[self.m, T])  # 初始化状态序列数组
        self.y = torch.zeros(size=[self.n, T])  # 初始化观测序列数组
        self.x_prev = self.m1x_0  # 将初始状态设为上一时刻的状态
        xt = self.x_prev  # 当前状态初始化

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################   
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):# No noise
                 xt = self.f(self.x_prev)   
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen) # 生成一维正态噪声
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:  # 多维过程噪声
                xt = self.f(self.x_prev)  # 状态演化
                mean = torch.zeros([self.m])  # 噪声均值
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)  # 多维正态分布
                eq = distrib.rsample()  # 采样噪声
                eq = torch.reshape(eq[:], xt.size())  # 调整形状
                xt = torch.add(xt, eq)  # 加入噪声

            ################
            ### Emission ###
            ################
            yt = self.h(xt)  # 根据观测函数生成观测值
            if self.n == 1:  # 一维观测噪声
                er = torch.normal(mean=0, std=R_gen)  # 生成一维正态噪声
                yt = torch.add(yt, er)  # 加入观测噪声
            else:  # 多维观测噪声
                mean = torch.zeros([self.n])  # 噪声均值
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)  # 多维正态分布
                er = distrib.rsample()  # 采样噪声
                er = torch.reshape(er[:], yt.size())  # 调整形状
                yt = torch.add(yt, er)  # 加入噪声
            
            ########################
            ### Squeeze to Array ###
            ########################

            self.x[:, t] = torch.squeeze(xt, 1)  # 保存当前状态
            self.y[:, t] = torch.squeeze(yt, 1)  # 保存当前观测
            self.x_prev = xt  # 更新上一时刻的状态为当前状态


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if args.distribution == 'uniform':
                ### if Uniform Distribution for random init
                # 均匀分布随机初始化
                for i in range(size):
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)     
            
            elif args.distribution == 'normal':
                ### if Normal Distribution for random init
                # 正态分布随机初始化
                for i in range(size):
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m,1)
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')
            
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)### for sequence generation
        else: # fixed init
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
            self.Init_batched_sequence(initConditions, self.m2x_0)### for sequence generation
    
        if(args.randomLength):
            # Allocate Array for Input and Target (use zero padding)
            self.Input = torch.zeros(size, self.n, args.T_max)
            self.Target = torch.zeros(size, self.m, args.T_max)
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)# init with all false
            # Init Sequence Lengths
            T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min # Uniform distribution [100,1000]
            for i in range(0, size):
                # Generate Sequence
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input[i, :, 0:T_tensor[i].item()] = self.y             
                # Training sequence output
                self.Target[i, :, 0:T_tensor[i].item()] = self.x
                # Mask for sequence length
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)

            # Set x0 to be x previous
            self.x_prev = self.m1x_0_batch
            xt = self.x_prev

            # Generate in a batched manner
            for t in range(0, T):
                ########################
                #### State Evolution ###
                ########################   
                if torch.equal(self.Q,torch.zeros(self.m,self.m)):# No noise
                    xt = self.f(self.x_prev)
                elif self.m == 1: # 1 dim noise
                    xt = self.f(self.x_prev)
                    eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)
                else:            
                    xt = self.f(self.x_prev)
                    mean = torch.zeros([size, self.m])              
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                    eq = distrib.rsample().view(size,self.m,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)

                ################
                ### Emission ###
                ################
                # Observation Noise
                if torch.equal(self.R,torch.zeros(self.n,self.n)):# No noise
                    yt = self.h(xt)
                elif self.n == 1: # 1 dim noise
                    yt = self.h(xt)
                    er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    # Additive Observation Noise
                    yt = torch.add(yt,er)
                else:  
                    yt =  self.h(xt)
                    mean = torch.zeros([size,self.n])            
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                    er = distrib.rsample().view(size,self.n,1)          
                    # Additive Observation Noise
                    yt = torch.add(yt,er)

                ########################
                ### Squeeze to Array ###
                ########################

                # Save Current State to Trajectory Array
                self.Target[:, :, t] = torch.squeeze(xt,2)

                # Save Current Observation to Trajectory Array
                self.Input[:, :, t] = torch.squeeze(yt,2)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt
