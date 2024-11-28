"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func


# def quaternion_normalize(q):
#     """
#     对四元数进行单位化，若模长为零则返回 [1, 0, 0, 0]。
#     """
#     norm = torch.norm(q, dim=1, keepdim=True)  # 计算四元数的模长
    
#     # 若 norm 为 0，返回 [1, 0, 0, 0]，否则进行单位化
#     return torch.where(norm == 0, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=q.device), q / norm)

def quaternion_normalize(q, epsilon=1e-7):
    """
    对四元数进行单位化，若模长接近零则返回 [1, 0, 0, 0]。
    
    参数:
        q (torch.Tensor): 输入的四元数张量，形状为 (batch_size, 4)。
        epsilon (float): 防止除以 0 的偏移量，默认为 1e-7。
    返回:
        torch.Tensor: 单位化后的四元数，形状为 (batch_size, 4)。
    """
    # 计算四元数的模长，并添加偏移量
    norm = torch.norm(q, dim=1, keepdim=True) + epsilon
    
    # 对四元数进行单位化，模长接近零时返回 [1, 0, 0, 0]
    normalized_q = q / norm
    fallback = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=q.device)
    return torch.where(norm <= epsilon, fallback, normalized_q)

def quaternion_difference(quaternion1, quaternion2):
    """
    计算两个四元数之间的相对四元数[batch_size, 4]
    """

    quaternion1 = quaternion_normalize(quaternion1)
    quaternion2 = quaternion_normalize(quaternion2)
    
    # 计算四元数的共轭
    quaternion1_conjugate = torch.cat((quaternion1[:, 0:1], -quaternion1[:, 1:]), dim=1)
    
    # 计算四元数差
    q_diff = quaternion_multiply(quaternion1_conjugate, quaternion2)
   
    # 单位化四元数差异
    q_diff = quaternion_normalize(q_diff)
    
    return q_diff
    

def quaternion_multiply(q1, q2):
    """计算两个四元数的乘积"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), dim=1)


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    """
        KalmanNet 的神经网络实现，包含初始化系统动态模型和 Kalman 增益网络的方法。
    """
    def __init__(self):
        super().__init__()
    
    def NNBuild(self, SysModel, args):

        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        # Number of neurons in the 1st hidden layer
        #H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        #H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):

        self.seq_len_input = 1 # KNet calculates time-step by time-step

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        self.m = 4

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
       
        # GRU to track S 
        self.d_input_S = self.n ** 2 + 2 * (self.n * 2) * args.in_mult_KNet
        self.d_hidden_S = (2 * self.n) ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        
        # Fully connected 1 用于对 Sigma 的输出进行降维和非线性变换
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU()).to(self.device)

        # Fully connected 2         36                16
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = 2 * self.n * self.m  
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU()).to(self.device)
        
        # Fully connected 7
        self.d_input_FC7 = 2 * (self.n * 2)
        self.d_output_FC7 = 2 * (self.n * 2) * args.in_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU()).to(self.device)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        """
                初始化系统动态参数，包括状态转移函数、观测函数及状态/观测的维度。

                参数:
                    f: 状态转移函数，用于从 t-1 更新到 t 的状态。
                    h: 观测函数，用于从状态生成观测。
                    m: 状态向量的维度。
                    n: 观测向量的维度。
        """
        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, train_target_batch, y_training):
        
        #初始化状态m1x第一步的先验 
        #self.m1x_prior = train_target_batch
        #初始化状态m1x第一步的后验 X(t|t)    
        self.m1x_posterior = train_target_batch
     
        #初始化状态m1x第一步的后验预测  X(t-1|t-1)  
        self.m1x_posterior_previous = self.m1x_posterior

        #初始化状态m1x第一步的先验预测  
        self.m1x_prior_previous = self.m1x_posterior

        #初始化测量值y y(t-1)
        self.m1y = y_training

    ######################
    ### Compute Priors ###
    ######################   

    def step_prior(self, train_input_gyr):
        # 时间步长
        dt = 7.0/2000.0
        dm = 0

        # Predict the 1-st moment of x (单个数值）x(t|t-1) [N_T, 3]
        self.m1x_prior = self.f(self.m1x_posterior, train_input_gyr, dt)

        # Predict the 1-st moment of y（单个数值）y(t|t-1) [N_T, 6]
        self.y_previous = self.h(self.m1x_prior, dm)


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # both in size [batch_size, 2*n]
        obs_diff = y - self.y_previous
        obs_innov_diff = y - self.m1y

        # both in size [batch_size, m]
        # 前一时刻后验与当前后验的 相对四元数：状态更新差异
        fw_evol_diff = quaternion_difference(self.m1x_posterior, self.m1x_posterior_previous)
        
        # 前一时刻后验状态与当前预测值的 相对四元数：状态演化差异
        fw_update_diff = quaternion_difference(self.m1x_posterior, self.m1x_prior_previous)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)
                 

        # Kalman Gain Network Step [1, batch_size, m*m]
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)


        # Reshape Kalman Gain to a Matrix [batch_size, 4, 6]
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, 2 * self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, train_input_gyr, train_input_y):

        # Compute Priors self.m1x_prior_previous self.y_previous [batch_size, n&m]
        self.step_prior(train_input_gyr)

        # Compute Kalma n Gain 
        self.step_KGain_est(train_input_y) 

        # [batch_size, n] 预测测量值-预测值 角度
        dy = train_input_y - self.y_previous
        dy = dy.unsqueeze(2)
        
        self.KGain = torch.clamp(self.KGain, min=-1e6, max=1e6)
        dy = torch.clamp(dy, min=-1e6, max=1e6)

        INOV = torch.bmm(self.KGain, dy) #[batch, 4, 1] 
        
        # 前一时刻后验 X(t-1|t-1)
        self.m1x_posterior_previous = self.m1x_posterior #[batch, 4,]

        # 后验更新
        self.m1x_posterior = self.m1x_prior_previous + INOV.squeeze(-1)

        # update y_prev
        self.m1y = train_input_y

        # return
        return self.m1x_posterior

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_update_diff 
        out_FC5 = self.FC5(in_FC5) 

        # Q-GRU
        in_Q = out_FC5 
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q) 

        # FC 6
        in_FC6 = fw_evol_diff 
        out_FC6 = self.FC6(in_FC6) 

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
 
        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2) 

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2) #([1, 56, 21])
        out_FC3 = self.FC3(in_FC3) #([1, 56, 16])

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2) #([1, 56, 32])
        out_FC4 = self.FC4(in_FC4) #([1, 56, 16])

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4 

        return out_FC2
    ###############
    ### Forward ###
    ###############

    def forward(self, train_input_gyr, y_t):
        return self.KNet_step(train_input_gyr, y_t)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self, batch_size):
        self.batch_size = batch_size
        weight = next(self.parameters()).data
        # [1, 56, 9] self.d_hidden_S = self.n ** 2
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_() 
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        # [1, 56, 16] self.d_hidden_Sigma = self.m ** 2
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        # [1, 56, 16] self.d_hidden_Q = self.m ** 2
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion



