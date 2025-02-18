"""
This file contains the class Pipeline_EKF,
which is used to train and test KalmanNet.
"""

from sys import argv
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from Plot import Plot_extended
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import torch
import os

def quaternion_normalize(q, epsilon=1e-7):
    """
    对四元数进行单位化，若模长接近零则返回 [1, 0, 0, 0]。
    
    参数:
        q (torch.Tensor): 输入的四元数张量，形状为 (batch_size,length, 4)。
        epsilon (float): 防止除以 0 的偏移量，默认为 1e-7。
    返回:
        torch.Tensor: 单位化后的四元数，形状为 (batch_size,length, 4)。
    """
    # 计算四元数的模长，并添加偏移量
    norm = torch.norm(q, dim=2, keepdim=True) + epsilon
    
    # 对四元数进行单位化，模长接近零时返回 [1, 0, 0, 0]
    normalized_q = q / norm
    fallback = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=q.device)
    return torch.where(norm <= epsilon, fallback, normalized_q)
    #return normalized_q

def normalize_y(train_input, epsilon=1e-7):
    """
    对加速度/磁力计张量进行单位化处理。
    
    参数:
    train_input (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, 3]。
    epsilon (float): 一个很小的常数，避免除零错误，默认值为 1e-7。

    返回:
    torch.Tensor: 单位化后的加速度张量，形状同输入。
    """
    # 计算模长，并添加偏移量
    norm = train_input.norm(p=2, dim=2, keepdim=True) + epsilon # [batch_size, sequence_length, 1]
    
    # 对四元数进行单位化，模长接近零时返回 [1, 0, 0, 0]
    normalized_y = train_input / norm
    fallback = torch.tensor([[0.0, 0.0, 0.0]], device=train_input.device)
    return torch.where(norm <= epsilon, fallback, normalized_y)


def save_training_data(save_file, idx_opt, linear_opt, heading_opt, inclination_opt):
    """
    保存训练数据到文件，更新但避免 NaN 值的写入。
    """
    # 检查是否有 NaN 值
    if any(x != x for x in [idx_opt, linear_opt, heading_opt, inclination_opt]):  # NaN 判断条件
        print("NaN detected, skipping update to file.")
        return

    # 格式化数据
    data_line = (f"Optimal idx: {idx_opt}, Optimal: {linear_opt} [度], "
                f"Heading Error: {heading_opt} [度], "
                f"Inclination Error: {inclination_opt} [度]\n")

    # 写入文件（追加模式）
    with open(save_file, "a") as f:
        f.write(data_line)

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps 
        self.N_B = args.sequence_length # Number of Samples in Batch 
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay   

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        
    def loss_diff(self, q1, q2):
        """
        Calculates quaternion that represents the orientation estimation error in the global coordinate system.

        :param q1: First quaternion tensor (e.g., IMU orientation), shape (N, T, 4)
        :param q2: Second quaternion tensor (e.g., OMC orientation), shape (N, T, 4)
        :return: error quaternion tensor, shape (N, T, 4)
        """
        def quatmult(q1, q2):
            """
            Quaternion multiplication.

            :param q1: First quaternion tensor, shape (..., 4)
            :param q2: Second quaternion tensor, shape (..., 4)
            :return: Resulting quaternion tensor, shape (..., 4)
            """
            w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
            w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack((w, x, y, z), dim=-1)
        
        def invquat(q):
            """
            Quaternion inverse.

            :param q: Quaternion tensor, shape (..., 4)
            :return: Inverted quaternion tensor, shape (..., 4)
            """
            w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            
            # Invert the quaternion
            return torch.cat((w.unsqueeze(-1), -x.unsqueeze(-1), -y.unsqueeze(-1), -z.unsqueeze(-1)), dim=-1) 
        
        relative_quat = quatmult(q1, invquat(q2))
        relative_quat = quaternion_normalize(relative_quat)
        
        return  relative_quat
    
    def calculateTotalError(self, q_diff):
        """
        Calculates the total error, i.e., the total absolute rotation angle of the quaternion.

        :param q_diff: error quaternion tensor, shape (N, T, 4)
        :return: total rotation angle tensor in radians, shape (N, T)
        """
        # Extract the scalar part (q0) of the quaternion
        q0 = torch.abs(q_diff[..., 0])
        
        # Ensure numerical stability with torch.clamp and compute the total rotation angle
        total_error = 2 * torch.arccos(torch.clamp(q0, 0, 1))
        
        MSE_total_error = torch.abs(total_error).mean()

        return torch.rad2deg(MSE_total_error)
    
    def calculateHeadingError(self, q_diff_earth):
        """
        Calculates the heading error.

        :param q_diff_earth: error quaternion tensor in global coordinates, shape (N, T, 4)
        :return: heading error tensor in radians, shape (N, T)
        """
        # Extract the scalar part (q0) and z-axis component (qz)
        q0 = q_diff_earth[..., 0]
        qz = q_diff_earth[..., 3]
        
        # Compute the heading error using atan
        heading_error = 2 * torch.arctan(torch.abs(qz / q0))

        MSE_heading_error = heading_error.mean()
        
        return torch.rad2deg(MSE_heading_error)
    
    def calculateInclinationError(self, q_diff_earth):
        """
        Calculates the inclination error.

        :param q_diff_earth: error quaternion tensor in global coordinates, shape (N, T, 4)
        :return: inclination error tensor in radians, shape (N, T)
        """
        # Extract the scalar part (q0) and z-axis component (qz)
        q0 = q_diff_earth[..., 0]
        qz = q_diff_earth[..., 3]
        
        # Compute the magnitude of the scalar and z components
        scalar_z_magnitude = torch.sqrt(q0 ** 2 + qz ** 2)
        
        # Compute the inclination error using arccos
        inclination_error = 2 * torch.arccos(torch.clamp(scalar_z_magnitude, 0, 1))

        MSE_inclination_errot = inclination_error.mean()

        return torch.rad2deg(MSE_inclination_errot)

    def angle_error_fn(self, q1, q2):
        """
        计算两个四元数张量之间的角度差，用于评估模型的性能。

        参数:
        q1, q2: 形状为 (N, T, 4) 的四元数张量，N 为批次大小，T 为每批次的四元数数量。

        返回:
        每个样本的角度误差（以度为单位）
        """
        #q1 = F.normalize(q1, p=2, dim=-1)  # 归一化四元数
        #q2 = F.normalize(q2, p=2, dim=-1)  # 真值已经归一化过
        epsilon = 1e-6
        # 计算四元数内积，按最后一维（4）进行点积，保持 batch_size 和 step 的维度
        dot_product = torch.sum(q1 * q2, dim=-1)  # 点积沿着最后一个维度（即 4 维）

        # 防止由于浮点误差，dot_product 超出 [-1, 1] 范围
        dot_product = torch.clamp(dot_product, min=-1.0 + epsilon, max=1.0 - epsilon)

        # 计算角度（cos^-1）
        theta = 2 * torch.acos(torch.abs(dot_product))  # 得到的是弧度

        # 将弧度转换为度
        theta_deg = torch.rad2deg(theta)  # [batch_size, step]（角度，单位：度）

        # 返回平均角度损失
        return torch.mean(theta_deg)
    

    def NNTrain(self, SysModel, cv_input_acc, cv_input_gyr, cv_input_mag, cv_target, train_input_acc, train_input_gyr, train_input_mag,train_target):
        print("################ Start Training! ################")

        ### 每个训练周期（epoch）的均方误差MSE
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_totalerror_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_heading_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_inclination_epoch = torch.zeros([self.N_steps])
        self.Loss_train_linear_epoch = torch.zeros([self.N_steps])
        self.train_heading_epoch = torch.zeros([self.N_steps])
        self.train_inclination_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
       
        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_linear_opt = 1000 #初始化为一个很大的值，用于跟踪最佳交叉验证损失
        self.MSE_cv_idx_opt = 0 # 用于跟踪达到最佳交叉验证损失的周期索引。

        import os
        #torch.autograd.set_detect_anomaly(True)#速度超级慢
        # 定义保存路径和文件名
        save_folder = "./training_results"  # 文件夹路径
        save_file = os.path.join(save_folder, "FC56:training_log.txt")  # 文件路径

        # 创建文件夹（如果不存在）
        os.makedirs(save_folder, exist_ok=True)


        for ti in range(0, self.N_steps):
            self.model.train() 

            print(f"-------Epoch {ti} / {self.N_steps}-------")
            step_start_time = time.time()

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # 训练清零
            self.optimizer.zero_grad()
            # Training Mode
            self.model.batch_size = self.N_B #1000
            # Init Hidden State
            batch_size = train_input_gyr.shape[0]
            self.model.init_hidden_KNet(batch_size)
            
            x_training_batch = train_input_gyr.to(self.device) # [batch_size, 1000, 3]
            if torch.isnan(train_input_acc).any() or torch.isinf(train_input_acc).any():
                print("NaN or Inf found in input data")

            #对加速度计数据进行单位化
            train_input_acc = normalize_y(train_input_acc)
            # 对磁力计数据进行单位化
            train_input_mag = normalize_y(train_input_mag)
            # 将单位化后的加速度计和磁力计数据拼接
            y_training_batch = torch.cat([-train_input_acc, train_input_mag], dim=2).to(self.device)  # [batch_size, 1000, 6]

            train_target_batch = train_target.to(self.device)
            x_out_training_batch = torch.zeros([batch_size, self.N_B, SysModel.m]).to(self.device)#[batch_size,1000,4]
            
            # 初始化序列
            #self.model.InitSequence(train_target_batch[:, 0, :], y_training_batch[:, 0, :])
            self.model.InitSequence(batch_size, train_target_batch.shape[2] , y_training_batch.shape[2])

            # 初始化初值
            x_out_training_batch[:,0,:] = train_target_batch[:, 0, :]

            # [0]时刻 gyro 预测[1]时刻姿态 ，[1]时刻四元数姿态预测[1]时刻 y
            for t in range(1, self.N_B):
             x_out_training_batch[:,t,:] =  self.model(x_training_batch[:,t-1,:], y_training_batch[:,t,:])
            
            if torch.isnan(x_out_training_batch).any() or torch.isinf(x_out_training_batch).any():
                print("NaN or Inf detected in x_out_training_batch")
                break

            Quation_Diff = self.loss_diff(x_out_training_batch, train_target_batch)

            #MSE_trainbatch_linear_LOSS = self.loss_fn_5(x_out_training_batch, train_target_batch)
            MSE_trainbatch_linear_LOSS = self.calculateTotalError(Quation_Diff)
            MSE_heading_loss = self.calculateHeadingError(Quation_Diff)
            MSE_inclination_loss = self.calculateInclinationError(Quation_Diff)
            #MSE_trainbatch_linear_Metric = self.angle_error_fn(x_out_training_batch, train_target_batch)
            #plot_euler_angles(x_out_training_batch, train_target_batch, ti)

            # loss
            self.Loss_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.train_heading_epoch[ti] = MSE_heading_loss.item()
            self.train_inclination_epoch[ti] = MSE_inclination_loss.item()
            #self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_Metric.item()

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters

            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)

            # def check_gradients(model):
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             if torch.isnan(param.grad).any():
            #                 print(f'NaN found in gradient of {name}')
            #             if torch.isinf(param.grad).any():
            #                 print(f'Inf found in gradient of {name}')
            #         else:
            #             print(f'No gradient for {name}')
            # #检查梯度
            # check_gradients(self.model)
            
            # Clip gradients to prevent exploding gradients
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            cv_batch_size = cv_target.shape[0]
            self.model.init_hidden_KNet(cv_batch_size)
            with torch.no_grad():

                SysModel.T_test = self.N_B  #测试数量 
                cv_target_batch = cv_target.to(self.device)
                x_out_cv_batch = torch.zeros([cv_batch_size, SysModel.T_test, SysModel.m]).to(self.device)
                x_cv_batch = cv_input_gyr.to(self.device)
                
                #单位化 y
                cv_input_acc = normalize_y(cv_input_acc)
                cv_input_mag = normalize_y(cv_input_mag)
                y_cv_batch = torch.cat([-cv_input_acc, cv_input_mag], dim=2).to(self.device) # [batch_size, 1000, 6]

                x_out_cv_batch[:,0,:] = cv_target_batch[:, 0, :]

                self.model.InitSequence(cv_batch_size , cv_target_batch.shape[2] , y_cv_batch.shape[2])

                for t in range(1, SysModel.T_test):
                    x_out_cv_batch[:, t,: ] = self.model(x_cv_batch[:,t-1,:], y_cv_batch[:,t,:])

                cv_diff = self.loss_diff(x_out_cv_batch, cv_target_batch)

                #MSE_cvbatch_linear_LOSS = self.angle_error_fn(x_out_cv_batch, cv_target_batch)
                MSE_cvbatch_total_error = self.calculateTotalError(cv_diff)
                MSE_cv_heading_error = self.calculateHeadingError(cv_diff)
                MSE_cv_inclination_error = self.calculateInclinationError(cv_diff)
                
                if torch.isnan(x_out_cv_batch).any() or torch.isinf(x_out_cv_batch).any():
                    print("NaN or Inf detected in x_out_cv_batch")

                # dB Loss
                #self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_totalerror_epoch[ti] = MSE_cvbatch_total_error.item()
                self.MSE_cv_heading_epoch[ti] = MSE_cv_heading_error.item()
                self.MSE_cv_inclination_epoch[ti] = MSE_cv_inclination_error.item()

                # 获取每次迭代的结束时间
                step_end_time = time.time()
               # 计算这一轮迭代的耗时
                step_duration = step_end_time - step_start_time
                print(f"Epoch {ti} took {step_duration:.2f} seconds")

            ########################
            ### Training Summary ###
            ########################

            if self.MSE_cv_totalerror_epoch[ti] < self.MSE_cv_linear_opt:
                self.MSE_cv_linear_opt = self.MSE_cv_totalerror_epoch[ti]
                self.MSE_cv_heading_opt = self.MSE_cv_heading_epoch[ti]
                self.MSE_cv_inclination_opt = self.MSE_cv_inclination_epoch[ti]
                self.MSE_cv_idx_opt = ti
                # 保存到文件中（确保 self.MSE_* 是你的变量）
                save_training_data(save_file ,self.MSE_cv_idx_opt, 
                            self.MSE_cv_linear_opt, 
                            self.MSE_cv_heading_opt, 
                            self.MSE_cv_inclination_opt)

            print("Training Loss:", self.Loss_train_linear_epoch[ti], "[度]", "Heading Error: ",self.train_heading_epoch[ti],"[度]", 
                  "Inclination Error: ", self.train_inclination_epoch[ti],"[度]")
            print("Validating Error:", self.MSE_cv_totalerror_epoch[ti],"[度]", "Heading Error: ",self.MSE_cv_heading_epoch[ti],"[度]", 
                  "Inclination Error: ", self.MSE_cv_inclination_epoch[ti],"[度]")

            if (ti > 1):
                d_train = self.Loss_train_linear_epoch[ti] - self.Loss_train_linear_epoch[ti - 1]
                d_cv = self.MSE_cv_totalerror_epoch[ti] - self.MSE_cv_totalerror_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[度]", "diff MSE Validation :", d_cv, "[度]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_linear_opt, "[度]","Heading Error: ",self.MSE_cv_heading_opt,"[度]", 
                  "Inclination Error: ", self.MSE_cv_inclination_opt,"[度]")
            

        return [self.MSE_cv_linear_epoch, self.Loss_train_linear_epoch, self.MSE_cv_idx_opt, self.MSE_cv_linear_opt]

    def NNTest(self, SysModel, test_acc, test_gyr, test_mag, test_target):

        self.N_T = test_gyr.shape[0]
        SysModel.T_test = self.N_B  #测试数量
        test_batch_size = self.N_T

        test_target_batch = test_target.to(self.device)
        x_out_test = torch.zeros([test_batch_size, SysModel.T_test, SysModel.m]).to(self.device)
        x_test_batch = test_gyr.to(self.device)

        test_acc = normalize_y(test_acc)
        test_mag = normalize_y(test_mag)
        y_test_batch = torch.cat([-test_acc, test_mag], dim=2).to(self.device) # [batch_size, 1000, 6]
        
        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet(self.model.batch_size)
        torch.no_grad()

        start = time.time()

        x_out_test[:,0,:] = test_target_batch[:, 0, :]

        self.model.InitSequence(test_batch_size , test_target_batch.shape[2] , y_test_batch.shape[2])
       
        for t in range(1, SysModel.T_test):
            x_out_test[:, t, :] = self.model(x_test_batch[:, t-1,:], y_test_batch[:, t,:])
                                          
        end = time.time()
        t = end - start

        test_diff = self.loss_diff(x_out_test, test_target_batch)

        # Average
        self.MSE_test_linear_avg = self.calculateTotalError(test_diff)
        self.MSE_test_heading_error = self.calculateHeadingError(test_diff)
        self.MSE_test_inclination_error = self.calculateInclinationError(test_diff)

        # Print MSE 
        #str = self.modelName + "-" + "MSE Test:"
        print("Test Error :", self.MSE_test_linear_avg, "[度]", "Heading Error :", self.MSE_test_heading_error, "[度]", 
              "Inclination Error :", self.MSE_test_inclination_error, "[度]", )
        
        print("Inference Time:", t)

        return [self.MSE_test_linear_avg, x_out_test, t]
