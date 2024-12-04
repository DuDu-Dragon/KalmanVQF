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
import numpy as np


class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline03_" + self.modelName + ".pt"

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
        self.N_B = args.batch  # Number of Samples in Batch 
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha # Composition loss factor

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def loss_fn(self, q1, q2):
        """
        计算基于四元数叉乘结果虚部的 L1 范数损失。

        参数:
        q1, q2: 形状为 (N, T, 4) 的四元数张量，N 为批次大小，T 为每批次的四元数数量。

        返回:
        L1 范数损失值
        """
        q1 = F.normalize(q1, p=2, dim=-1)  # 归一化四元数
        #q2 = F.normalize(q2, p=2, dim=-1)  # 真值已经归一化过

        # # 计算 q1 的 L2 范数
        # q1_norm = torch.norm(q1, p=2, dim=-1)  # 范数形状为 (N, T)

        # # 判断是否单位化
        # is_normalized = torch.allclose(q1_norm, torch.ones_like(q1_norm), atol=1e-3)
        # if (is_normalized):
        #     print(f"q1单位化")
        # else:
        #     print(f"q1未单位化")

        # 提取虚部 (x, y, z)
        im_q1 = q1[..., 1:]  # q1 的虚部: [x1, y1, z1], 形状为 (N, T, 3)
        im_q2 = q2[..., 1:]  # q2 的虚部: [x2, y2, z2], 形状为 (N, T, 3)
        
        # q2 的共轭虚部（假设共轭只反转虚部符号）
        im_q2_conj = -im_q2  # [-x2, -y2, -z2], 形状为 (N, T, 3)
        
        # 计算叉乘
        cross_product = torch.cross(im_q1, im_q2_conj, dim=-1)  # 叉乘结果形状为 (N, T, 3)
        
        # 计算叉乘结果虚部的 L1 范数
        l1_norm = torch.norm(cross_product, p=1, dim=-1)  # L1 范数，形状为 (N, T)
        
        # 计算最终损失
        loss = 2 * torch.mean(l1_norm)  # 平均损失
        
        return loss


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

    def NNTrain(self, SysModel, cv_input_acc, cv_input_gyr, cv_input_mag, cv_target, train_input_acc, train_input_gyr, train_input_mag, train_target):
        print("################ Start Training! ################")
        self.N_E = len(train_input_gyr) #训练数据样本数
        self.N_CV = len(cv_input_gyr) #交叉验证数据样本数

        ### 每个训练周期（epoch）的均方误差MSE
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        self.Loss_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_linear_opt = 1000 #初始化为一个很大的值，用于跟踪最佳交叉验证损失
        self.MSE_cv_idx_opt = 0 # 用于跟踪达到最佳交叉验证损失的周期索引。

        #torch.autograd.set_detect_anomaly(True) # 速度超级慢

        # 早停机制的参数
        patience = 10  # 最大容忍验证损失未改善的次数
        no_improve_count = 0  # 记录连续未改善的次数

        for ti in range(0, self.N_steps):
            print(f"-------Epoch {ti} / {self.N_steps}-------")
            step_start_time = time.time()

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B 
            # Init Hidden State
            batch_size = train_input_gyr.shape[0]
            # print("------batch_size1", batch_size)  #ToDo 4 1984
            self.model.init_hidden_KNet(batch_size)
            x_training_batch = train_input_gyr.to(self.device) # [batch_size, 1000, 3]  ToDo 输入 陀螺仪数据
            
            # 拼接y的测量值  ToDo 输出 加速度和磁力计--拼接
            y_training_batch = torch.cat([-train_input_acc, train_input_mag], dim=2).to(self.device) # [batch_size, 1000, 6]
            train_target_batch = train_target.to(self.device)
            x_out_training_batch = torch.zeros([batch_size, self.N_B, SysModel.m]).to(self.device)#[batch_size,1000,4]
            
            # 初始化序列
            self.model.InitSequence(train_target_batch[:, 0, :], y_training_batch[:, 0, :])
            # 初始化初值  四元数
            x_out_training_batch[:,0,:] = train_target_batch[:, 0, :]

            # ToDo  [0]时刻 gyro 预测[1]时刻姿态 ，[1]时刻四元数姿态预测[1]时刻 y
            for t in range(1, self.N_B):
                x_out_training_batch[:,t,:] =  self.model(x_training_batch[:,t-1,:], y_training_batch[:,t,:])
            
            if torch.isnan(x_out_training_batch).any() or torch.isinf(x_out_training_batch).any():
                print("NaN or Inf detected in x_out_training_batch")

            MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)
            MSE_trainbatch_linear_Metric = self.angle_error_fn(x_out_training_batch, train_target_batch)

            # loss
            self.Loss_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_Metric.item()
            print("--Training Loss:", self.Loss_train_linear_epoch[ti], "[度]", "  Training MSE:", self.MSE_train_linear_epoch[ti], "[度]")
            # 获取每次迭代的结束时间
            step_end_time_train = time.time()
            # 计算这一轮迭代的耗时
            step_duration_train = step_end_time_train - step_start_time
            #print(f"  Training took {step_duration_train:.2f} seconds")


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

            def check_gradients(model):
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f'NaN found in gradient of {name}')
                        if torch.isinf(param.grad).any():
                            print(f'Inf found in gradient of {name}')
                    else:
                        print(f'No gradient for {name}')
            # 检查梯度
            check_gradients(self.model)
            
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
                # 拼接y的测量值
                y_cv_batch = torch.cat([-cv_input_acc, cv_input_mag], dim=2).to(self.device) # [batch_size, 1000, 6]

                x_out_cv_batch[:,0,:] = cv_target_batch[:, 0, :]

                self.model.InitSequence(cv_target_batch[:, 0, :], y_cv_batch[:, 0, :])

                for t in range(1, SysModel.T_test):
                    x_out_cv_batch[:, t,: ] = self.model(x_cv_batch[:,t-1,:], y_cv_batch[:,t,:])

                MSE_cvbatch_linear_Metric = self.angle_error_fn(x_out_cv_batch, cv_target_batch)
                
                if torch.isnan(x_out_cv_batch).any() or torch.isinf(x_out_cv_batch).any():
                    print("NaN or Inf detected in x_out_cv_batch")

                #  Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_Metric.item()
                print("--Validating MSE:", self.MSE_cv_linear_epoch[ti], "[度]")

                # 更新最佳验证损失并重置计数
                if self.MSE_cv_linear_epoch[ti] < self.MSE_cv_linear_opt:
                    self.MSE_cv_linear_opt = self.MSE_cv_linear_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    no_improve_count = 0  # 重置未改善计数
                else:
                    no_improve_count += 1  # 无改进次数加一

                # # 检查早停条件
                # if no_improve_count >= patience:
                #     print(f"早停触发！在 Epoch {ti} 停止训练。最佳验证损失: {self.MSE_cv_linear_opt }，发生在 Epoch {self.MSE_cv_idx_opt}")
                #     break

                # 获取每次迭代的结束时间
                step_end_time_val = time.time()
                # 计算这一轮迭代的耗时
                step_duration_val = step_end_time_val - step_end_time_train
                #print(f"  Validating took {step_duration_val:.2f} seconds")


            ########################
            ### Training Summary ###
            ########################


            if (ti > 1):
                d_train = self.MSE_train_linear_epoch[ti] - self.MSE_train_linear_epoch[ti - 1]
                d_cv = self.MSE_cv_linear_epoch[ti] - self.MSE_cv_linear_epoch[ti - 1]
                print("--Diff Training MSE:", d_train, "[度]", "  Diff Validating MSE:", d_cv, "[度]")

            print("--Optimal epoch idx:", self.MSE_cv_idx_opt, "  Optimal MSE:", self.MSE_cv_linear_opt, "[度]")

            step_end_time = time.time()
            # 计算这一轮迭代的耗时
            step_duration_total = step_end_time - step_start_time
            print(f"Epoch {ti} totally took {step_duration_total:.2f} seconds")

        return [self.MSE_cv_linear_epoch, self.MSE_train_linear_epoch, self.MSE_cv_idx_opt, self.MSE_cv_linear_opt]

    def NNTest(self, SysModel, test_acc, test_gyr, test_mag, test_target):
        print("################ Start Testing! ################")
        self.N_T = test_gyr.shape[0]
        SysModel.T_test = self.N_B  #测试数量
        batch_size = self.N_T

        test_target_batch = test_target.to(self.device)
        x_out_test = torch.zeros([batch_size, SysModel.T_test, SysModel.m]).to(self.device)
        x_test_batch = test_gyr.to(self.device)
        y_test_batch = torch.cat([-test_acc, test_mag], dim=2).to(self.device) # [batch_size, 1000, 6]
        
        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet(self.model.batch_size)
        torch.no_grad()

        start = time.time()

        x_out_test[:,0,:] = test_target_batch[:, 0, :]
        self.model.InitSequence(test_target_batch[:, 0, :], y_test_batch[:, 0, :])
       
        for t in range(1, SysModel.T_test):
            x_out_test[:, t, :] = self.model(x_test_batch[:, t-1,:], y_test_batch[:, t,:])
                                          
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = self.angle_error_fn(x_out_test, test_target_batch)

        # Print MSE 
        #str = self.modelName + "-" + "MSE Test:"
        print("Test linear_avg:", self.MSE_test_linear_avg, "[度]")
        print("Inference Time:", t)

        return [self.MSE_test_linear_avg, x_out_test, t]
