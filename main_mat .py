import os
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
import scipy.io
from Filters.EKF_test import EKFTest
import numpy as np

# Load data (assuming you have a .mat file and use scipy.io)
from scipy.io import loadmat

from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config

from Pipelines.Pipeline_EKF import Pipeline_EKF

from datetime import datetime

from KNet.KalmanNet_nn import KalmanNetNN

from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
f, h,  load_data,  Q_structure, R_structure

import matplotlib.pyplot as plt

print("################ Pipeline Start ################")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1000 # 批次数值大小
args.N_CV = 100
args.N_T = 200
args.T = 2000
args.T_test = 1000
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 160
args.lr = 1e-3
args.wd = 1e-3

### settings for KalmanNet
args.in_mult_KNet = 3
args.out_mult_KNet = 4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop data sequences into shorter sequences
switch = 'full' # 'full' or 'partial' or 'estH'
   
# 噪声配置：观测噪声 r2 和过程噪声 q2
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)  # 转换为线性比率
q2 = torch.mul(v,r2)

# 生成过程噪声协方差 Q 和观测噪声协方差 R
Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

#traj_resultName = ['traj_lorDT_rq1030_T100.pt']
#dataFileName = ['data_lor_v20_rq1030_T100.pt']

#############################
###   load data DT case   ###
#############################


validation_files = ['35_disturbed_attached_magnet_4cm.mat','36_disturbed_attached_magnet_5cm.mat', '37_disturbed_office_A.mat']  # 验证集文件名列表
test_files = ['38_disturbed_office_B.mat','39_disturbed_mixed.mat']  # 测试集文件名列表
path_data = 'Simulations/data_mat'

# 运行数据加载
(train_data, val_data, test_data) = load_data(path_data, validation_files, test_files, batch_size=1000)

# 分别解包训练、验证和测试数据 (num_batches, batch_size, n)
train_imu_acc_batches, train_imu_gyr_batches, train_imu_mag_batches, train_opt_quat_batches, train_batch_nums = train_data
val_imu_acc_batches, val_imu_gyr_batches, val_imu_mag_batches, val_opt_quat_batches, val_batch_nums = val_data
test_imu_acc_batches, test_imu_gyr_batches, test_imu_mag_batches, test_opt_quat_batches, test_batch_nums = test_data

# 初始化系统模型，设置系统动态方程 f 和观测方程 h
sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n) # parameters for GT
sys_model.InitSequence(m1x_0, m2x_0) # 初始化状态和协方差

print("################ Data Load ################")

#训练集的输入数据
train_input_acc = train_imu_acc_batches
train_input_gyr = train_imu_gyr_batches
train_input_mag = train_imu_mag_batches
train_target = train_opt_quat_batches
#验证集的输入数据
cv_input_acc = val_imu_acc_batches
cv_input_gyr = val_imu_gyr_batches
cv_input_mag = val_imu_mag_batches
cv_target = val_opt_quat_batches
#测试集的输入数据
test_input_acc =  test_imu_acc_batches
test_input_gyr = test_imu_gyr_batches
test_input_mag = test_imu_mag_batches
test_target = test_opt_quat_batches

# Model with partial info 使用部分信息的系统模型
sys_model_partial = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model_partial.InitSequence(m1x_0, m2x_0)

########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(train_input_gyr)

MSE_mag_linear_arr = torch.zeros(N_T)# MSE [Linear]
Estimate_q_arr = torch.zeros((N_T, 4), dtype = torch.double)# 估计四元数
Estimate_y_arr = torch.zeros((N_T, 6), dtype = torch.double)

# 创建一个 3x3 的单位矩阵
identity_matrix = torch.eye((3), dtype = torch.double)
# 初始化一个张量，形状为 (N_T, 3, 3)，其中每个元素都是一个 3x3 的单位矩阵
Q_rotate_arr = torch.stack([identity_matrix] * N_T)

dt=7.0/2000.0

print("dt:", dt)

#####################
### Evaluate KNet ###
#####################
if switch == 'full':
  ## KNet with full info ####################################################################################
  ################
  ## KNet full ###
  ################  
  ## Build Neural Network
  print("KNet with full model info")
  KNet_model = KalmanNetNN()
  KNet_model.NNBuild(sys_model, args)
  # ## Train Neural Network
  KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
  KNet_Pipeline.setssModel(sys_model)
  KNet_Pipeline.setModel(KNet_model)
  print("Number of trainable parameters for KNet:", sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
  KNet_Pipeline.setTrainingParams(args) 
  if(chop):
      [MSE_cv_linear_epoch, MSE_train_linear_epoch, MSE_cv_idx_opt, MSE_cv_linear_opt] = KNet_Pipeline.NNTrain \
      (sys_model, cv_input_acc, cv_input_gyr, cv_input_mag, cv_target, train_input_acc, train_input_gyr, train_input_mag,train_target)
  else:
      [MSE_cv_linear_epoch, MSE_train_linear_epoch, MSE_cv_idx_opt, MSE_cv_linear_opt] = KNet_Pipeline.NNTrain \
      (sys_model, cv_input_acc, cv_input_gyr, cv_input_mag, cv_target, train_input_acc, train_input_gyr, train_input_mag, train_target)
  KNet_Pipeline.save()
  # Test Neural Network
  [MSE_test_linear_avg, Knet_out, RunTime] = KNet_Pipeline.NNTest(sys_model,test_input_acc, test_input_gyr, test_input_mag, test_target)



   





