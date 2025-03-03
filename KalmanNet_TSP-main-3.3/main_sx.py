import os
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
import scipy.io
import numpy as np
import random

# Load data (assuming you have a .mat file and use scipy.io)
from scipy.io import loadmat

from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config

from Pipelines.Pipeline_EKF import Pipeline_EKF

from datetime import datetime

from KNet.KalmanNet_1Dconv import KalmanNetNN

from Simulations.Lorenz_Atractor.parameters_sx import  m, n, m1x_0, m2x_0,\
f, h,  load_data,  Q_structure, R_structure

import matplotlib.pyplot as plt
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

print("Pipeline Start")
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
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps  = 666 # 训练次数
args.sequence_length = 700 # 一个批次的长度
args.lr = 1e-3
args.wd = 1e-4
args.T = args.sequence_length 
args.T_test = args.sequence_length 

### settings for KalmanNet
args.in_mult_KNet = 4
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

chop = False # whether to chop data sequences into shorter sequences
switch = 'full' # 'full' or 'partial' or 'estH'
   
# noise q and r
r2 = torch.tensor([0.001]) 
v = 1e-5
q2 = torch.mul(v,r2)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("Q: ", Q )
print("R: ", R )

#############################
###   load data DT case   ###
#############################

validation_files = ['05_undisturbed_slow_rotation_with_breaks_B.mat','15_undisturbed_fast_translation_A.mat','25_disturbed_tapping_B.mat']  # 验证集文件名列表
test_files = ['10_undisturbed_slow_translation_A.mat','20_undisturbed_slow_combined_360s.mat','30_disturbed_stationary_magnet_C.mat']  # 测试集文件名列表
path_data = 'Simulations/data_mat'

# validation_files = ['03_undisturbed_slow_rotation_C.mat']  # 验证集文件名列表
# test_files = ['04_undisturbed_slow_rotation_with_breaks_A.mat']  # 测试集文件名列表
# path_data = 'Simulations/test_mat'

seed_torch(seed=1029)
# 运行数据加载
(train_data, val_data, test_data) = load_data(path_data, validation_files, test_files, args.sequence_length)

# 分别解包训练、验证和测试数据
train_imu_acc_batches, train_imu_gyr_batches, train_imu_mag_batches, train_opt_quat_batches, train_batch_sizes = train_data
val_imu_acc_batches, val_imu_gyr_batches, val_imu_mag_batches, val_opt_quat_batches, val_batch_sizes = val_data
test_imu_acc_batches, test_imu_gyr_batches, test_imu_mag_batches, test_opt_quat_batches, test_batch_sizes = test_data

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Data Load")

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


########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(train_input_gyr) #批次数目

MSE_cv_linear_epoch = torch.zeros(args.n_steps)
MSE_train_linear_epoch = torch.zeros(args.n_steps)

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
  print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
  KNet_Pipeline.setTrainingParams(args) 

[MSE_cv_linear_epoch, MSE_train_linear_epoch, MSE_cv_idx_opt, MSE_cv_linear_opt] = KNet_Pipeline.NNTrain\
(sys_model, cv_input_acc, cv_input_gyr, cv_input_mag, cv_target, train_input_acc, train_input_gyr, train_input_mag,train_target)

# Test Neural Network
#KNet_Pipeline.save()
[MSE_test_linear_avg, Knet_out, RunTime] = KNet_Pipeline.NNTest(sys_model,test_input_acc, test_input_gyr,test_input_mag, test_target)



   





