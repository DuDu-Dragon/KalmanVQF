import os
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
import numpy as np
import random
# Load data (assuming you have a .mat file and use scipy.io)
from scipy.io import loadmat
from Simulations.Dataset import BroadDataset
from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Pipelines.Pipeline_EKF import Pipeline_EKF
from datetime import datetime
from KNet.KalmanNet_1Dy import KalmanNetNN
from Simulations.Lorenz_Atractor.parameters import  m, n, m1x_0, m2x_0,\
f, h,  load_data,  Q_structure, R_structure
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from wtconv import WTConv2d


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
args.n_steps  = 1000 # 训练次数
args.sequence_length = 1000 # 一个批次的长度
args.lr = 1e-3
args.wd = 1e-4
args.T = args.sequence_length
args.T_test = args.sequence_length

### settings for KalmanNet
args.in_mult_KNet = 6
args.out_mult_KNet = 6

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

# 设置随机种子函数
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止 Python hash 随机化
    np.random.seed(seed)  # NumPy 随机数种子
    torch.manual_seed(seed)  # PyTorch 随机数种子
    torch.cuda.manual_seed(seed)  # GPU 随机数种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 随机数种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 确保 worker 的随机种子也被初始化
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 设置随机种子
seed_torch(1029)
# 数据集文件夹路径
data_dir = 'Simulations/data_mat'
validation_files = ['05_undisturbed_slow_rotation_with_breaks_B.mat',
                    '15_undisturbed_fast_translation_A.mat',
                    '25_disturbed_tapping_B.mat']  # 验证集文件列表

test_files = ['10_undisturbed_slow_translation_A.mat',
              '20_undisturbed_slow_combined_360s.mat',
              '30_disturbed_stationary_magnet_C.mat']  # 测试集文件列表

train_dataset = BroadDataset(data_dir, validation_files, test_files, args.sequence_length,  dataset_type = 'train') 
# 打印总样本数
train_batch_size = train_dataset.all_batch_size
train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=0, drop_last=True)
print(f"Dataset length (number of sequences): {len(train_dataset)}")

# 遍历 train_dataloader 中的每个批次
for batch in train_dataloader:
    # 假设每个批次是字典，包含多个键（'acc', 'gyr', 'mag', 'quaternion' 等）
    train_input_acc = batch['acc']          # 形状：[batch_size, num_chunks, 700, 3]
    train_input_gyr = batch['gyro']          # 形状：[batch_size, num_chunks, 700, 3]
    train_input_mag = batch['mag']          # 形状：[batch_size, num_chunks, 700, 3]
    train_target = batch['quaternion']  # 形状：[batch_size, num_chunks, 700, 4]

# 查看张量的形状
print(f"train_input_acc.shape: {train_input_acc.shape}")
print(f"train_input_gyr.shape: {train_input_gyr.shape}")
print(f"train_input_mag.shape: {train_input_mag.shape}")
print(f"train_input_quaternion.shape: {train_target.shape}")

# 初始化验证集 Dataset 和 DataLoader
val_dataset = BroadDataset(data_dir, validation_files, test_files, args.sequence_length, dataset_type = 'val')
val_all_batch = val_dataset.all_batch_size
val_dataloader = DataLoader(val_dataset, val_all_batch, shuffle=False, num_workers=0, drop_last=True)

for batch_idx, batch in enumerate(val_dataloader):
    # 提取数据
    cv_input_acc = batch['acc']  # 加速度数据
    cv_input_gyr = batch['gyro']  # 陀螺仪数据
    cv_input_mag = batch['mag']  # 磁力计数据
    cv_target = batch['quaternion']  # 目标（比如位置）
    break  # 只查看第一个批次

# 初始化测试集 Dataset 和 DataLoader
test_dataset = BroadDataset(data_dir, validation_files, test_files, args.sequence_length, dataset_type = 'test')
test_all_batch = test_dataset.all_batch_size
test_dataloader = DataLoader(test_dataset, test_all_batch, shuffle=False, num_workers=0, drop_last=True)

for batch_idx, batch in enumerate(test_dataloader):
    # 提取数据
    test_input_acc = batch['acc']  # 加速度数据
    test_input_gyr = batch['gyro']  # 陀螺仪数据
    test_input_mag = batch['mag']  # 磁力计数据
    test_target = batch['quaternion']  # 目标（比如位置）
    break  # 只查看第一个批次

print(f"train_input shape: {train_input_acc.shape}",f"val_input shape: {cv_input_acc.shape}", f"test_input_acc shape: {test_input_acc.shape}")

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Data Load")

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
KNet_Pipeline.save()
[MSE_test_linear_avg, Knet_out, RunTime] = KNet_Pipeline.NNTest(sys_model,test_input_acc, test_input_gyr,test_input_mag, test_target)



   





