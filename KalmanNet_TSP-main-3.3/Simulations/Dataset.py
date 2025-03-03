import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat

class BroadDataset(Dataset):
    def __init__(self, data_dir, validation_files, test_files, sequence_length, dataset_type):
        """
        初始化 BroadDataset 数据集。
        Args:
            data_dir (str): 数据文件夹路径，包含所有的 .mat 文件。
            validation_files (list): 验证集文件列表。
            test_files (list): 测试集文件列表。
        """
        self.data_dir = data_dir
        self.validation_files = validation_files if validation_files else []
        self.test_files = test_files if test_files else []
        self.file_paths = self._get_all_files(data_dir) # 获取文件夹中所有 .mat 文件的路径
        self.dataset_type = dataset_type 
        self.sequence_length = sequence_length
        self.data = []
        self.split_data = []
        self.load_data()
        self.all_batch_size = len(self.data)
        print("Dataset length (number of sequences): ",self.all_batch_size )

        #self.show_data_structure()


    def _get_all_files(self, data_dir):
        """
        获取文件夹中所有的 .mat 文件路径。
        Args
            data_dir (str): 文件夹路径。
        Returns:
            list: 包含所有 .mat 文件路径的列表。
        """
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')]

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回索引 idx 对应的数据（一个小块的数据）"""
        entry = self.data[idx]

        return {
            'gyro': entry['gyro'],
            'acc': entry['acc'],
            'mag': entry['mag'],
            'quaternion': entry['quaternion'],
            'position': entry['position'],
            'file_name': entry['file_name']
        }
    def __getitem__(self, idx):
        """返回索引 idx 对应的数据（一个小块的数据）"""
        entry = self.data[idx]

        return {
            'gyro': entry['gyro'],
            'acc': entry['acc'],
            'mag': entry['mag'],
            'quaternion': entry['quaternion'],
            'position': entry['position'],
            'file_name': entry['file_name']
        }

    def __len__(self):
        """
        返回切分后的序列数量。
        """
        return len(self.data)
    
    def load_data(self):
        """
        加载对应数据集类型的数据（训练集、验证集或测试集）。
        """
        for file_path in self.file_paths:
            file_name = os.path.basename(file_path)

            # 根据 dataset_type 加载相应的数据文件
            if self.dataset_type == 'train':
                if file_name in self.validation_files or file_name in self.test_files:
                    continue  # 训练集：排除验证集和测试集文件

            elif self.dataset_type == 'val':
                if file_name not in self.validation_files:
                    continue  # 验证集：只加载验证集文件

            elif self.dataset_type == 'test':
                if file_name not in self.test_files:
                    continue  # 测试集：只加载测试集文件

            else:
                raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be 'train', 'val', or 'test'.")

            # 加载 .mat 文件内容
            mat_contents = loadmat(file_path)
            print(f"Processing {file_name}")

            # 加载 IMU 和光学数据
            raw_imu_acc = torch.tensor(mat_contents['imu_acc'], dtype=torch.float32)
            raw_imu_gyr = torch.tensor(mat_contents['imu_gyr'], dtype=torch.float32)
            raw_imu_mag = torch.tensor(mat_contents['imu_mag'], dtype=torch.float32)
            raw_opt_quat = torch.tensor(mat_contents['opt_quat'], dtype=torch.float32)
            raw_opt_pos = torch.tensor(mat_contents['opt_pos'], dtype=torch.float32)

            # 数据清理（如处理 NaN）
            imu_acc, imu_gyr, imu_mag, opt_quat, opt_pos = self.clean_fix_data(
                raw_imu_acc, raw_imu_gyr, raw_imu_mag, raw_opt_quat, raw_opt_pos
            )

            if imu_acc is None or imu_gyr is None or imu_mag is None or opt_quat is None:
                print(f"Skipping {file_name} due to missing data")
                continue

            N = imu_acc.shape[0]
            if N != imu_gyr.shape[0] or N != imu_mag.shape[0] or N != opt_quat.shape[0]:
                print(f"Skipping {file_name} due to inconsistent data lengths")
                continue

            seq_length = len(imu_acc)  # 获取每个序列的总长度

            # 计算可以切分的最大整数倍，去掉末尾不足一个块的部分
            max_full_blocks = seq_length // self.sequence_length  # 计算能切分的完整块数

            # 切分序列数据，每个小块长度为 sequence_length
            for start_idx in range(0, max_full_blocks * self.sequence_length, self.sequence_length):
                # 提取当前小块
                end_idx = start_idx + self.sequence_length
                self.data.append({
                    'gyro': imu_gyr[start_idx:end_idx],
                    'acc': imu_acc[start_idx:end_idx],
                    'mag': imu_mag[start_idx:end_idx],
                    'quaternion': opt_quat[start_idx:end_idx],
                    'position': opt_pos[start_idx:end_idx],
                    'file_name': file_path  # 也可以保存文件名等信息
                })


    # 展示结构，数据加载完毕后可调用
    def show_data_structure(self):
        """
        打印 self.data 的结构信息，显示每个数据项的文件名和包含的键。
        """
        if not self.data:
            print("self.data is empty.")
            return

        # 打印每个数据条目中的文件名和数据的键
        for i, data_entry in enumerate(self.data):
            print(f"Data Entry {i+1}:")
            print(f"  File Name: {data_entry['file_name']}")
            print(f"  Keys: {', '.join(data_entry.keys())}")
            print(f"  Example Data Shapes:")
            print(f"    gyro: {data_entry['gyro'].shape}")
            print(f"    acc: {data_entry['acc'].shape}")
            print(f"    mag: {data_entry['mag'].shape}")
            print(f"    quaternion: {data_entry['quaternion'].shape}")
            print(f"    position: {data_entry['position'].shape}")
            print("-" * 40)

    ### input_nan fix
    def slerp(self, q0, q1, taus):
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

    def interpolate_quaternions(self, input_quat):
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
            assert end_idx != start_idx,"end_idx cannot equal start_idx"
            taus = (nan_indices - start_idx).float() / (end_idx - start_idx)
            
            # 批量插值四元数
            interpolated_quats = self.slerp(q0.unsqueeze(0), q1.unsqueeze(0), taus.unsqueeze(1))
            output_quat[nan_indices] = interpolated_quats.squeeze(1)

        return output_quat

    def clean_fix_data(self, input_acc, input_gyr, input_mag, input_quat_nan, input_opt_pos):
        """清理并修复加速度计、陀螺仪、磁力计和四元数数据"""
        quat_nan_mask = torch.isnan(input_quat_nan[:, 0])

        first_valid_idx = torch.nonzero(~quat_nan_mask, as_tuple=False)[0].item()
        last_valid_idx = torch.nonzero(~quat_nan_mask, as_tuple=False)[-1].item()

        cleaned_quat = input_quat_nan[first_valid_idx:last_valid_idx + 1]
        fixed_quat = self.interpolate_quaternions(cleaned_quat)
        cleaned_acc = input_acc[first_valid_idx:last_valid_idx + 1]
        cleaned_gyr = input_gyr[first_valid_idx:last_valid_idx + 1]
        cleaned_mag = input_mag[first_valid_idx:last_valid_idx + 1]
        cleaned_pos = input_opt_pos[first_valid_idx:last_valid_idx + 1]

        if torch.any(torch.isnan(fixed_quat)):
            print("opt_quat 中仍存在 NaN 值")

        return cleaned_acc, cleaned_gyr, cleaned_mag, fixed_quat, cleaned_pos 






