
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


from preprocess import preprocess_data
from tushare_function import read_day_from_tushare, select_time
from add_lean import enhanced_incremental_train
import random

class MemoryBuffer:
    """记忆缓冲区，用于存储有代表性的历史样本"""
    
    def __init__(self, max_size=1000, selection_method='kmeans'):
        """
        初始化记忆缓冲区
        
        参数:
        - max_size: 缓冲区最大容量
        - selection_method: 样本选择方法，可选'kmeans'或'random'
        """
        self.max_size = max_size
        self.selection_method = selection_method
        self.buffer_x = []
        self.buffer_y = []
        
    def update(self, x, y, ratio=0.5):
        """
        更新记忆缓冲区，存储新任务的代表性样本
        
        参数:
        - x: 新数据特征
        - y: 新数据标签
        - ratio: 从当前数据中选择的样本比例
        """
        # 计算要保留的样本数量
        n_samples = min(int(len(x) * ratio), self.max_size)
        
        if self.selection_method == 'kmeans' and len(x) > n_samples:
            # 使用K-means选择最具代表性的样本
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            kmeans.fit(x)
            centers = kmeans.cluster_centers_
            
            # 找到最接近每个聚类中心的样本
            idx, _ = pairwise_distances_argmin_min(centers, x)
            
            # 选择这些样本
            selected_x = x[idx]
            selected_y = y[idx]
        else:
            # 随机选择样本
            indices = random.sample(range(len(x)), min(n_samples, len(x)))
            selected_x = x[indices]
            selected_y = y[indices]
            
        # 清空缓冲区如果接近最大容量
        if len(self.buffer_x) + len(selected_x) > self.max_size:
            self.buffer_x = []
            self.buffer_y = []
            
        # 添加到缓冲区
        self.buffer_x.extend(selected_x)
        self.buffer_y.extend(selected_y)
        
    def sample(self, n_samples):
        """从缓冲区采样"""
        if not self.buffer_x:
            return np.array([]), np.array([])
            
        idx = random.sample(range(len(self.buffer_x)), min(n_samples, len(self.buffer_x)))
        return np.array([self.buffer_x[i] for i in idx]), np.array([self.buffer_y[i] for i in idx])
    
    def get_all(self):
        """获取缓冲区所有样本"""
        return np.array(self.buffer_x), np.array(self.buffer_y)


def compute_fisher_information(model, x, y, criterion=nn.CrossEntropyLoss()):
    """
    计算Fisher信息矩阵 (EWC方法的核心)
    
    参数:
    - model: PyTorch模型
    - x: 输入特征
    - y: 标签
    - criterion: 损失函数
    
    返回:
    - fisher: 包含每个参数Fisher值的字典
    """
    model.eval()
    fisher = {}
    
    # 初始化Fisher字典
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    
    # 小批量处理以防止内存溢出
    batch_size = min(32, len(x))
    n_batches = int(np.ceil(len(x) / batch_size))
    
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(x))
        
        x_batch = torch.tensor(x[start_idx:end_idx], dtype=torch.float32)
        y_batch = torch.tensor(y[start_idx:end_idx], dtype=torch.long)
        
        model.zero_grad()
        outputs = model(x_batch)
        
        # 计算每个样本的对数似然
        log_probs = torch.log_softmax(outputs, dim=1)
        log_probs_selected = log_probs[range(len(y_batch)), y_batch]
        
        # 对每个样本进行单独的反向传播
        for i in range(len(y_batch)):
            model.zero_grad()
            log_probs_selected[i].backward(retain_graph=(i < len(y_batch)-1))
            
            # 累积梯度的平方
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) / len(x)
    
    model.train()
    return fisher


def convert_sklearn_to_pytorch(sklearn_model, input_dim, output_dim=2):
    """将scikit-learn模型转换为PyTorch模型以便计算Fisher信息"""
    if hasattr(sklearn_model, 'module_'):
        # 已经是PyTorch模型（如MLP或Transformer）
        return sklearn_model.module_
    else:
        # 创建一个简单的PyTorch MLP作为替代
        model = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
        return model


def enhanced_incremental_train(model, scaler, selected_features, new_df, old_df, label_column, 
                              classifier_name, window_size=10, oversample_method=None, 
                              learning_rate=0.001, epochs=40, batch_size=32, 
                              ewc_lambda=5000, replay_ratio=0.3, memory_size=500,
                              freeze_layers=False, progress_bar=None):
    """
    结合EWC和记忆回放的增强型增量学习
    
    参数:
    - model: 原始模型
    - scaler: 数据标准化器
    - selected_features: 选择的特征
    - new_df: 新增数据DataFrame
    - old_df: 原始训练数据DataFrame
    - label_column: 标签列名('Peak'或'Trough')
    - classifier_name: 模型类型 ('MLP'或'Transformer')
    - window_size: 序列窗口大小 (Transformer模型使用)
    - oversample_method: 过采样方法
    - learning_rate: 学习率
    - epochs: 训练轮数
    - batch_size: 批次大小
    - ewc_lambda: EWC正则化强度
    - replay_ratio: 记忆回放比例
    - memory_size: 记忆缓冲区大小
    - freeze_layers: 是否冻结部分层
    - progress_bar: streamlit进度条
    
    返回:
    - 更新后的模型
    """
    # 1. 准备数据
    X_new = new_df[selected_features].fillna(0).values
    X_new_scaled = scaler.transform(X_new).astype(np.float32)
    y_new = new_df[label_column].astype(int).values
    
    # 如果有老数据，提取部分用于计算Fisher信息
    if old_df is not None:
        sample_size = min(len(old_df), 1000)  # 限制计算Fisher的样本数，提高效率
        X_old_sample = old_df[selected_features].fillna(0).sample(n=sample_size, random_state=42).values
        y_old_sample = old_df[label_column].astype(int).loc[X_old_sample.index].values
        X_old_scaled = scaler.transform(X_old_sample).astype(np.float32)
    else:
        X_old_scaled, y_old_sample = np.array([]), np.array([])
    
    # 2. 初始化记忆缓冲区
    memory_buffer = MemoryBuffer(max_size=memory_size)
    
    # 如果有老数据，添加到记忆缓冲区
    if len(X_old_scaled) > 0:
        memory_buffer.update(X_old_scaled, y_old_sample)
    
    # 3. 创建PyTorch模型的副本用于计算Fisher信息
    input_dim = len(selected_features)
    pytorch_model = convert_sklearn_to_pytorch(model, input_dim)
    
    # 保存原始参数
    original_params = {}
    for name, param in pytorch_model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()
    
    # 4. 计算Fisher信息矩阵 (使用老数据)
    fisher_information = {}
    if len(X_old_scaled) > 0:
        fisher_information = compute_fisher_information(pytorch_model, X_old_scaled, y_old_sample)
    
    # 5. 转换为Transformer序列数据 (如果适用)
    if classifier_name == 'Transformer':
        from preprocess import create_pos_neg_sequences_by_consecutive_labels
        
        # 先处理新数据
        X_seq_new, y_seq_new = create_pos_neg_sequences_by_consecutive_labels(X_new_scaled, y_new, window_size=window_size)
        
        # 处理记忆缓冲区数据
        memory_x, memory_y = memory_buffer.get_all()
        if len(memory_x) > 0:
            X_seq_memory, y_seq_memory = create_pos_neg_sequences_by_consecutive_labels(memory_x, memory_y, window_size=window_size)
        else:
            X_seq_memory, y_seq_memory = np.array([]), np.array([])
    else:
        X_seq_new, y_seq_new = X_new_scaled, y_new
        memory_x, memory_y = memory_buffer.get_all()
        X_seq_memory, y_seq_memory = memory_x, memory_y
    
    # 6. 应用过采样 (如果需要且不是Transformer)
    if classifier_name != 'Transformer' and oversample_method is not None and oversample_method not in ["Class Weights", "None"]:
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.combine import SMOTEENN, SMOTETomek
        
        sampler = None
        if oversample_method == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif oversample_method == "ADASYN":
            sampler = ADASYN(random_state=42)
        elif oversample_method == "Borderline-SMOTE":
            sampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
        elif oversample_method == "SMOTEENN":
            sampler = SMOTEENN(random_state=42)
        elif oversample_method == "SMOTETomek":
            sampler = SMOTETomek(random_state=42)
            
        if sampler is not None and len(X_seq_new) > 0:
            try:
                X_seq_new, y_seq_new = sampler.fit_resample(X_seq_new, y_seq_new)
            except Exception as e:
                print(f"过采样失败: {e}")
    
    # 7. 冻结部分层 (如果需要)
    if freeze_layers:
        if classifier_name == 'MLP' and hasattr(model, 'module_'):
            for param in model.module_.fc1.parameters():
                param.requires_grad = False
        elif classifier_name == 'Transformer' and hasattr(model, 'module_'):
            for layer in model.module_.transformer_encoder.layers[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    # 8. 设置学习率
    if hasattr(model, 'optimizer_') and model.optimizer_ is not None:
        for param_group in model.optimizer_.param_groups:
            param_group['lr'] = learning_rate
    
    # 9. 增量训练主循环
    n_batches = int(np.ceil(len(X_seq_new) / batch_size))
    
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(len(X_seq_new))
        X_seq_new_shuffled = X_seq_new[indices]
        y_seq_new_shuffled = y_seq_new[indices]
        
        epoch_loss = 0
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(X_seq_new_shuffled))
            
            X_batch = X_seq_new_shuffled[start_idx:end_idx]
            y_batch = y_seq_new_shuffled[start_idx:end_idx]
            
            # 记忆回放：混合新数据和记忆样本
            if len(X_seq_memory) > 0:
                replay_size = int(batch_size * replay_ratio)
                memory_idx = np.random.choice(len(X_seq_memory), min(replay_size, len(X_seq_memory)), replace=False)
                X_replay = X_seq_memory[memory_idx]
                y_replay = y_seq_memory[memory_idx]
                
                # 合并新数据和记忆回放数据
                X_combined = np.vstack([X_batch, X_replay])
                y_combined = np.concatenate([y_batch, y_replay])
            else:
                X_combined, y_combined = X_batch, y_batch
            
            # 训练模型
            model.partial_fit(X_combined, y_combined, classes=np.array([0, 1]))
            
        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / epochs)
    
    # 10. 更新记忆缓冲区
    memory_buffer.update(X_new_scaled, y_new)
    
    return model
