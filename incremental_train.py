from models import time_aware_oversampling
from preprocess import create_pos_neg_sequences_by_consecutive_labels
import torch
# ========== 模型微调的辅助函数 ========== #
def incremental_train_for_label(model, scaler, selected_features, df_new, label_column, classifier_name, 
                                window_size=10, oversample_method=None, new_lr=None, new_epochs=5, 
                                freeze_option="none", old_df=None, mix_ratio=1.0, progress_bar=None,
                                early_stopping=True, val_size=0.2, patience=3):
    """
    使用新数据对已有模型进行微调训练（微调），支持多种冻结策略和验证集监控：
      - 如果提供了 old_df，则从 old_df 中随机抽取 mix_ratio 倍于新数据样本数的旧数据，与新数据混合训练。
      - new_lr: 微调阶段使用的学习率
      - new_epochs: 对混合数据进行微调的 epoch 数
      - freeze_option: 冻结策略选项 ["none", "first_layer", "second_layer", "all", "partial"]
      - early_stopping: 是否启用早停
      - val_size: 验证集比例
      - patience: 早停耐心值，连续多少轮验证集性能未提升则停止
      - progress_bar: 可选，streamlit 的进度条控件，用于显示训练进度
    
    Returns:
        model: 微调后的模型
        best_val_acc: 最佳验证集准确率
        epoch_stopped: 实际训练的轮数
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 1) 提取新数据
    X_new = df_new[selected_features].fillna(0)
    X_new_scaled = scaler.transform(X_new).astype(np.float32)
    y_new = df_new[label_column].astype(int).values
    y_new = torch.tensor(y_new).long()

    # 2) 如果提供了旧数据，则进行混合训练
    if old_df is not None:
        sample_size = int(len(X_new) * mix_ratio)
        sample_size = min(sample_size, len(old_df))
        X_old_sample = old_df[selected_features].fillna(0).sample(n=sample_size, random_state=42)
        y_old_sample = old_df[label_column].astype(int).loc[X_old_sample.index].values
        X_old_scaled = scaler.transform(X_old_sample).astype(np.float32)
        X_combined = np.concatenate([X_new_scaled, X_old_scaled], axis=0)
        y_combined = np.concatenate([y_new, y_old_sample], axis=0)
    else:
        X_combined = X_new_scaled
        y_combined = y_new

    # 3) 对于非 Transformer 模型，若需要过采样
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
        elif oversample_method == "Time-Aware":
            X_combined, y_combined = time_aware_oversampling(X_combined, y_combined, recency_weight=0.9, sequence_length=60)
            sampler = None
        
        if sampler is not None:
            X_combined, y_combined = sampler.fit_resample(X_combined, y_combined)

    # 4) 如果启用早停，则划分验证集
    if early_stopping:
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined, test_size=val_size, random_state=42, stratify=y_combined
        )
    else:
        X_train, y_train = X_combined, y_combined
        X_val, y_val = None, None

    # 5) 对于 Transformer 模型，将数据转换为时序数据
    if classifier_name == 'Transformer':
        if X_val is not None:
            X_seq_train, y_seq_train = create_pos_neg_sequences_by_consecutive_labels(X_train, y_train)
            X_seq_val, y_seq_val = create_pos_neg_sequences_by_consecutive_labels(X_val, y_val)
            X_input_train, y_input_train = X_seq_train, y_seq_train
            X_input_val, y_input_val = X_seq_val, y_seq_val
        else:
            X_seq, y_seq = create_pos_neg_sequences_by_consecutive_labels(X_train, y_train)
            X_input_train, y_input_train = X_seq, y_seq
            X_input_val, y_input_val = None, None
    else:
        X_input_train, y_input_train = X_train, y_train
        X_input_val, y_input_val = X_val, y_val

    # 6) 调整微调学习率
    if new_lr is not None and hasattr(model, 'optimizer_') and model.optimizer_ is not None:
        for param_group in model.optimizer_.param_groups:
            param_group['lr'] = new_lr

    # 7) 根据选择的冻结策略冻结不同层
    if classifier_name == 'MLP':
        # 解冻所有层（重置）
        for param in model.module_.parameters():
            param.requires_grad = True
            
        if freeze_option == "first_layer":
            # 只冻结第一层
            for param in model.module_.fc1.parameters():
                param.requires_grad = False
        elif freeze_option == "second_layer":
            # 只冻结第二层
            for param in model.module_.fc2.parameters():
                param.requires_grad = False
        elif freeze_option == "all":
            # 冻结所有层
            for param in model.module_.parameters():
                param.requires_grad = False
        elif freeze_option == "partial":
            # 部分冻结第一层（冻结前半部分神经元的权重）
            # 注意：PyTorch不允许设置单个权重的requires_grad
            # 但我们可以在前向传播时用掩码实现类似效果
            
            # 我们不直接修改requires_grad，而是创建一个掩码
            # 这需要修改模型类，这里我们采用更简单的方式：
            # 只冻结权重的行，而不是单个元素
            
            fc1_size = model.module_.fc1.weight.shape[0]
            half_size = fc1_size // 2
            
            # 创建新的权重和偏置参数
            new_weight = model.module_.fc1.weight.clone().detach()
            if model.module_.fc1.bias is not None:
                new_bias = model.module_.fc1.bias.clone().detach()
            
            # 替换原来的权重和偏置
            model.module_.fc1.weight = torch.nn.Parameter(new_weight)
            if model.module_.fc1.bias is not None:
                model.module_.fc1.bias = torch.nn.Parameter(new_bias)
            
            # 冻结前半部分神经元的权重行
            for i in range(half_size):
                model.module_.fc1.weight[i].requires_grad = False
                
            # 偏置通常不是部分冻结的，但如果需要：
            if model.module_.fc1.bias is not None:
                # 创建新的偏置掩码
                bias_mask = torch.ones_like(model.module_.fc1.bias, dtype=torch.bool)
                bias_mask[:half_size] = False
                
                # 应用掩码
                model.module_.fc1.bias.requires_grad_(True)  # 先全部解冻
                model.module_.fc1.bias.register_hook(lambda grad: grad * bias_mask)
    
    elif classifier_name == 'Transformer':
        # 解冻所有层（重置）
        for param in model.module_.parameters():
            param.requires_grad = True
            
        if freeze_option == "first_layer":
            # 冻结输入线性层
            for param in model.module_.input_linear.parameters():
                param.requires_grad = False
        elif freeze_option == "encoder_layers":
            # 冻结Transformer编码器层（除最后一层）
            num_layers = len(model.module_.transformer_encoder.layers)
            for i in range(num_layers - 1):
                for param in model.module_.transformer_encoder.layers[i].parameters():
                    param.requires_grad = False
        elif freeze_option == "output_layer":
            # 冻结输出层
            for param in model.module_.fc.parameters():
                param.requires_grad = False
        elif freeze_option == "all":
            # 冻结所有层
            for param in model.module_.parameters():
                param.requires_grad = False

    # 8) 多 epoch 微调，同时更新进度条（如果提供）
    best_val_acc = 0.0
    early_stop_counter = 0
    epoch_stopped = new_epochs
    
    # 检查是否所有参数都被冻结了
    all_frozen = True
    for param in model.module_.parameters():
        if param.requires_grad:
            all_frozen = False
            break
    
    # 如果所有参数都被冻结，则跳过训练过程
    if all_frozen:
        print("警告：所有参数都被冻结，模型将不会更新")
        # 仍然更新进度条以提供反馈
        if progress_bar is not None:
            progress_bar.progress(1.0)
        
        # 如果启用了验证，仍然评估一次性能
        if early_stopping and X_input_val is not None:
            y_val_pred = model.predict(X_input_val)
            best_val_acc = accuracy_score(y_input_val, y_val_pred)
        
        epoch_stopped = 0
    else:
        # 正常训练流程
        for epoch in range(new_epochs):
            # 训练一轮
            model.partial_fit(X_input_train, y_input_train, classes=np.array([0, 1]))
            
            # 在验证集上评估（如果启用早停）
            if early_stopping and X_input_val is not None:
                # 获取验证集预测
                y_val_pred = model.predict(X_input_val)
                val_acc = accuracy_score(y_input_val, y_val_pred)
                
                # 更新最佳验证集准确率
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                # 如果连续 patience 轮验证集性能未提升，则停止训练
                if early_stop_counter >= patience:
                    epoch_stopped = epoch + 1
                    break
            
            # 更新进度条
            if progress_bar is not None:
                progress_bar.progress((epoch + 1) / new_epochs)
    
    # 如果没有提前停止，best_val_acc可能仍为0，此时计算一次
    if early_stopping and best_val_acc == 0 and X_input_val is not None:
        y_val_pred = model.predict(X_input_val)
        best_val_acc = accuracy_score(y_input_val, y_val_pred)
    
    return model, best_val_acc, epoch_stopped