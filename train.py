# train.py

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score, recall_score, average_precision_score,
    matthews_corrcoef, roc_auc_score, f1_score,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from joblib import Parallel, delayed, parallel_backend

from preprocess import create_pos_neg_sequences_by_consecutive_labels
from models import get_transformer_classifier, get_mlp_classifier

# 这两个函数请确保在 filter_feature.py 中已经定义
# 如果想放在同文件，可直接将其实现贴在此处并保持引用一致
from filter_feature import select_top_n_features_tree, auto_select_features,filter_features
from models import time_aware_oversampling
def identity_transform(x):
    """空操作的特征选择器，用于保持接口一致。"""
    return x


def optimize_threshold(y_true, y_proba, metric='precision'):
    """
    根据给定评估指标（'precision', 'f1', 'recall', 'accuracy', 'mcc'）在 [0,1] 区间内寻找最佳分类阈值。
    
    参数：
        y_true (array-like): 真实标签（0 或 1）。
        y_proba (array-like): 预测的正类概率。
        metric (str): 评估指标，支持 'precision', 'f1', 'recall', 'accuracy', 'mcc'。
        
    返回：
        best_thresh (float): 使指定指标达到最佳的分类阈值。
    """
    best_thresh = 0.5
    best_score = -1
    for thresh in np.linspace(0, 1, 101):
        y_pred_temp = (y_proba > thresh).astype(int)
        if metric == 'precision':
            score = precision_score(y_true, y_pred_temp)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred_temp)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_temp)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred_temp)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred_temp)
        else:
            raise ValueError("metric must be one of 'precision', 'f1', 'recall', 'accuracy', 'mcc'")
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh


def train_model_for_label(
    df: pd.DataFrame,
    N: int,
    label_column: str,
    all_features: list,
    classifier_name: str,         # 'MLP' or 'Transformer'
    n_features_selected,          # int 或 'auto'
    window_size: int = 10,
    oversample_method: str = 'SMOTE',
    class_weight=None
):
    """
    对指定 df 的某个标签（label_column = 'Peak' 或 'Trough'）进行模型训练。
    当 n_features_selected='auto' 时，将在 [5, 10, 15, 20, 30] 范围内自动搜索最优特征数；
    当 n_features_selected 为整数时，使用随机森林保留前 n_features_selected 个最重要特征。
    最终返回 8 个值，供 train_model(...) 并行调用并返回 16 个。
    """

    print(f"\n=== 开始训练 {label_column} 模型 ===")

    # ---------- (1) 拿到数据 ----------
    data = df.copy()
    X = data[all_features]
    y = data[label_column].astype(np.int64)

    # ---------- (2) 简单相关性过滤 ----------
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"检测到高相关特征 {len(to_drop)} 个，将进行剔除。")
    else:
        print("未检测到高相关特征。")

    # 这里的 all_features_filtered 就是“仅相关性过滤后”的特征集合
    all_features_filtered = [f for f in all_features if f not in to_drop]

    # 先在 X 中保留 correlation 过滤后的列，并填充 NaN
    X = X[all_features_filtered].fillna(0)
    print(f"相关性过滤后特征数量: {len(all_features_filtered)}")

    # ---------- (3) 特征选择：自动 or 手动 ----------
    if n_features_selected == 'auto':
        # 自动搜索最优特征数 => 返回最优 n 的特征列表
        #top_feats = auto_select_features(
        #    X, y, n_candidates=[70], scoring='f1'
        #)
        #selected_features = top_feats
        selected_features = all_features_filtered
        print(f"[自动筛选] 实际保留 {len(selected_features)} 个特征。")
    elif isinstance(n_features_selected, int):
        # 手动指定 n_features_selected => 直接选前 n_features_selected 个最重要特征
        print(f"[手动指定] 使用随机森林保留前 {n_features_selected} 个特征...")
        #top_feats = select_top_n_features_tree(X, y, n_features_selected)
        top_feats = filter_features(X, y, method='pearson', n_features=n_features_selected)
       
        selected_features = top_feats
        print(f"[手动指定] 保留特征数量: {len(selected_features)}")
    else:
        # 如果既不是 'auto' 也不是 int，就直接用全部(仅相关性过滤后)
        selected_features = all_features_filtered
        print("n_features_selected 参数无效，直接使用相关性过滤后的特征。")

    # 在 X 中只保留最终的 selected_features
    X = X[selected_features].fillna(0)

    # ---------- (4) 标准化 ----------
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ---------- (5) 根据模型类型做数据处理 ----------
    if classifier_name == 'Transformer':
        print("构造 Transformer 时序数据集...")
        X_seq, y_seq = create_pos_neg_sequences_by_consecutive_labels(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, 
            random_state=42, stratify=y_seq
        )
        # 获取 Transformer
        model_name, model = get_transformer_classifier(
            num_features=X_train.shape[-1],
            window_size=window_size,
            class_weights=None
        )
        # 参数搜索网格
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [10]
        }
        scoring_used = 'precision'
    else:
        # 这里当成 MLP
        print("准备对 MLP 做过采样处理...")
        sampler = None
        if oversample_method == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif oversample_method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        elif oversample_method == 'Borderline-SMOTE':
            sampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
        elif oversample_method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=42)
        elif oversample_method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=42)
        elif oversample_method == 'Time-Aware':
        # 使用自定义时间感知过采样
            X_os, y_os = time_aware_oversampling(X_scaled, y, recency_weight=0.5, sequence_length=60)
        elif oversample_method in ['Class Weights', 'None']:
            sampler = None
        else:
            raise ValueError(f"未知过采样: {oversample_method}")

        # 是否过采样
        if sampler is not None:
            X_os, y_os = sampler.fit_resample(X_scaled, y)
        else:
            X_os, y_os = X_scaled, y

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_os, y_os, test_size=0.5,
            random_state=42, stratify=y_os
        )

        # 获取 MLP
        model_name, model = get_mlp_classifier(
            input_dim=X_train.shape[-1],
            class_weights=None
        )
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [20]
        }
        scoring_used = 'f1'

    # ---------- (6) 网格搜索 ----------
    print(f"开始网格搜索... scoring={scoring_used}")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring=scoring_used,
        verbose=0,
        error_score='raise'
    )
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}, 最佳得分: {grid_search.best_score_:.4f}")

    # ---------- (7) 评估 & 最佳阈值 ----------
    y_proba = best_estimator.predict_proba(X_test)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    else:
        # 如果是一维，需要手动 softmax
        y_proba = F.softmax(torch.tensor(y_proba), dim=1)[:, 1].numpy()

    best_thresh = optimize_threshold(y_test, y_proba, metric=scoring_used)
    y_pred = (y_proba > best_thresh).astype(int)

    # 计算指标
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_value = roc_auc_score(y_test, y_proba)

    print("\n=== 评估结果 ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {roc_value:.4f}, PR AUC: {pr_auc:.4f}, MCC: {mcc:.4f}")

    metrics = {
        'ROC AUC': roc_value,
        'PR AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc
    }

    # 注意：我们要返回 8 个值 以便和原先的解包完全对应
    # item 4 => peak_selected_features / trough_selected_features
    # item 5 => all_features_peak / all_features_trough
    return (
        best_estimator,               # (1) 训练完的模型
        scaler,                       # (2) 标准化器
        FunctionTransformer(func=identity_transform),  # (3) 空操作特征选择器
        selected_features,            # (4) 最终特征列表(经相关性+自动/手动)
        all_features_filtered,        # (5) 仅相关性过滤后的特征
        grid_search.best_score_,      # (6) 网格搜索CV得分
        metrics,                      # (7) 各种评估指标
        best_thresh                   # (8) 最佳阈值
    )


def train_model(
    df_preprocessed: pd.DataFrame,
    N: int,
    all_features: list,
    classifier_name: str,
    mixture_depth: int,
    n_features_selected,
    oversample_method: str,
    window_size: int = 10
):
    """
    分别训练高点 (Peak) 和低点 (Trough) 两个模型，支持自动/手动特征选择。
    返回总共16个值(每个标签8个)，与原 unpack 形式对应。
    """

    print("开始训练模型...")
    data = df_preprocessed.copy()

    labels = ['Peak', 'Trough']

    # 并行训练
    with parallel_backend('threading', n_jobs=-1):
        results = Parallel()(
            delayed(train_model_for_label)(
                data,
                N,
                label,  # 'Peak' or 'Trough'
                all_features,
                classifier_name,
                n_features_selected,
                window_size,
                oversample_method,
                class_weight='balanced' if oversample_method == 'Class Weights' else None
            )
            for label in labels
        )

    # results[0] => peak结果(8项), results[1] => trough结果(8项)
    peak_results = results[0]
    trough_results = results[1]

    # peak_results 解包（8项）
    (peak_model,
     peak_scaler,
     peak_selector,
     peak_selected_features,
     all_features_peak,
     peak_best_score,
     peak_metrics,
     peak_threshold) = peak_results

    # trough_results 解包（8项）
    (trough_model,
     trough_scaler,
     trough_selector,
     trough_selected_features,
     all_features_trough,
     trough_best_score,
     trough_metrics,
     trough_threshold) = trough_results

    # 返回 16 项
    return (
        peak_model,
        peak_scaler,
        peak_selector,
        peak_selected_features,
        all_features_peak,
        peak_best_score,
        peak_metrics,
        peak_threshold,

        trough_model,
        trough_scaler,
        trough_selector,
        trough_selected_features,
        all_features_trough,
        trough_best_score,
        trough_metrics,
        trough_threshold
    )

