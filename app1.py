#app.py
import streamlit as st 
from datetime import datetime
import pandas as pd
import numpy as np
import tushare as ts
import pickle
import io
from itertools import product
import streamlit.components.v1 as components
import copy
import math
import random
from sklearn.model_selection import train_test_split
import torch

from models import set_seed
from preprocess import preprocess_data, create_pos_neg_sequences_by_consecutive_labels
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick

# 设置随机种子
set_seed(42)

# 修改页面配置
st.set_page_config(
    page_title="东吴秀享AI超额收益系统", 
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------- 初始化 session_state -------------------- #
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = None

if 'peak_models_list' not in st.session_state:
    st.session_state.peak_models_list = []
if 'trough_models_list' not in st.session_state:
    st.session_state.trough_models_list = []

if 'train_df_preprocessed' not in st.session_state:
    st.session_state.train_df_preprocessed = None
if 'train_all_features' not in st.session_state:
    st.session_state.train_all_features = None

# 预测 / 回测 结果（未模型微调）
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'final_bt' not in st.session_state:
    st.session_state.final_bt = {}

# ★ 新增：模型微调后的预测 / 回测结果，用于对比
if 'inc_final_result' not in st.session_state:
    st.session_state.inc_final_result = None
if 'inc_final_bt' not in st.session_state:
    st.session_state.inc_final_bt = {}

# ★ 新增：存储预测集原始 DataFrame（模型微调后需要再次预测）
if 'new_df_raw' not in st.session_state:
    st.session_state.new_df_raw = None


def inject_orientation_script():
    orientation_script = """
    <style>
    #rotate-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        color: #fff;
        z-index: 9999;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 24px;
    }
    </style>
    <div id="rotate-overlay">
      请旋转手机至横屏模式使用
    </div>
    <script>
    function checkOrientation() {
        if (window.innerHeight > window.innerWidth) {
            document.getElementById('rotate-overlay').style.display = 'flex';
        } else {
            document.getElementById('rotate-overlay').style.display = 'none';
        }
    }
    window.addEventListener('resize', checkOrientation);
    checkOrientation();
    </script>
    """
    components.html(orientation_script, height=0)


def load_custom_css():
    custom_css = """
    <style>
    .strategy-row {
        margin-bottom: 8px;
        display: flex;
        flex-direction: row;
        align-items: center;
    }
    .strategy-label {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }
    @media only screen and (max-width: 768px) {
        .strategy-row {
            flex-direction: column;
            align-items: flex-start;
        }
        .strategy-label {
            justify-content: flex-start;
            margin-bottom: 4px;
        }
        .stPlotlyChart, .stDataFrame {
            width: 100% !important;
            overflow-x: auto;
        }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ========== 改进后的模型微调辅助函数 ========== #
def temporal_aware_sampling(old_df, sample_size, recency_weight=0.6):
    """
    进行时间感知采样，保留时间序列结构
    - old_df: 原始数据集
    - sample_size: 需要采样的样本数量
    - recency_weight: 时间权重因子，越大越偏向近期数据
    """
    if len(old_df) <= sample_size:
        return old_df  # 如果原始数据量不足，则全部返回
    
    # 计算时间权重（越近期的数据权重越大）
    weights = np.linspace(1-recency_weight, 1, len(old_df))
    
    # 采样时保留连续序列
    sampled_indices = []
    remaining = sample_size
    
    while len(sampled_indices) < sample_size and remaining > 0:
        # 确定当前序列长度
        seq_length = min(20, remaining)
        
        # 选择一个有效的起始点
        max_start = len(old_df) - seq_length
        if max_start <= 0:
            break
            
        # 根据权重选择起始点
        start_probs = weights[:max_start] / weights[:max_start].sum()
        start_idx = np.random.choice(max_start, p=start_probs)
        
        # 添加连续序列
        seq_indices = list(range(start_idx, start_idx + seq_length))
        sampled_indices.extend(seq_indices)
        remaining -= seq_length
    
    # 确保不超过请求的样本数
    sampled_indices = sampled_indices[:sample_size]
    
    return old_df.iloc[sampled_indices]


def get_lr_scheduler(current_epoch, total_epochs, base_lr):
    """
    学习率调度器：前10%预热，后面余弦衰减
    """
    warmup_epochs = max(int(total_epochs * 0.1), 1)
    
    if current_epoch < warmup_epochs:
        # 预热阶段，线性增加学习率
        return base_lr * (current_epoch / warmup_epochs)
    else:
        # 余弦衰减阶段
        decay_ratio = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return base_lr * (0.1 + 0.9 * cosine_decay)  # 从 base_lr 衰减到 0.1*base_lr


def enhanced_freeze_layers(model, classifier_name, num_frozen_layers=0):
    import copy
    import torch.nn as nn
    # 保存原始模型副本用于后续计算权重偏差
    original_model = copy.deepcopy(model)
    
    if classifier_name == 'MLP':
        if hasattr(model, 'module_'):
            # 当用户选择冻结1层或以上时，冻结 fc1
            if num_frozen_layers >= 1 and hasattr(model.module_, 'fc1'):
                for param in model.module_.fc1.parameters():
                    param.requires_grad = False
            # 当用户选择冻结2层时，再冻结 fc2
            if num_frozen_layers >= 2 and hasattr(model.module_, 'fc2'):
                for param in model.module_.fc2.parameters():
                    param.requires_grad = False
            # 如有需要，可选择冻结批归一化层（这里暂不冻结或根据需要修改）
            # for name, module in model.module_.named_modules():
            #     if isinstance(module, nn.BatchNorm1d):
            #         for param in module.parameters():
            #             param.requires_grad = False
    elif classifier_name == 'Transformer':
        if hasattr(model, 'module_') and hasattr(model.module_, 'transformer_encoder'):
            layers = model.module_.transformer_encoder.layers
            freeze_count = min(num_frozen_layers, len(layers))
            for i in range(freeze_count):
                for param in layers[i].parameters():
                    param.requires_grad = False
            # 也可以选择冻结位置编码和嵌入层
            if hasattr(model.module_, 'positional_encoding'):
                for param in model.module_.positional_encoding.parameters():
                    param.requires_grad = False
            if hasattr(model.module_, 'embedding'):
                for param in model.module_.embedding.parameters():
                    param.requires_grad = False
    return original_model



def incremental_train_for_label(model, scaler, selected_features, df_new, label_column, classifier_name, 
                                window_size=10, oversample_method=None, new_lr=None, new_epochs=5, 
                                freeze_layers=False, old_df=None, mix_ratio=1.0, progress_bar=None):
    """
    使用新数据对已有模型进行微调训练（防止灾难性遗忘的改进版本）：
      - 修复了 PyTorch 数据类型不匹配问题
      - 采用增强的冻结层策略
      - 时间感知采样保留时序结构
      - 学习率调度与早停机制
    """
    import numpy as np
    import torch
    
    # 1) 提取新数据并确保类型正确
    X_new = df_new[selected_features].fillna(0)
    X_new_scaled = scaler.transform(X_new).astype(np.float32)  # 确保X为float32
    y_new = df_new[label_column].astype(np.int64)  # 确保y为int64
    y_new = y_new.values

    # 2) 划分验证集，用于早停 - 确保类型一致
    try:
        from sklearn.model_selection import train_test_split
        X_new_train, X_new_val, y_new_train, y_new_val = train_test_split(
            X_new_scaled, y_new, test_size=0.2, random_state=42, 
            stratify=y_new if len(np.unique(y_new)) > 1 else None
        )
    except Exception as e:
        # 如果分割失败，使用全量数据
        X_new_train, y_new_train = X_new_scaled, y_new
        X_new_val, y_new_val = X_new_scaled[:10], y_new[:10]  # 创建一个小验证集
    
    # 3) 如果提供了旧数据，则进行混合训练（改进的时间感知采样）
    if old_df is not None and len(old_df) > 0:
        # 提高默认混合比例，防止过度关注新数据
        mix_ratio = max(mix_ratio, 0.5)
        sample_size = int(len(X_new) * mix_ratio)
        
        # 使用时间感知采样
        try:
            #old_sample_df = temporal_aware_sampling(old_df, sample_size, recency_weight=0.6)
            old_sample_df = old_df
            X_old_sample = old_sample_df[selected_features].fillna(0)
            y_old_sample = old_sample_df[label_column].astype(np.int64).values  # 确保类型为int64
            X_old_scaled = scaler.transform(X_old_sample).astype(np.float32)  # 确保类型为float32
            
            # 合并新旧数据
            X_combined = np.concatenate([X_new_train, X_old_scaled], axis=0)
            y_combined = np.concatenate([y_new_train, y_old_sample], axis=0)
        except Exception as e:
            # 如果混合失败，仅使用新数据
            X_combined = X_new_train
            y_combined = y_new_train
    else:
        X_combined = X_new_train
        y_combined = y_new_train
    
    # 4) 对于非 Transformer 模型，若需要过采样
    if classifier_name != 'Transformer' and oversample_method is not None and oversample_method not in ["Class Weights", "None"]:
        try:
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
            if sampler is not None:
                X_combined, y_combined = sampler.fit_resample(X_combined, y_combined)
                # 确保过采样后类型正确
                X_combined = X_combined.astype(np.float32)
                y_combined = y_combined.astype(np.int64)
        except Exception as e:
            # 如果过采样失败，继续使用原始数据
            pass

    # 5) 对于 Transformer 模型，将数据转换为时序数据
    if classifier_name == 'Transformer':
        try:
            # 直接创建 PyTorch 张量并指定类型
            X_tensor = torch.tensor(X_combined, dtype=torch.float32)
            y_tensor = torch.tensor(y_combined, dtype=torch.long)  # 关键：这里必须用 torch.long
            
            X_seq, y_seq = create_pos_neg_sequences_by_consecutive_labels(X_tensor, y_tensor)
            
            # 确保序列也是正确的 PyTorch 张量类型
            if not isinstance(X_seq, torch.Tensor):
                X_seq = torch.tensor(X_seq, dtype=torch.float32)
            if not isinstance(y_seq, torch.Tensor):
                y_seq = torch.tensor(y_seq, dtype=torch.long)  # 关键：这里必须用 torch.long
            else:
                y_seq = y_seq.to(torch.long)
            X_input, y_input = X_seq, y_seq
            
            # 对验证集也做相同处理
            X_val_tensor = torch.tensor(X_new_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_new_val, dtype=torch.long)  # 关键：这里必须用 torch.long
            
            X_val_seq, y_val_seq = create_pos_neg_sequences_by_consecutive_labels(X_val_tensor, y_val_tensor)
            
            if not isinstance(X_val_seq, torch.Tensor):
                X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32)
            if not isinstance(y_val_seq, torch.Tensor):
                y_val_seq = torch.tensor(y_val_seq, dtype=torch.long)  # 关键：这里必须用 torch.long
                
            X_val_input, y_val_input = X_val_seq, y_val_seq
        except Exception as e:
            # 如果序列创建失败，使用原始数据
            print(f"序列创建错误: {str(e)}")
            # 回退到直接使用 numpy 数组
            X_input, y_input = X_combined, y_combined
            X_val_input, y_val_input = X_new_val, y_new_val
    else:
        # 对于非 Transformer 模型直接使用 numpy 数组，但确保类型正确
        X_input = X_combined.astype(np.float32)
        y_input = y_combined.astype(np.int64)
        X_val_input = X_new_val.astype(np.float32)
        y_val_input = y_new_val.astype(np.int64)

    # 6) 增强的层冻结，防止灾难性遗忘
    original_model = None
    if freeze_layers:
        try:
            original_model = enhanced_freeze_layers(model, classifier_name)
        except Exception as e:
            # 如果冻结失败，继续不冻结
            print(f"层冻结错误: {str(e)}")

    # 7) 初始化早停所需的变量
    best_val_score = 0.0
    best_model_state = None
    patience = 5  # 早停耐心值
    patience_counter = 0
    
    # 8) 设置基础学习率
    base_lr = new_lr if new_lr is not None else 0.001
    
    # 9) 多epoch微调，同时更新进度条
    for epoch in range(new_epochs):
        # 动态调整学习率
        current_lr = get_lr_scheduler(epoch, new_epochs, base_lr)
        
        # 更新优化器学习率
        if hasattr(model, 'optimizer_') and model.optimizer_ is not None:
            for param_group in model.optimizer_.param_groups:
                param_group['lr'] = current_lr
        
        # 训练一个epoch
        try:
            # 确保标签类型正确
            if isinstance(y_input, np.ndarray):
                classes = np.array([0, 1], dtype=np.int64)
            else:  # 如果是 torch.Tensor
                classes = torch.tensor([0, 1], dtype=torch.long)
                
            model.partial_fit(X_input, y_input, classes=classes)
            
            # 在验证集上评估性能
            val_score = model.score(X_val_input, y_val_input)
            
            # 检查是否需要保存最佳模型
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                
                # 保存最佳模型状态
                if hasattr(model, 'module_'):
                    best_model_state = copy.deepcopy(model.module_.state_dict())
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= patience:
                # 提前停止训练
                if progress_bar is not None:
                    progress_bar.progress(1.0)  # 完成进度条
                
                # 恢复最佳模型
                if best_model_state is not None and hasattr(model, 'module_'):
                    model.module_.load_state_dict(best_model_state)
                break
        except Exception as e:
            # 如果验证失败，记录错误并继续
            print(f"训练/验证错误: {str(e)}")
            
        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / new_epochs)
    
    # 10) 如果训练完成但没有触发早停，仍然恢复最佳模型
    if best_model_state is not None and hasattr(model, 'module_'):
        try:
            model.module_.load_state_dict(best_model_state)
        except Exception as e:
            # 如果恢复失败，保持当前模型
            print(f"模型状态恢复错误: {str(e)}")
    
    return model

def main_product():
    inject_orientation_script()
    st.title("东吴秀享AI超额收益系统")

    # ========== 侧边栏参数设置 ========== 
    with st.sidebar:
        st.header("参数设置")
        with st.expander("数据设置", expanded=True):
            data_source = st.selectbox("选择数据来源", ["指数", "股票"])
            symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
            N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)
        with st.expander("模型设置", expanded=True):
            classifier_name = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
            if classifier_name == "深度学习":
                classifier_name = "MLP"
            mixture_depth = st.slider("因子混合深度", 1, 3, 1)
            oversample_method = st.selectbox(
                "类别不均衡处理", 
                ["过采样", "类别权重", 'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek',"时间感知过采样"]
            )
            if oversample_method == "过采样":
                oversample_method = "SMOTE"
            if oversample_method == "类别权重":
                oversample_method = "Class Weights"
            if oversample_method == "时间感知过采样":
                oversample_method = "Time-Aware"
            use_best_combo = True
        with st.expander("特征设置", expanded=True):
            auto_feature = st.checkbox("自动特征选择", True)
            n_features_selected = st.number_input(
                "选择特征数量", 
                min_value=5, max_value=100, value=20, 
                disabled=auto_feature
            )

    load_custom_css()

    # ========== 四个选项卡 ========== 
    tab1, tab2, tab3, tab4 = st.tabs(["训练模型", "预测", "模型微调", "上传模型预测"])

    # =======================================
    #    Tab1: 训练模型
    # =======================================
    with tab1:
        st.subheader("训练参数")
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("训练开始日期", datetime(2000, 1, 1))
        with col2:
            train_end = st.date_input("训练结束日期", datetime(2020, 12, 31))

        num_rounds = 10  # 这里写死为 10 轮
        if st.button("开始训练"):
            try:
                with st.spinner("数据预处理中..."):
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    
                    raw_data, all_features_train = preprocess_data(
                        raw_data, N, mixture_depth, mark_labels=True
                    )
                    df_preprocessed_train = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
                with st.spinner(f"开始多轮训练，共 {num_rounds} 次..."):
                    st.session_state.peak_models_list.clear()
                    st.session_state.trough_models_list.clear()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i in range(num_rounds):
                        progress_val = (i + 1) / num_rounds
                        status_text.text(f"正在训练第 {i+1}/{num_rounds} 个模型...")
                        progress_bar.progress(progress_val)

                        (peak_model, peak_scaler, peak_selector, peak_selected_features,
                         all_features_peak, peak_best_score, peak_metrics, peak_threshold,
                         trough_model, trough_scaler, trough_selector, trough_selected_features,
                         all_features_trough, trough_best_score, trough_metrics, trough_threshold
                        ) = train_model(
                            df_preprocessed_train,
                            N,
                            all_features_train,
                            classifier_name,
                            mixture_depth,
                            n_features_selected if not auto_feature else 'auto',
                            oversample_method
                        )
                        st.session_state.peak_models_list.append(
                            (peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
                        )
                        st.session_state.trough_models_list.append(
                            (trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
                        )

                    progress_bar.progress(1.0)
                    status_text.text("多轮训练完成！")

                # 将最后一次训练的模型存入 session_state
                st.session_state.models = {
                    'peak_model': peak_model,
                    'peak_scaler': peak_scaler,
                    'peak_selector': peak_selector,
                    'peak_selected_features': peak_selected_features,
                    'peak_threshold': peak_threshold,
                    'trough_model': trough_model,
                    'trough_scaler': trough_scaler,
                    'trough_selector': trough_selector,
                    'trough_selected_features': trough_selected_features,
                    'trough_threshold': trough_threshold,
                    'N': N,
                    'mixture_depth': mixture_depth
                }
                st.session_state.train_df_preprocessed = df_preprocessed_train
                st.session_state.train_all_features = all_features_train
                st.session_state.trained = True

                st.success(f"多轮训练全部完成！共训练 {num_rounds} 组峰/谷模型。")

                # 训练可视化
                peaks = df_preprocessed_train[df_preprocessed_train['Peak'] == 1]
                troughs = df_preprocessed_train[df_preprocessed_train['Trough'] == 1]
                fig = plot_candlestick(
                    df_preprocessed_train,
                    symbol_code,
                    train_start.strftime("%Y%m%d"),
                    train_end.strftime("%Y%m%d"),
                    peaks=peaks,
                    troughs=troughs
                )
                st.plotly_chart(fig, use_container_width=True, key="chart1")
            except Exception as e:
                st.error(f"训练失败: {str(e)}")

        # 训练集可视化（仅展示）
        try:
            st.markdown("<h2 style='font-size:20px;'>训练集可视化</h2>", unsafe_allow_html=True)
            symbol_type = 'index' if data_source == '指数' else 'stock'
            raw_data = read_day_from_tushare(symbol_code, symbol_type)
            
            raw_data, _ = preprocess_data(
                raw_data, N, mixture_depth, mark_labels=True
            )
            df_preprocessed_vis = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
            peaks_vis = df_preprocessed_vis[df_preprocessed_vis['Peak'] == 1]
            troughs_vis = df_preprocessed_vis[df_preprocessed_vis['Trough'] == 1]
            fig_vis = plot_candlestick(
                df_preprocessed_vis,
                symbol_code,
                train_start.strftime("%Y%m%d"),
                train_end.strftime("%Y%m%d"),
                peaks=peaks_vis,
                troughs=troughs_vis
            )
            st.plotly_chart(fig_vis, use_container_width=True, key="chart2")
        except Exception as e:
            st.warning(f"可视化失败: {e}")


    # =======================================
    #   Tab2: 预测 + 回测
    # =======================================
    with tab2:
        if not st.session_state.get('trained', False):
            st.warning("请先完成模型训练")
        else:
            st.subheader("预测参数")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start = st.date_input("预测开始日期", datetime(2021, 1, 1))
            with col_date2:
                pred_end = st.date_input("预测结束日期", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab2")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab2"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab2")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab2"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal = st.checkbox("调整买卖信号", value=False, help="阳线买，阴线卖，高点需创X日新高", key="enable_change_signal_tab2")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab2"
                    )

            if st.button("开始预测"):
                try:
                    if st.session_state.train_df_preprocessed is None or st.session_state.train_all_features is None:
                        st.error("无法获取训练集数据，请先在Tab1完成训练。")
                        return

                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data, _ = preprocess_data(
                        raw_data, N, mixture_depth, mark_labels=False
                    )
                    new_df_raw = select_time(raw_data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))

                    # ★ 存到 session_state，供模型微调使用
                    st.session_state.new_df_raw = new_df_raw

                    enable_chase_val = enable_chase
                    enable_stop_loss_val = enable_stop_loss
                    enable_change_signal_val = enable_change_signal
                    n_buy_val = n_buy
                    n_sell_val = n_sell
                    n_newhigh_val = n_newhigh

                    peak_models = st.session_state.peak_models_list
                    trough_models = st.session_state.trough_models_list

                    best_excess = -np.inf
                    best_models = None
                    final_result, final_bt, final_trades_df = None, {}, pd.DataFrame()
                    use_best_combo_val = use_best_combo

                    # ---------- 多组合搜索 ----------
                    if use_best_combo_val:
                        model_combinations = list(product(peak_models, trough_models))
                        total_combos = len(model_combinations)
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, (peak_m, trough_m) in enumerate(model_combinations):
                            combo_progress = (idx + 1) / total_combos
                            status_text.text(f"正在进行第 {idx+1}/{total_combos} 轮预测...")
                            progress_bar.progress(combo_progress)

                            pm, ps, psel, pfeats, pth = peak_m
                            tm, ts, tsel, tfeats, tth = trough_m
                            try:
                                _, bt_result, _ = predict_new_data(
                                    new_df_raw,
                                    pm, ps, psel, pfeats, pth,
                                    tm, ts, tsel, tfeats, tth,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=10,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
                                )
                                current_excess = bt_result.get('超额收益率', -np.inf)
                                if current_excess > best_excess:
                                    best_excess = current_excess
                                    best_models = {
                                        'peak_model': pm,
                                        'peak_scaler': ps,
                                        'peak_selector': psel,
                                        'peak_selected_features': pfeats,
                                        'peak_threshold': pth,
                                        'trough_model': tm,
                                        'trough_scaler': ts,
                                        'trough_selector': tsel,
                                        'trough_selected_features': tfeats,
                                        'trough_threshold': tth
                                    }
                            except:
                                continue

                        progress_bar.empty()
                        status_text.empty()

                        if best_models is None:
                            raise ValueError("所有组合均测试失败，无法完成预测。")

                        final_result, final_bt, final_trades_df = predict_new_data(
                            new_df_raw,
                            best_models['peak_model'],
                            best_models['peak_scaler'],
                            best_models['peak_selector'],
                            best_models['peak_selected_features'],
                            best_models['peak_threshold'],
                            best_models['trough_model'],
                            best_models['trough_scaler'],
                            best_models['trough_selector'],
                            best_models['trough_selected_features'],
                            best_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess * 100:.2f}%")

                    # ---------- 单模型预测 ----------
                    else:
                        single_models = st.session_state.models
                        _, bt_result_temp, _ = predict_new_data(
                            new_df_raw,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=True,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        best_excess = bt_result_temp.get('超额收益率', -np.inf)
                        final_result, final_bt, final_trades_df = predict_new_data(
                            new_df_raw,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！(单模型) 超额收益率: {best_excess*100:.2f}%")

                    # ---------- 显示回测结果 ----------
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',   '{:.4f}'.format(final_bt.get('年化夏普比率', 0)))
                    ]
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # ---------- 显示图表 ----------
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]
                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start.strftime("%Y%m%d"),
                        pred_end.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart3")

                    # ---------- 显示交易详情 ----------
                    col_left, col_right = st.columns(2)
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(
                                final_trades_df[['买入日', '买入原因', '买入价', '卖出日', '卖出原因', '卖出价', '持仓日', '盈亏']].style.format({'盈亏': '{:.2f}%'}))
                        else:
                            st.write("暂无交易记录")

                    # ---------- 保存到 session_state ----------
                    st.session_state.final_result = final_result
                    st.session_state.final_bt = final_bt
                    st.session_state.pred_start = pred_start
                    st.session_state.pred_end = pred_end
                    st.session_state.n_buy_val = n_buy_val
                    st.session_state.n_sell_val = n_sell_val
                    st.session_state.n_newhigh_val = n_newhigh_val
                    st.session_state.enable_chase_val = enable_chase_val
                    st.session_state.enable_stop_loss_val = enable_stop_loss_val
                    st.session_state.enable_change_signal_val = enable_change_signal_val

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")


    # =======================================
    #   Tab3: 模型微调 （新标签页）
    # =======================================
    with tab3:
        st.subheader("模型微调（微调已有模型）")
        if st.session_state.final_result is None or st.session_state.new_df_raw is None:
            st.warning("请先在 [预测] 标签页完成一次预测，才能进行模型微调。")
        else:
            # 1) 模型微调日期（默认与预测区间一致）
            inc_col1, inc_col2 = st.columns(2)
            with inc_col1:
                inc_start_date = st.date_input(
                    "模型微调起始日期", 
                    st.session_state.get('pred_start', datetime(2021,1,1))
                )
            with inc_col2:
                inc_end_date = st.date_input(
                    "模型微调结束日期", 
                    st.session_state.get('pred_end', datetime.now())
                )

            # 2) 学习率选择
            lr_dict = {"低 (1e-5)": 1e-5, "中 (1e-4)": 1e-4, "高 (1e-3)": 1e-3}
            lr_choice = st.selectbox("学习率", list(lr_dict.keys()), index=1)
            inc_lr = lr_dict[lr_choice]

            # 3) 训练轮数 (滑条 20-100)
            inc_epochs = st.slider("训练轮数", 20, 600, 40)

            # 4) 是否启用混合训练 & 旧数据与新数据比例
            mix_enabled = st.checkbox("启用混合训练", value=True)
            inc_mix_ratio = 0.5  # 默认提高到0.5，防止过度遗忘
            if mix_enabled:
                inc_mix_ratio = st.slider("旧数据与新数据比例", 0.5, 2.0, 0.5, step=0.1)

            # 5) 是否冻结部分层
            freeze_layers = st.checkbox("冻结部分层（保留原有知识）", value=True)
            # 增加冻结层数选择（根据模型类型设置不同上限）
            if freeze_layers:
                if classifier_name == "MLP":
                    frozen_layers_count = st.slider("选择冻结层数", min_value=0, max_value=2, value=1)
                elif classifier_name == "Transformer":
                    frozen_layers_count = st.slider("选择冻结层数", min_value=0, max_value=3, value=1)
            else:
                frozen_layers_count = 0

            # 点击按钮开始模型微调
            if st.button("执行模型微调"):
                try:
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_full = read_day_from_tushare(symbol_code, symbol_type)

                    # ① 获取全量数据 + 自动打标签
                    df_preprocessed_all, _ = preprocess_data(
                        raw_data_full,
                        N,
                        mixture_depth,
                        mark_labels=True
                    )

                    # ② 截取微调区间
                    add_df = select_time(
                        df_preprocessed_all,
                        inc_start_date.strftime("%Y%m%d"),
                        inc_end_date.strftime("%Y%m%d")
                    )

                    # ③ 分别对 peak_model / trough_model 做 partial_fit，并显示训练进度
                    st.write("正在对峰模型进行微调训练...")
                    peak_prog = st.progress(0)
                    updated_peak_model = incremental_train_for_label(
                        model=st.session_state.models['peak_model'],
                        scaler=st.session_state.models['peak_scaler'],
                        selected_features=st.session_state.models['peak_selected_features'],
                        df_new=add_df,  
                        label_column='Peak',
                        classifier_name=classifier_name,
                        window_size=10,
                        oversample_method=oversample_method,
                        new_lr=inc_lr,      
                        new_epochs=inc_epochs,      
                        freeze_layers=freeze_layers,
                        old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                        mix_ratio=inc_mix_ratio,
                        progress_bar=peak_prog
                    )

                    st.write("正在对谷模型进行微调训练...")
                    trough_prog = st.progress(0)
                    updated_trough_model = incremental_train_for_label(
                        model=st.session_state.models['trough_model'],
                        scaler=st.session_state.models['trough_scaler'],
                        selected_features=st.session_state.models['trough_selected_features'],
                        df_new=add_df,
                        label_column='Trough',
                        classifier_name=classifier_name,
                        window_size=10,
                        oversample_method=oversample_method,
                        new_lr=inc_lr,
                        new_epochs=inc_epochs,
                        freeze_layers=freeze_layers,
                        old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                        mix_ratio=inc_mix_ratio,
                        progress_bar=trough_prog
                    )


                    # ④ 更新 session_state 中的模型
                    st.session_state.models['peak_model'] = updated_peak_model
                    st.session_state.models['trough_model'] = updated_trough_model

                    st.success("模型微调完成！下面对比微调前后的回测结果...")

                    # ⑤ 用微调后的模型再次预测 (针对 "预测" 时保存下来的 new_df_raw)
                    #    保持原先预测时的策略参数
                    refreshed_new_df = st.session_state.new_df_raw
                    if refreshed_new_df is None:
                        st.warning("未发现预测集数据，请先完成预测再查看对比结果。")
                        return

                    # ---- 重跑预测，得到模型微调后的结果 ----
                    if use_best_combo:
                        # 如果之前是多组合策略，这里同样把最优阈值、特征等拿来
                        best_models_inc = {
                            'peak_model': st.session_state.models['peak_model'],
                            'peak_scaler': st.session_state.models['peak_scaler'],
                            'peak_selector': st.session_state.models['peak_selector'],
                            'peak_selected_features': st.session_state.models['peak_selected_features'],
                            'peak_threshold': st.session_state.models['peak_threshold'],
                            'trough_model': st.session_state.models['trough_model'],
                            'trough_scaler': st.session_state.models['trough_scaler'],
                            'trough_selector': st.session_state.models['trough_selector'],
                            'trough_selected_features': st.session_state.models['trough_selected_features'],
                            'trough_threshold': st.session_state.models['trough_threshold']
                        }
                        inc_final_result, inc_final_bt, inc_final_trades_df = predict_new_data(
                            refreshed_new_df,
                            best_models_inc['peak_model'],
                            best_models_inc['peak_scaler'],
                            best_models_inc['peak_selector'],
                            best_models_inc['peak_selected_features'],
                            best_models_inc['peak_threshold'],
                            best_models_inc['trough_model'],
                            best_models_inc['trough_scaler'],
                            best_models_inc['trough_selector'],
                            best_models_inc['trough_selected_features'],
                            best_models_inc['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=st.session_state.n_buy_val,
                            N_sell=st.session_state.n_sell_val,
                            N_newhigh=st.session_state.n_newhigh_val,
                            enable_chase=st.session_state.enable_chase_val,
                            enable_stop_loss=st.session_state.enable_stop_loss_val,
                            enable_change_signal=st.session_state.enable_change_signal_val,
                        )
                    else:
                        single_models = st.session_state.models
                        inc_final_result, inc_final_bt, inc_final_trades_df = predict_new_data(
                            refreshed_new_df,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=st.session_state.n_buy_val,
                            N_sell=st.session_state.n_sell_val,
                            N_newhigh=st.session_state.n_newhigh_val,
                            enable_chase=st.session_state.enable_chase_val,
                            enable_stop_loss=st.session_state.enable_stop_loss_val,
                            enable_change_signal=st.session_state.enable_change_signal_val,
                        )

                    # ---- 保存微调结果到 session_state ----
                    st.session_state.inc_final_result = inc_final_result
                    st.session_state.inc_final_bt = inc_final_bt

                    # ---- 对比模型微调前后的回测 ----
                    st.markdown("### 对比：未模型微调 vs 模型微调后")
                    orig_bt = st.session_state.final_bt
                    inc_bt = st.session_state.inc_final_bt

                    # 原模型 vs 新模型 主要指标对比
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.write("**未模型微调回测**")
                        st.metric("累计收益率", f"{orig_bt.get('累计收益率', 0)*100:.2f}%")
                        st.metric("超额收益率", f"{orig_bt.get('超额收益率', 0)*100:.2f}%")
                        st.metric("胜率", f"{orig_bt.get('胜率', 0)*100:.2f}%")
                        st.metric("最大回撤", f"{orig_bt.get('最大回撤', 0)*100:.2f}%")
                        st.metric("交易笔数", f"{orig_bt.get('交易笔数', 0)}")
                    with col_after:
                        st.write("**模型微调后回测**")
                        st.metric("累计收益率", f"{inc_bt.get('累计收益率', 0)*100:.2f}%")
                        st.metric("超额收益率", f"{inc_bt.get('超额收益率', 0)*100:.2f}%")
                        st.metric("胜率", f"{inc_bt.get('胜率', 0)*100:.2f}%")
                        st.metric("最大回撤", f"{inc_bt.get('最大回撤', 0)*100:.2f}%")
                        st.metric("交易笔数", f"{inc_bt.get('交易笔数', 0)}")

                    st.subheader("模型微调后图表")
                    peaks_pred_inc = inc_final_result[inc_final_result['Peak_Prediction'] == 1]
                    troughs_pred_inc = inc_final_result[inc_final_result['Trough_Prediction'] == 1]
                    fig_updated = plot_candlestick(
                        inc_final_result,
                        symbol_code,
                        st.session_state.pred_start.strftime("%Y%m%d"),
                        st.session_state.pred_end.strftime("%Y%m%d"),
                        peaks_pred_inc,
                        troughs_pred_inc,
                        prediction=True
                    )
                    st.plotly_chart(fig_updated, use_container_width=True, key="chart_updated")

                except Exception as e:
                    st.error(f"模型微调过程中出错: {e}")


    # =======================================
    #   Tab4: 上传模型文件，独立预测
    # =======================================
    with tab4:
        st.subheader("上传模型文件（.pkl）并预测")
        st.markdown("在此页面可以上传之前已保存的最佳模型或单模型文件，直接进行预测。")
        uploaded_file = st.file_uploader("选择本地模型文件进行预测：", type=["pkl"])
        if uploaded_file is not None:
            with st.spinner("正在加载模型..."):
                best_models_loaded = pickle.load(uploaded_file)
                st.session_state.best_models = best_models_loaded
                st.session_state.trained = True
            st.success("已成功加载本地模型，可进行预测！")

        if not st.session_state.trained or (st.session_state.best_models is None):
            st.warning("请先上传模型文件，或前往其他页面训练并保存模型。")
        else:
            st.markdown("### 预测参数")
            col_date1_up, col_date2_up = st.columns(2)
            with col_date1_up:
                pred_start_up = st.date_input("预测开始日期(上传模型Tab)", datetime(2021, 1, 1))
            with col_date2_up:
                pred_end_up = st.date_input("预测结束日期(上传模型Tab)", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase_up = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab4")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase_up),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab4"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss_up = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab4")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss_up),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab4"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal_up = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab4")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal_up),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab4"
                    )

            # --------- 上传模型后的预测 --------- 
            if st.button("开始预测(上传模型Tab)"):
                try:
                    best_models = st.session_state.best_models
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_up = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data_up, _ = preprocess_data(
                        raw_data_up, N, mixture_depth, mark_labels=False
                    )
                    new_df_up = select_time(raw_data_up, pred_start_up.strftime("%Y%m%d"), pred_end_up.strftime("%Y%m%d"))

                    # 取出 N, mixture_depth，若之前训练时保存了这两个信息，可以直接读
                    N_val = st.session_state.models.get('N', 30)
                    mixture_val = st.session_state.models.get('mixture_depth', 1)

                    final_result_up, final_bt_up, final_trades_df_up = predict_new_data(
                        new_df_up,
                        best_models['peak_model'],
                        best_models['peak_scaler'],
                        best_models['peak_selector'],
                        best_models['peak_selected_features'],
                        best_models['peak_threshold'],
                        best_models['trough_model'],
                        best_models['trough_scaler'],
                        best_models['trough_selector'],
                        best_models['trough_selected_features'],
                        best_models['trough_threshold'],
                        N_val,
                        mixture_val,
                        window_size=10,
                        eval_mode=False,
                        N_buy=n_buy_up,
                        N_sell=n_sell_up,
                        N_newhigh=n_newhigh_up,
                        enable_chase=enable_chase_up,
                        enable_stop_loss=enable_stop_loss_up,
                        enable_change_signal=enable_change_signal_up,
                    )
                    st.success("预测完成！（使用已上传模型）")

                    st.subheader("回测结果")
                    metrics_up = [
                        ('累计收益率',   final_bt_up.get('累计收益率', 0)),
                        ('超额收益率',   final_bt_up.get('超额收益率', 0)),
                        ('胜率',         final_bt_up.get('胜率', 0)),
                        ('交易笔数',     final_bt_up.get('交易笔数', 0)),
                        ('最大回撤',     final_bt_up.get('最大回撤', 0)),
                        ('夏普比率',     final_bt_up.get('年化夏普比率', 0)),
                    ]
                    first_line_up = metrics_up[:3]
                    cols_1_up = st.columns(3)
                    for col, (name, value) in zip(cols_1_up, first_line_up):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")
                    second_line_up = metrics_up[3:]
                    cols_2_up = st.columns(3)
                    for col, (name, value) in zip(cols_2_up, second_line_up):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    peaks_pred_up = final_result_up[final_result_up['Peak_Prediction'] == 1]
                    troughs_pred_up = final_result_up[final_result_up['Trough_Prediction'] == 1]
                    fig_up = plot_candlestick(
                        final_result_up,
                        symbol_code,
                        pred_start_up.strftime("%Y%m%d"),
                        pred_end_up.strftime("%Y%m%d"),
                        peaks_pred_up,
                        troughs_pred_up,
                        prediction=True
                    )
                    st.plotly_chart(fig_up, use_container_width=True, key="chart_upload_tab")

                    col_left_up, col_right_up = st.columns(2)
                    final_result_up = final_result_up.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left_up:
                        st.subheader("预测明细")
                        st.dataframe(final_result_up[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df_up = final_trades_df_up.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df_up.empty:
                        final_trades_df_up['盈亏'] = final_trades_df_up['盈亏'] * 100
                        final_trades_df_up['买入日'] = final_trades_df_up['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df_up['卖出日'] = final_trades_df_up['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right_up:
                        st.subheader("交易记录")
                        if not final_trades_df_up.empty:
                            st.dataframe(
                                final_trades_df_up[['买入日', '买入原因', '买入价', '卖出日', '卖出原因', '卖出价', '持仓日', '盈亏']].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

if __name__ == "__main__":
    main_product()