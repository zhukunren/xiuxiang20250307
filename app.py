import streamlit as st 
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from itertools import product
import torch
import os
from models import set_seed
from preprocess import preprocess_data
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick
from incremental_train import incremental_train_for_label
from CSS import inject_orientation_script, load_custom_css
import re
import json
import time
import plotly.express as px
from tab5_function import load_user_factor_map, save_user_factor_map, save_user_function, \
    get_generated_code_cached, register_user_factor, my_factors

# 设置随机种子
set_seed(42)
# ---------------- 常量 & 全局设置 ----------------
USER_FACTOR_MAP_FILE = "user_factor_map.json"
BASE_DIR = "user_functions"
USER_ID = "user_123"  # 实际项目应替换为真实用户ID
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# 加载系统因子字典
with open("system_factor.json", 'r', encoding='utf-8') as f:
    system_factor_dict = json.load(f)

# 加载用户因子组合列表，如果文件不存在或为空则创建空白文件
selections_filename = "user_factor_selections.json"
if os.path.exists(selections_filename):
    try:
        if os.path.getsize(selections_filename) > 0:
            with open(selections_filename, 'r', encoding='utf-8') as f:
                factor_selection_dict = json.load(f)
        else:
            factor_selection_dict = {}
            with open(selections_filename, 'w', encoding='utf-8') as f:
                json.dump(factor_selection_dict, f, indent=4, ensure_ascii=False)
    except json.JSONDecodeError:
        factor_selection_dict = {}
        with open(selections_filename, 'w', encoding='utf-8') as f:
            json.dump(factor_selection_dict, f, indent=4, ensure_ascii=False)
else:
    factor_selection_dict = {}
    with open(selections_filename, 'w', encoding='utf-8') as f:
        json.dump(factor_selection_dict, f, indent=4, ensure_ascii=False)



if 'user_factor_map' not in st.session_state:
    st.session_state.user_factor_map = load_user_factor_map(USER_FACTOR_MAP_FILE)
    user_factor_map = load_user_factor_map(USER_FACTOR_MAP_FILE)
if USER_ID not in st.session_state.user_factor_map:
    st.session_state.user_factor_map[USER_ID] = {}
    user_factor_map = load_user_factor_map(USER_FACTOR_MAP_FILE)

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

# 模型微调后的预测 / 回测结果
if 'inc_final_result' not in st.session_state:
    st.session_state.inc_final_result = None
if 'inc_final_bt' not in st.session_state:
    st.session_state.inc_final_bt = {}

# 存储预测集原始 DataFrame（模型微调后需要再次预测）
if 'new_df_raw' not in st.session_state:
    st.session_state.new_df_raw = None


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
                ["过采样", "类别权重", 'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek', "时间感知过采样"]
            )
            if oversample_method == "过采样":
                oversample_method = "SMOTE"
            if oversample_method == "类别权重":
                oversample_method = "Class Weights"
            if oversample_method == "时间感知过采样":
                oversample_method = "Time-Aware"
            use_best_combo = True
        with st.expander("因子设置", expanded=True):
            #选择用户自定义因子组合列表
            user_factor_select_dic = factor_selection_dict.get(USER_ID, {})
            if user_factor_select_dic:
                # 取出所有保存的列表名称（第二层键）
                selection_names = list(user_factor_select_dic.keys())
                selected_factor_list = st.selectbox("请选择因子列表", selection_names)
                unselected_system = user_factor_select_dic[selected_factor_list].get('unselected_system', [])
                selected_system = user_factor_select_dic[selected_factor_list].get('selected_system', [])
                selected_func_names = user_factor_select_dic[selected_factor_list].get('selected_custom', [])
            #因子数量选择
            auto_feature = st.checkbox("自动因子选择", True)
            n_features_selected = st.number_input(
                "选择特征数量", 
                min_value=5, max_value=100, value=20, 
                disabled=auto_feature
            )

    load_custom_css()

    # ========== 四个选项卡 ========== 
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["训练模型", "预测",  "我的模型", "模型微调", "因子研究"])

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

        num_rounds = 2  # 这里写死为 10 轮

        # 只有在点击“开始训练”后才进行数据获取、打标和训练
        if st.button("开始训练"):
            try:
                with st.spinner("数据预处理中..."):
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    
                    raw_data, all_features_train = preprocess_data(
                        raw_data, N, mixture_depth, mark_labels=True, selected_system=selected_system, selected_func_names=selected_func_names,
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

                        (
                            peak_model, peak_scaler, peak_selector, peak_selected_features,
                            peak_threshold, trough_model, trough_scaler, trough_selector,
                            trough_selected_features, trough_threshold
                        ) = (None,) * 10

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

                # 训练数据可视化，仅在训练结束后由用户触发即可
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
                        "追涨长度",
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
                        "止损长度",
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
                        "高点需创X日新高",
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
                        raw_data, N, mixture_depth, mark_labels=False, selected_system=selected_system, selected_func_names=selected_func_names,
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
                        from itertools import product
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
                        
                        # 将最佳模型组合保存到 session_state
                        st.session_state.best_models = best_models

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
                            col.metric(name, f"{float(value)*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{float(value)*100:.2f}%")
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
                                final_trades_df[['买入日', '买入原因', '买入价', '卖出日', '卖出原因', '卖出价', '持仓日', '盈亏']]\
                                    .style.format({'盈亏': '{:.2f}%'}))
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
            # ========= 新增：保存当前模型 ==========
            # ---------- 保存当前模型 ========== 
            st.markdown("### 保存当前模型")
            model_name_save = st.text_input("请输入模型名称", key="model_name_save")

            # 检查是否点击了保存按钮
            button_state = st.button("保存模型", key="save_model_button")
            if button_state:
                # 检查模型名称是否为空
                if not model_name_save.strip():
                    st.error("请输入有效的模型名称！")
                else:
                    # 优先保存 best_models，否则保存训练得到的 models
                    if st.session_state.get("best_models") is not None:
                        # 创建副本并添加数据源信息
                        model_to_save = st.session_state.best_models.copy()
                        model_to_save['data_source'] = data_source  # 数据信息源
                        model_to_save['symbol_code'] = symbol_code  # 代码
                        model_to_save['symbol_type'] = symbol_type  # 代码类型
                        model_to_save['N'] = N #窗口长度
                        model_to_save['model_type'] = classifier_name #模型类型
                        model_to_save['model_name'] = model_name_save #模型名称
                        model_to_save['strat_date'] = train_start #训练开始日期
                        model_to_save['end_date'] = train_end #训练结束日期
               
                    else:
                        st.error("当前无可保存的模型，请先进行预测验证模型有效性")
                        model_to_save = None

                    if model_to_save is not None:
                        # Define the directory for the user model to be saved in
                        user_model_dir = os.path.join("user_models", USER_ID)
                        #user_model_dir = os.path.expanduser(f"~/user_models/{USER_ID}")
                        #os.makedirs(user_model_dir, exist_ok=True)
                        
                        # Ensure the user model directory exists, if not, create it
                        if not os.path.exists(user_model_dir):
                            try:
                                os.makedirs(user_model_dir)  # This creates the directory if it doesn't exist
                                st.success(f"目录 {user_model_dir} 已创建")
                            except Exception as e:
                                st.error(f"创建目录失败: {str(e)}")
                                return  # Stop further execution if directory creation fails
                        
                        # Define the full path for the model to be saved
                        save_path = os.path.join(user_model_dir, f"{model_name_save}.pkl")
                        
                        try:
                            # Save the model using pickle
                            with open(save_path, "wb") as f:
                                pickle.dump(model_to_save, f)
                            st.success(f"模型已保存为 {save_path}")
                        except Exception as e:
                            st.error(f"保存模型失败: {str(e)}")


                

    # =======================================
    #   Tab3: 我的模型 —— 修改为从保存的模型列表中选择加载
    # =======================================
    with tab3:
        st.subheader("选择已保存的模型进行预测")
        # 确保 user_models/{USER_ID} 目录存在
        user_model_dir = os.path.join("user_models", USER_ID)

        # 如果目录不存在，则创建目录
        if not os.path.exists(user_model_dir):
            os.makedirs(user_model_dir)  # 创建目录
            st.success(f"目录 {user_model_dir} 已创建")  # 可选：通知用户目录已创建
        saved_files = [f for f in os.listdir(os.path.join("user_models", USER_ID)) if f.endswith(".pkl")]
        if saved_files:
            selected_model_file = st.selectbox("请选择一个模型", saved_files)
            if st.button("加载模型", key="load_saved_model"):
                model_path = os.path.join("user_models", USER_ID, selected_model_file)
                with st.spinner("正在加载模型..."):
                    with open(model_path, "rb") as f:
                        loaded_model = pickle.load(f)
                    st.session_state.best_models = loaded_model
                    st.session_state.trained = True
                    
                    # 新增：获取保存的数据源信息
                    data_source_loaded = loaded_model.get('data_source', '未知')
                    symbol_code_loaded = loaded_model.get('symbol_code', '未知')
                    
                st.success(f"已加载模型 {selected_model_file}")
                # 新增：显示来源信息
                st.info(f"数据来源：{data_source_loaded} | 标的代码：{symbol_code_loaded}")
            # 下面可以保留预测参数和预测代码，复用 Tab2 中的逻辑
            st.markdown("### 预测参数（使用已加载模型）")
            col_date1_up, col_date2_up = st.columns(2)
            with col_date1_up:
                pred_start_up = st.date_input("预测开始日期(已加载模型)", datetime(2021, 1, 1))
            with col_date2_up:
                pred_end_up = st.date_input("预测结束日期(已加载模型)", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase_up = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab3")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy_up = st.number_input(
                        "追涨长度",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase_up),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab3"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss_up = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab3")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell_up = st.number_input(
                        "止损长度",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss_up),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab3"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal_up = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab3")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh_up = st.number_input(
                        "高点需创X日新高",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal_up),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab3"
                    )
            if st.button("开始预测(已加载模型)"):
                try:
                    best_models = st.session_state.best_models
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_up = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data_up, _ = preprocess_data(
                        raw_data_up, N, mixture_depth, mark_labels=False, selected_system=selected_system, selected_func_names=selected_func_names,
                    )
                    new_df_up = select_time(raw_data_up, pred_start_up.strftime("%Y%m%d"), pred_end_up.strftime("%Y%m%d"))

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
                    st.success("预测完成！（使用已加载模型）")

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
                                final_trades_df_up[['买入日', '买入原因', '买入价', '卖出日', '卖出原因', '卖出价', '持仓日', '盈亏']]\
                                    .style.format({'盈亏': '{:.2f}%'}))
                        else:
                            st.write("暂无交易记录")
                except Exception as e: 
                    print("我的模型预测失败", e)

        else:
            st.info("请先保存或加载模型，再进行预测。")

    # =======================================
    #   Tab4: 模型微调
    # =======================================
    with tab4:
        st.subheader("模型微调（微调已有模型）")

        # 如果没有选择模型或者没有预测结果，提示用户
        if 'final_result' not in st.session_state or 'new_df_raw' not in st.session_state:
            st.warning("请先在 [预测] 标签页完成一次预测并保存模型，才能进行模型微调。")
        else:
            # 让用户选择微调的模型
            st.markdown("### 选择初始模型进行微调")

            # 获取保存的模型列表
            user_model_dir = os.path.join("user_models", USER_ID)
            if not os.path.exists(user_model_dir):
                os.makedirs(user_model_dir)  # 如果目录不存在，创建目录
            saved_files = [f for f in os.listdir(user_model_dir) if f.endswith(".pkl")]
            
            # 如果有保存的模型，允许用户选择
            if saved_files:
                selected_model_file = st.selectbox("请选择初始模型", saved_files)
                if st.button("加载模型", key="load_initial_model"):
                    model_path = os.path.join(user_model_dir, selected_model_file)
                    with st.spinner("正在加载初始模型..."):
                        with open(model_path, "rb") as f:
                            loaded_model = pickle.load(f)
                        st.session_state.best_models = loaded_model  # 加载并保存到 session_state
                        st.session_state.trained = True  # 设置为已训练
                    st.success(f"已加载初始模型 {selected_model_file}")
            else:
                st.info("没有可用的保存模型，请先保存一个模型再进行微调。")
            
            # 微调相关参数设置
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

            lr_dict = {"极低 (1e-6)": 1e-6, "低 (1e-5)": 1e-5, "中 (1e-4)": 1e-4, "高 (1e-3)": 1e-3}
            lr_choice = st.selectbox("学习率", list(lr_dict.keys()), index=1)
            inc_lr = lr_dict[lr_choice]

            inc_epochs = st.slider("最大训练轮数", 5, 100, 20)

            # 冻结层选项
            if classifier_name == "MLP":
                freeze_options = {
                    "不冻结任何层": "none",
                    "只冻结第一层 (fc1)": "first_layer",
                    "只冻结第二层 (fc2)": "second_layer", 
                    "冻结所有层": "all",
                    "部分冻结第一层": "partial"
                }
            else:  # Transformer
                freeze_options = {
                    "不冻结任何层": "none",
                    "冻结输入层": "first_layer",
                    "冻结编码器层 (除最后一层)": "encoder_layers",
                    "冻结输出层": "output_layer",
                    "冻结所有层": "all"
                }
            freeze_choice = st.selectbox("冻结策略", list(freeze_options.keys()), index=0)
            freeze_option = freeze_options[freeze_choice]

            mix_enabled = st.checkbox("启用混合训练", value=True)
            inc_mix_ratio = 0.2
            if mix_enabled:
                inc_mix_ratio = st.slider("旧数据与新数据比例", 0.1, 2.0, 0.2, step=0.1)

            early_stopping = st.checkbox("启用早停", value=True)
            col_val1, col_val2 = st.columns(2)
            with col_val1:
                val_size = st.slider("验证集比例", 0.1, 0.5, 0.2, step=0.05, 
                                    disabled=not early_stopping)
            with col_val2:
                patience = st.slider("早停耐心值", 1, 10, 3, step=1,
                                    disabled=not early_stopping)

            if st.button("执行模型微调"):
                try:
                    # 微调操作，使用从 session_state 加载的 best_models
                    if 'best_models' not in st.session_state:
                        st.error("请加载一个已训练的模型进行微调。")
                        return

                    # 从加载的模型中获取峰谷模型
                    peak_model = st.session_state.best_models.get('peak_model')
                    trough_model = st.session_state.best_models.get('trough_model')

                    if peak_model is None or trough_model is None:
                        st.error("加载的模型不包含峰/谷模型，无法进行微调。")
                        return

                    # 其他训练和微调步骤
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_full = read_day_from_tushare(symbol_code, symbol_type)

                    df_preprocessed_all, _ = preprocess_data(
                        raw_data_full,
                        N,
                        mixture_depth,
                        mark_labels=True,
                        selected_system=selected_system,
                        selected_func_names=selected_func_names,
                    )

                    add_df = select_time(
                        df_preprocessed_all,
                        inc_start_date.strftime("%Y%m%d"),
                        inc_end_date.strftime("%Y%m%d")
                    )

                    st.write("正在对峰模型进行微调训练...")
                    peak_prog = st.progress(0)

                    from incremental_train import incremental_train_for_label
                    updated_peak_model, peak_val_acc, peak_epochs = incremental_train_for_label(
                        model=peak_model,  # 使用从 best_models 加载的峰模型
                        scaler=st.session_state.best_models.get('peak_scaler'),
                        selected_features=st.session_state.best_models.get('peak_selected_features'),
                        df_new=add_df,  
                        label_column='Peak',
                        classifier_name=classifier_name,
                        window_size=10,
                        oversample_method=oversample_method,
                        new_lr=inc_lr,      
                        new_epochs=inc_epochs,      
                        freeze_option=freeze_option,
                        old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                        mix_ratio=inc_mix_ratio,
                        progress_bar=peak_prog,
                        early_stopping=early_stopping,
                        val_size=val_size,
                        patience=patience
                    )
                    st.success(f"峰模型微调完成! 最佳验证准确率: {peak_val_acc:.4f}，实际训练轮数: {peak_epochs}/{inc_epochs}")

                    st.write("正在对谷模型进行微调训练...")
                    trough_prog = st.progress(0)

                    updated_trough_model, trough_val_acc, trough_epochs = incremental_train_for_label(
                        model=trough_model,  # 使用从 best_models 加载的谷模型
                        scaler=st.session_state.best_models.get('trough_scaler'),
                        selected_features=st.session_state.best_models.get('trough_selected_features'),
                        df_new=add_df,
                        label_column='Trough',
                        classifier_name=classifier_name,
                        window_size=10,
                        oversample_method=oversample_method,
                        new_lr=inc_lr,
                        new_epochs=inc_epochs,
                        freeze_option=freeze_option,
                        old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                        mix_ratio=inc_mix_ratio,
                        progress_bar=trough_prog,
                        early_stopping=early_stopping,
                        val_size=val_size,
                        patience=patience
                    )
                    st.success(f"谷模型微调完成! 最佳验证准确率: {trough_val_acc:.4f}，实际训练轮数: {trough_epochs}/{inc_epochs}")

                    # 更新微调后的模型
                    st.session_state.best_models['peak_model'] = updated_peak_model
                    st.session_state.best_models['trough_model'] = updated_trough_model

                    # 记录微调参数
                    st.session_state.finetune_params = {
                        'lr': inc_lr,
                        'epochs': inc_epochs,
                        'freeze_option': freeze_option,
                        'mix_ratio': inc_mix_ratio if mix_enabled else 0,
                        'peak_val_acc': peak_val_acc,
                        'peak_epochs': peak_epochs,
                        'trough_val_acc': trough_val_acc,
                        'trough_epochs': trough_epochs
                    }

                    st.info("模型微调完成！下面对比微调前后的回测结果...")

                    # 回测等后续操作...
                    refreshed_new_df = st.session_state.new_df_raw
                    if refreshed_new_df is None:
                        st.warning("未发现预测集数据，请先完成预测再查看对比结果。")
                        return

                    # 比较微调前后的回测结果
                    if use_best_combo:
                        best_models_inc = {
                            'peak_model': updated_peak_model,
                            'peak_scaler': st.session_state.best_models.get('peak_scaler'),
                            'peak_selector': st.session_state.best_models.get('peak_selector'),
                            'peak_selected_features': st.session_state.best_models.get('peak_selected_features'),
                            'peak_threshold': st.session_state.best_models.get('peak_threshold'),
                            'trough_model': updated_trough_model,
                            'trough_scaler': st.session_state.best_models.get('trough_scaler'),
                            'trough_selector': st.session_state.best_models.get('trough_selector'),
                            'trough_selected_features': st.session_state.best_models.get('trough_selected_features'),
                            'trough_threshold': st.session_state.best_models.get('trough_threshold')
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
                        single_models = st.session_state.best_models
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

                    # 更新回测结果
                    st.session_state.inc_final_result = inc_final_result
                    st.session_state.inc_final_bt = inc_final_bt

                    # 显示回测结果
                    st.markdown("### 对比：未模型微调 vs 模型微调后")
                    orig_bt = st.session_state.final_bt
                    inc_bt = st.session_state.inc_final_bt

                    col_before, col_after, col_diff = st.columns(3)
                    with col_before:
                        st.write("**微调前**")
                        st.metric("累计收益率", f"{orig_bt.get('累计收益率', 0)*100:.2f}%")
                        st.metric("超额收益率", f"{orig_bt.get('超额收益率', 0)*100:.2f}%")
                        st.metric("胜率", f"{orig_bt.get('胜率', 0)*100:.2f}%")
                        st.metric("最大回撤", f"{orig_bt.get('最大回撤', 0)*100:.2f}%")
                        st.metric("交易笔数", f"{orig_bt.get('交易笔数', 0)}")
                    
                    with col_after:
                        st.write("**微调后**")
                        st.metric("累计收益率", f"{inc_bt.get('累计收益率', 0)*100:.2f}%")
                        st.metric("超额收益率", f"{inc_bt.get('超额收益率', 0)*100:.2f}%")
                        st.metric("胜率", f"{inc_bt.get('胜率', 0)*100:.2f}%")
                        st.metric("最大回撤", f"{inc_bt.get('最大回撤', 0)*100:.2f}%")
                        st.metric("交易笔数", f"{inc_bt.get('交易笔数', 0)}")
                    
                    with col_diff:
                        st.write("**变化量**")
                        st.metric("累计收益率变化", 
                                f"{(inc_bt.get('累计收益率', 0) - orig_bt.get('累计收益率', 0))*100:.2f}%",
                                delta_color="normal")
                        st.metric("超额收益率变化", 
                                f"{(inc_bt.get('超额收益率', 0) - orig_bt.get('超额收益率', 0))*100:.2f}%",
                                delta_color="normal")
                        st.metric("胜率变化", 
                                f"{(inc_bt.get('胜率', 0) - orig_bt.get('胜率', 0))*100:.2f}%",
                                delta_color="normal")
                        st.metric("最大回撤变化", 
                                f"{(inc_bt.get('最大回撤', 0) - orig_bt.get('最大回撤', 0))*100:.2f}%",
                                delta_color="inverse")
                        st.metric("交易笔数变化", 
                                f"{inc_bt.get('交易笔数', 0) - orig_bt.get('交易笔数', 0)}",
                                delta_color="normal")

                    st.subheader("本次微调参数摘要")
                    ft_params = st.session_state.finetune_params
                    params_df = pd.DataFrame({
                        '参数': ['学习率', '最大训练轮数', '冻结策略', '混合数据比例', 
                            '峰模型验证准确率', '峰模型实际训练轮数',
                            '谷模型验证准确率', '谷模型实际训练轮数'],
                        '值': [
                            f"{ft_params['lr']:.1e}", 
                            str(ft_params['epochs']),
                            freeze_choice,
                            f"{ft_params['mix_ratio']:.1f}",
                            f"{ft_params['peak_val_acc']:.4f}",
                            f"{ft_params['peak_epochs']}/{ft_params['epochs']}",
                            f"{ft_params['trough_val_acc']:.4f}",
                            f"{ft_params['trough_epochs']}/{ft_params['epochs']}"
                        ]
                    })
                    st.dataframe(params_df, use_container_width=True)

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
                    st.error(f"模型微调过程中出错: {str(e)}")
                    st.exception(e)


    # =======================================
    #   Tab5: 因子研究
    # =======================================
    with tab5:
        st.title("因子研究系统")
        st.markdown("---")
        
        main_col1, main_col2 = st.columns([1, 1], gap="large")
        
        # =========== 左侧主区域 ============
        with main_col1:
            # ---------------- 因子选择 ----------------
            with st.container():
                st.subheader("因子选择")
                user_factor_select = []
                if 'factor_label_map' not in st.session_state:
                    # 载入系统因子字典，并构建因子标签映射
                    with open("system_factor.json", "r", encoding="utf-8-sig") as f:
                        system_factors = json.load(f)
                    st.session_state.factor_label_map = {
                        fname: flabel 
                        for category in system_factors.values() 
                        for fname, flabel in category.items()
                    }
                
                # 系统因子选择
                st.markdown("### 系统因子库")
                tab5_selected_system = []
                for category, factors in system_factor_dict.items():
                    with st.expander(category, expanded=False):
                        for factor_name, label in factors.items():
                            is_checked = factor_name in st.session_state.get('selected_system_factors', [])
                            checked = st.checkbox(label, value=is_checked, key=f"sys_{factor_name}")
                            if checked:
                                tab5_selected_system.append(factor_name)
                st.session_state.selected_system_factors = tab5_selected_system
                # 计算未选择的系统因子
                tab5_unselected_system = []
                for category, factors in system_factor_dict.items():
                    for factor_name in factors.keys():
                        if factor_name not in st.session_state.selected_system_factors:
                            tab5_unselected_system.append(factor_name)
                
                # 我的因子库选择
                st.markdown("### 我的因子库")
                tab5_selected_custom = []
                user_data = st.session_state.user_factor_map.get(USER_ID, {})
                for factor_name, detail in user_data.items():
                    label = st.session_state.factor_label_map.get(factor_name, factor_name)
                    is_checked = factor_name in st.session_state.get('selected_custom_factors', [])
                    checked = st.checkbox(label, value=is_checked, key=f"cust_{factor_name}")
                    if checked:
                        tab5_selected_custom.append(factor_name)
                st.session_state.selected_custom_factors = tab5_selected_custom
                
                # 显示选择情况
                with st.container():
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**已选系统因子**")
                        if st.session_state.selected_system_factors:
                            st.write("\n".join(
                                [f"- {st.session_state.factor_label_map.get(f, f)}" 
                                for f in st.session_state.selected_system_factors]
                            ))
                        else:
                            st.caption("（未选择）")
                    with cols[1]:
                        st.markdown("**已选自定义因子**")
                        tab5_selected_func_names = []
                        if st.session_state.selected_custom_factors:
                            custom_data = st.session_state.user_factor_map.get(USER_ID, {})
                            for fac in st.session_state.selected_custom_factors:
                                label = st.session_state.factor_label_map.get(fac, fac)
                                func_name = custom_data[fac].get("func_name", "")
                                tab5_selected_func_names.append(func_name)
                                st.write(f"- {label} ({func_name})")
                        else:
                            st.caption("（未选择）")
                
            # ---------------- 创建因子 ----------------
            with st.container():
                st.subheader("创建因子")
                
                if 'generated_code' not in st.session_state:
                    st.session_state.generated_code = None
                    
                prompt_str = st.text_area(
                    "描述你的因子需求（支持量价因子）", 
                    placeholder="例：创建一个基于20日均线突破的动量因子",
                    height=100
                )
                
                if st.button("生成因子代码", key="generate_factor_code"):
                    if prompt_str.strip():
                        with st.spinner("AI正在生成代码..."):
                            code_str = get_generated_code_cached(prompt_str)
                            st.session_state.generated_code = code_str
                    else:
                        st.warning("请输入有效的需求描述")
        
                if st.session_state.generated_code:
                    st.code(st.session_state.generated_code, language="python")
                    factor_name = st.text_input("命名你的因子", placeholder="请输入中文名称")
                    
                    if st.button("保存因子", key="save_custom_factor"):
                        if factor_name.strip():
                            try:
                                file_path, func_name = save_user_function(USER_ID, st.session_state.generated_code)
                                register_user_factor(USER_ID, factor_name, os.path.basename(file_path), func_name)
                                st.success("因子保存成功！")
                                st.session_state.generated_code = None
                            except Exception as e:
                                st.error(f"保存失败: {str(e)}")
                        else:
                            st.warning("请输入有效的因子名称")
            
            # ---------------- 保存当前选择 ----------------
            with st.container():
                st.subheader("保存当前选择")
                selection_name = st.text_input("为当前选择命名", placeholder="请输入一个名称")
                if st.button("保存选择", key="save_selection"):
                    if selection_name.strip():
                        try:
                            filename = "user_factor_selections.json"
                            # 尝试读取已有的保存文件
                            if os.path.exists(filename):
                                with open(filename, "r", encoding="utf-8") as f:
                                    selections = json.load(f)
                            else:
                                selections = {}
                            # 更新当前用户下的记录
                            if USER_ID not in selections:
                                selections[USER_ID] = {}
                            selections[USER_ID][selection_name] = {
                                "unselected_system": tab5_unselected_system,
                                "selected_system": tab5_selected_system,
                                "selected_custom": tab5_selected_custom,
                                "selected_custom_func_name": tab5_selected_func_names,
                            }
                            with open(filename, "w", encoding="utf-8") as f:
                                json.dump(selections, f, indent=4, ensure_ascii=False)
                            st.success("选择保存成功！")
                        except Exception as e:
                            st.error(f"保存失败: {str(e)}")
                    else:
                        st.warning("请输入有效的名称")
        
        # =========== 右侧主区域：因子验证 ============
        with main_col2:
            with st.container():
                st.subheader("因子验证")
                
                with st.expander("验证参数", expanded=True):
                    param_col1, param_col2 = st.columns(2)
                    with param_col1:
                        fv_data_type = st.selectbox("数据类型", ["指数", "股票"], key="fv_type")
                        fv_asset_code = st.text_input("标的代码", "000001.SH")
                    with param_col2:
                        fv_model_type = st.selectbox("模型类型", ["高点模型", "低点模型"], key="fv_model")
                        fv_scoring_method = st.selectbox(
                            "评分方法", 
                            ["pearson", "spearman", "mutual_info", "chi2", "anova_f", "rftree", "mlp"],
                            format_func=lambda x: x.upper()
                        )
                    
                    date_col1, date_col2 = st.columns(2)
                    with date_col1:
                        fv_start_date = st.date_input("起始日期", datetime(2021, 1, 1))
                    with date_col2:
                        fv_end_date = st.date_input("结束日期", datetime.now())
                
                if st.button("开始验证", type="primary", use_container_width=True):
                    try:
                        symbol_type = "index" if fv_data_type == "指数" else "stock"
                        raw_data = read_day_from_tushare(fv_asset_code, symbol_type)
                        
                        N_val = st.session_state.models.get("N", 30) if "models" in st.session_state else 30
                        mixture_depth_val = st.session_state.models.get("mixture_depth", 1) if "models" in st.session_state else 1
                        
                        df_all, _ = preprocess_data(raw_data, N_val, mixture_depth_val, mark_labels=True, selected_system=selected_system, selected_func_names=selected_func_names,)
                        df_data = select_time(
                            df_all, 
                            fv_start_date.strftime("%Y%m%d"), 
                            fv_end_date.strftime("%Y%m%d")
                        )
                        
                        target_col = "Peak" if fv_model_type == "高点模型" else "Trough"
                        y = df_data[target_col]
                        
                        system_keys = st.session_state.selected_system_factors
                        custom_names = st.session_state.selected_custom_factors
                        selected_columns = system_keys + custom_names
        
                        X = df_data[selected_columns].dropna()
                        y = df_data[target_col].loc[X.index]
                        
                        method = fv_scoring_method.lower()
                        score_series = None
                        if method in ["pearson", "spearman"]:
                            corr = X.corrwith(y, method=method)
                            score_series = corr.abs().sort_values(ascending=False)
                        elif method == "mutual_info":
                            from sklearn.feature_selection import mutual_info_classif
                            mi_scores = mutual_info_classif(X, y, random_state=42)
                            score_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
                        elif method == "chi2":
                            from sklearn.feature_selection import chi2
                            chi2_vals, p_vals = chi2(X, y)
                            df_chi = pd.DataFrame({"feature": X.columns, "chi2": chi2_vals, "pval": p_vals})
                            df_chi.sort_values(by="chi2", ascending=False, inplace=True)
                            score_series = pd.Series(df_chi["chi2"].values, index=df_chi["feature"])
                        elif method == "anova_f":
                            from sklearn.feature_selection import f_classif
                            fvals, pvals = f_classif(X, y)
                            df_f = pd.DataFrame({"feature": X.columns, "fval": fvals, "pval": pvals})
                            df_f.sort_values(by="fval", ascending=False, inplace=True)
                            score_series = pd.Series(df_f["fval"].values, index=df_f["feature"])
                        elif method == "rftree":
                            from sklearn.ensemble import RandomForestClassifier
                            rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                            rf_fs.fit(X, y)
                            importances = rf_fs.feature_importances_
                            score_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                        elif method == "mlp":
                            from sklearn.neural_network import MLPClassifier
                            mlp_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, random_state=42)
                            mlp_model.fit(X, y)
                            importance = np.abs(mlp_model.coefs_[0]).sum(axis=1)
                            score_series = pd.Series(importance, index=X.columns).sort_values(ascending=False)
                        else:
                            st.error(f"未知评分方式: {fv_scoring_method}")
                        
                        st.session_state.validation_result = score_series
        
                    except Exception as e:
                        st.error(f"验证失败: {str(e)}")
        
                    if 'validation_result' in st.session_state:
                        score_series = st.session_state.validation_result
                        label_map = st.session_state.factor_label_map
                        score_series.index = [label_map.get(idx, idx) for idx in score_series.index]
        
                        fig = px.bar(
                            score_series.reset_index(),
                            x='index',
                            y=0,
                            labels={'index': '因子名称', 0: '评分值'},
                            title="因子评分分布"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
                        with st.expander("📋 详细数据"):
                            df_display = score_series.reset_index()
                            df_display.columns = ['因子名称', '评分']
                            df_display['排名'] = range(1, len(df_display) + 1)
                            st.dataframe(
                                df_display[['排名', '因子名称', '评分']],
                                column_config={
                                    "排名": st.column_config.NumberColumn(width="small"),
                                    "评分": st.column_config.ProgressColumn(
                                        format="%.2f",
                                        min_value=0,
                                        max_value=float(df_display['评分'].max())
                                    )
                                },
                                hide_index=True,
                                use_container_width=True
                            )
        
        st.markdown("---")
        st.caption("💡 提示：右键表格可排序，点击图表可交互")


if __name__ == "__main__":
    main_product()
