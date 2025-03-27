# predict.py
import numpy as np
import torch
from preprocess import preprocess_data
from skorch import NeuralNetClassifier
from backtest import backtest_results
from models import  TransformerClassifier
import pandas as pd

#绘图函数

# ============== 预测新数据的函数 (修改后返回数据与回测结果) ==============
def merge_trades(data_preprocessed, trades_df):
    """
    合并交易数据并保持原始索引，确保在合并后日期列 'date' 与交易信号一致。
    """
    # 保存原始索引
    original_index = data_preprocessed.index

    # 合并卖出日期，确保 exit_date 对齐到 data_preprocessed['date']
    data_preprocessed = pd.merge(
        data_preprocessed, 
        trades_df[['exit_date']],  # 选择 trades_df 中的 'exit_date'
        left_on='date',            # 使用 data_preprocessed 中的 'date' 列进行合并
        right_on='exit_date',      # 使用 trades_df 中的 'exit_date' 列进行合并
        how='left'                 # 使用左连接，保留 data_preprocessed 中所有行
    )

    # 设置 trade 为 'sell' 当 exit_date 非空时
    data_preprocessed['trade'] = np.where(data_preprocessed['exit_date'].notna(), 'sell', data_preprocessed['trade'])

    # 合并 entry_date
    data_preprocessed = pd.merge(
        data_preprocessed, 
        trades_df[['entry_date']],  # 选择 trades_df 中的 'entry_date'
        left_on='date',                           # 使用 data_preprocessed 中的 'date' 列进行合并
        right_on='entry_date',                    # 使用 trades_df 中的 'entry_date' 列进行合并
        how='left'                                # 使用左连接，保留 data_preprocessed 中所有行
    )

    # 设置 trade 为 'buy' 当 entry_date 非空时
    data_preprocessed['trade'] = np.where(data_preprocessed['entry_date'].notna(), 'buy', data_preprocessed['trade'])

    # 删除重复日期
    data_preprocessed = data_preprocessed.drop_duplicates(subset=['date'])
    print(data_preprocessed['trade'])
    
    # 恢复原始索引
    data_preprocessed.index = original_index

    return data_preprocessed
def predict_new_data(
    new_df,
    peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold,
    trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold,
    N, mixture_depth=3, window_size=300, eval_mode=False, 
    N_buy=None, N_sell=None,  # 追涨、止损窗口
    enable_chase=True, 
    enable_stop_loss=True,
    enable_change_signal=False,
    N_newhigh=60
):
    """
    使用训练好的模型（峰/谷）对 new_df 做预测，并可选做回测。
    假设 new_df 已经是全量原始数据经过 select_time 截取后的数据，
    因此不再在内部进行额外的特征工程处理。
    """
    print("开始预测新数据...")
    try:
        # 直接复制传入的数据，不再额外调用 preprocess_data
        data_preprocessed = new_df.copy()
        
        # ========== 预测 Peak ==========
        print("\n开始 Peak 预测...")
        missing_peak = [f for f in peak_selected_features if f not in data_preprocessed.columns]
        if missing_peak:
            print(f"填充缺失特征(Peak): {missing_peak}")
            for feature in missing_peak:
                data_preprocessed[feature] = 0
        X_new_peak = data_preprocessed[peak_selected_features].fillna(0)
        X_new_peak_scaled = peak_scaler.transform(X_new_peak).astype(np.float32)
        print(f"Peak数据形状: {X_new_peak_scaled.shape}")
        
        from skorch import NeuralNetClassifier
        from models import TransformerClassifier
        # 如果是 Transformer 模型，则构建时序数据
        if (hasattr(peak_model, "module_") and isinstance(peak_model.module_, TransformerClassifier)):
            print("创建 Peak 序列数据...")
            X_seq_list = [X_new_peak_scaled[i - window_size:i] for i in range(window_size, len(X_new_peak_scaled) + 1)]
            X_new_seq_peak = np.array(X_seq_list, dtype=np.float32)
            print(f"Peak序列数据形状: {X_new_seq_peak.shape}")
            batch_size = 64
            predictions = []
            peak_model.module_.eval()
            import torch
            with torch.no_grad():
                for i in range(0, len(X_new_seq_peak), batch_size):
                    batch = torch.from_numpy(X_new_seq_peak[i : i + batch_size]).float()
                    batch = batch.to(peak_model.device)
                    outputs = peak_model.module_(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predictions.append(probs.cpu().numpy())
            all_probas = np.concatenate(predictions)
            peak_probas = np.zeros(len(data_preprocessed))
            peak_probas[window_size - 1:] = all_probas
        else:
            if hasattr(peak_model, "predict_proba"):
                if peak_selector is not None:
                    X_new_peak_selected = peak_selector.transform(X_new_peak_scaled)
                    logits = peak_model.predict_proba(X_new_peak_selected)
                else:
                    logits = peak_model.predict_proba(X_new_peak_scaled)
                if logits.ndim == 2:
                    peak_probas = logits[:, 1]
                else:
                    import torch
                    peak_probas = torch.sigmoid(torch.tensor(logits)).numpy()
            else:
                peak_probas = peak_model.predict(X_new_peak_scaled).astype(float)
        peak_preds = (peak_probas > peak_threshold).astype(int)
        data_preprocessed['Peak_Probability'] = peak_probas
        data_preprocessed['Peak_Prediction'] = peak_preds
        
        # ========== 预测 Trough ==========
        print("\n开始 Trough 预测...")
        missing_trough = [f for f in trough_selected_features if f not in data_preprocessed.columns]
        if missing_trough:
            print(f"填充缺失特征(Trough): {missing_trough}")
            for feature in missing_trough:
                data_preprocessed[feature] = 0
        X_new_trough = data_preprocessed[trough_selected_features].fillna(0)
        X_new_trough_scaled = trough_scaler.transform(X_new_trough).astype(np.float32)
        print(f"Trough数据形状: {X_new_trough_scaled.shape}")
        if (hasattr(trough_model, "module_") and isinstance(trough_model.module_, TransformerClassifier)):
            print("创建 Trough 序列数据...")
            X_seq_list = [X_new_trough_scaled[i - window_size:i] for i in range(window_size, len(X_new_trough_scaled) + 1)]
            X_new_seq_trough = np.array(X_seq_list, dtype=np.float32)
            print(f"Trough序列数据形状: {X_new_seq_trough.shape}")
            batch_size = 64
            predictions = []
            trough_model.module_.eval()
            with torch.no_grad():
                for i in range(0, len(X_new_seq_trough), batch_size):
                    batch = torch.from_numpy(X_new_seq_trough[i : i + batch_size]).float()
                    batch = batch.to(trough_model.device)
                    outputs = trough_model.module_(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predictions.append(probs.cpu().numpy())
            all_probas = np.concatenate(predictions)
            trough_probas = np.zeros(len(data_preprocessed))
            trough_probas[window_size - 1:] = all_probas
        else:
            if hasattr(trough_model, "predict_proba"):
                if trough_selector is not None:
                    X_new_trough_selected = trough_selector.transform(X_new_trough_scaled)
                    logits = trough_model.predict_proba(X_new_trough_selected)
                else:
                    logits = trough_model.predict_proba(X_new_trough_scaled)
                if logits.ndim == 2:
                    trough_probas = logits[:, 1]
                else:
                    import torch
                    trough_probas = torch.sigmoid(torch.tensor(logits)).numpy()
            else:
                trough_probas = trough_model.predict(X_new_trough_scaled).astype(float)
        trough_preds = (trough_probas > trough_threshold).astype(int)
        data_preprocessed['Trough_Probability'] = trough_probas
        data_preprocessed['Trough_Prediction'] = trough_preds
        
        # ========== 后处理：20日内不重复预测 ==========
        print("\n进行后处理...")
        if not data_preprocessed.empty:
            # 先重置索引为整数索引，便于后续操作
            data_preprocessed = data_preprocessed.reset_index(drop=True)
            for idx in range(len(data_preprocessed)):
                if data_preprocessed.loc[idx, 'Peak_Prediction'] == 1:
                    start = idx + 1
                    end = min(idx + 20, len(data_preprocessed))
                    data_preprocessed.loc[start:end, 'Peak_Prediction'] = 0
                if data_preprocessed.loc[idx, 'Trough_Prediction'] == 1:
                    start = idx + 1
                    end = min(idx + 20, len(data_preprocessed))
                    data_preprocessed.loc[start:end, 'Trough_Prediction'] = 0

        # ========== 回测 ==========
        print("\n进行回测...")
        signal_df = get_trade_signal(data_preprocessed)
        bt_result, trades_df = backtest_results(
            data_preprocessed, 
            signal_df,
            N_buy,           # 追涨窗口
            N_sell,          # 止损窗口
            enable_chase,    # 是否启用追涨
            enable_stop_loss,# 是否启用止损
            initial_capital=1_000_000
        )
        # 统一使用日期列（TradeDate 或 index）
        if 'TradeDate' in data_preprocessed.columns:
            data_preprocessed['date'] = pd.to_datetime(data_preprocessed['TradeDate'], errors='coerce')
        else:
            data_preprocessed['date'] = pd.to_datetime(data_preprocessed.index, errors='coerce')
        data_preprocessed['trade'] = None
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[['exit_date']],
            left_on='date',
            right_on='exit_date',
            how='left'
        )
        data_preprocessed['trade'] = np.where(
            data_preprocessed['exit_date'].notna(), 
            'sell', 
            data_preprocessed['trade']
        )
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[['entry_date']],
            left_on='date',
            right_on='entry_date',
            how='left'
        )
        data_preprocessed['trade'] = np.where(
            data_preprocessed['entry_date'].notna(),
            'buy',
            data_preprocessed['trade']
        )
        data_preprocessed = data_preprocessed.drop_duplicates(subset=['date'])
        data_preprocessed.set_index('date', inplace=True)
    except Exception as e:
        print('predict_new_data函数出错:', e)
        if 'trades_df' in locals():
            print("回测结果：", trades_df)
        else:
            print("未生成交易结果")
        raise e

    return data_preprocessed, bt_result, trades_df

'''
def predict_new_data(
    new_df,
    peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold,
    trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold,
    N, mixture_depth=3, window_size=300, eval_mode=False, 
    N_buy=None, N_sell=None,  # 追涨、止损窗口
    enable_chase=True, 
    enable_stop_loss=True,
    enable_change_signal=False,
    N_newhigh=60
):
    """
    使用训练好的模型(峰/谷)对 new_df 做预测，并可选做回测。
    注意：peak_selected_features/trough_selected_features 是模型真正见过的特征列表。
    """
    print("开始预测新数据...")

    try:
        # 首先做预处理
        data_preprocessed, _ = preprocess_data(
            new_df, 
            N, 
            mixture_depth=mixture_depth, 
            mark_labels=eval_mode
        )

        # ========== 预测 Peak ==========
        print("\n开始 Peak 预测...")

        # 若有 selected_features 中的列在新数据缺失，就补全为0
        missing_peak = [f for f in peak_selected_features if f not in data_preprocessed.columns]
        if missing_peak:
            print(f"填充缺失特征(Peak): {missing_peak}")
            for feature in missing_peak:
                data_preprocessed[feature] = 0

        # 只取模型真正用的列
        X_new_peak = data_preprocessed[peak_selected_features].fillna(0)

        # 调用训练时的 scaler
        X_new_peak_scaled = peak_scaler.transform(X_new_peak).astype(np.float32)
        print(f"Peak数据形状: {X_new_peak_scaled.shape}")

        # 如果是 Transformer 模型，还得构建序列
        from skorch import NeuralNetClassifier
        from models import TransformerClassifier
        
        if (isinstance(peak_model, NeuralNetClassifier) and
            isinstance(peak_model.module_, TransformerClassifier)):
            print("创建 Peak 序列数据...")
            X_seq_list = []
            for i in range(window_size, len(X_new_peak_scaled) + 1):
                seq_x = X_new_peak_scaled[i - window_size:i]
                X_seq_list.append(seq_x)
            X_new_seq_peak = np.array(X_seq_list, dtype=np.float32)
            print(f"Peak序列数据形状: {X_new_seq_peak.shape}")

            batch_size = 64
            predictions = []
            peak_model.module_.eval()

            import torch
            with torch.no_grad():
                for i in range(0, len(X_new_seq_peak), batch_size):
                    batch = torch.from_numpy(X_new_seq_peak[i : i + batch_size]).float()
                    batch = batch.to(peak_model.device)
                    outputs = peak_model.module_(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predictions.append(probs.cpu().numpy())
            
            all_probas = np.concatenate(predictions)
            peak_probas = np.zeros(len(data_preprocessed))
            peak_probas[window_size-1:] = all_probas
        else:
            # 传统模型或 MLP
            if hasattr(peak_model, "predict_proba"):
                # 若 peak_selector 不为空，可再做 transform(不过您这儿是identity，影响不大)
                if peak_selector is not None:
                    X_new_peak_selected = peak_selector.transform(X_new_peak_scaled)
                    logits = peak_model.predict_proba(X_new_peak_selected)
                else:
                    logits = peak_model.predict_proba(X_new_peak_scaled)
                
                if logits.ndim == 2:
                    peak_probas = logits[:, 1]
                else:
                    # 万一只输出一维
                    import torch
                    peak_probas = torch.sigmoid(torch.tensor(logits)).numpy()
            else:
                # predict_proba 不存在的话，只能 predict
                peak_probas = peak_model.predict(X_new_peak_scaled).astype(float)

        peak_preds = (peak_probas > peak_threshold).astype(int)
        data_preprocessed['Peak_Probability'] = peak_probas
        data_preprocessed['Peak_Prediction'] = peak_preds

        # ========== 预测 Trough ==========
        print("\n开始 Trough 预测...")

        missing_trough = [f for f in trough_selected_features if f not in data_preprocessed.columns]
        if missing_trough:
            print(f"填充缺失特征(Trough): {missing_trough}")
            for feature in missing_trough:
                data_preprocessed[feature] = 0

        X_new_trough = data_preprocessed[trough_selected_features].fillna(0)
        X_new_trough_scaled = trough_scaler.transform(X_new_trough).astype(np.float32)
        print(f"Trough数据形状: {X_new_trough_scaled.shape}")

        if (isinstance(trough_model, NeuralNetClassifier) and
            isinstance(trough_model.module_, TransformerClassifier)):
            print("创建 Trough 序列数据...")
            X_seq_list = []
            for i in range(window_size, len(X_new_trough_scaled) + 1):
                seq_x = X_new_trough_scaled[i - window_size:i]
                X_seq_list.append(seq_x)
            X_new_seq_trough = np.array(X_seq_list, dtype=np.float32)
            print(f"Trough序列数据形状: {X_new_seq_trough.shape}")

            batch_size = 64
            predictions = []
            trough_model.module_.eval()

            with torch.no_grad():
                for i in range(0, len(X_new_seq_trough), batch_size):
                    batch = torch.from_numpy(X_new_seq_trough[i : i + batch_size]).float()
                    batch = batch.to(trough_model.device)
                    outputs = trough_model.module_(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predictions.append(probs.cpu().numpy())
            
            all_probas = np.concatenate(predictions)
            trough_probas = np.zeros(len(data_preprocessed))
            trough_probas[window_size-1:] = all_probas
        else:
            if hasattr(trough_model, "predict_proba"):
                if trough_selector is not None:
                    X_new_trough_selected = trough_selector.transform(X_new_trough_scaled)
                    logits = trough_model.predict_proba(X_new_trough_selected)
                else:
                    logits = trough_model.predict_proba(X_new_trough_scaled)
                
                if logits.ndim == 2:
                    trough_probas = logits[:, 1]
                else:
                    import torch
                    trough_probas = torch.sigmoid(torch.tensor(logits)).numpy()
            else:
                trough_probas = trough_model.predict(X_new_trough_scaled).astype(float)

        trough_preds = (trough_probas > trough_threshold).astype(int)
        data_preprocessed['Trough_Probability'] = trough_probas
        data_preprocessed['Trough_Prediction'] = trough_preds

        # ====== 后处理：20日内不重复预测 (根据您原先的逻辑) ======
        print("\n进行后处理...")
        data_preprocessed.index = data_preprocessed.index.astype(str)
        for idx, index in enumerate(data_preprocessed.index):
            if data_preprocessed.loc[index, 'Peak_Prediction'] == 1:
                start = idx + 1
                end = min(idx + 20, len(data_preprocessed))
                data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Peak_Prediction')] = 0
            if data_preprocessed.loc[index, 'Trough_Prediction'] == 1:
                start = idx + 1
                end = min(idx + 20, len(data_preprocessed))
                data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Trough_Prediction')] = 0

        # 若启用其他信号改动
        if enable_change_signal:
            data_preprocessed = change_troug_and_peak(data_preprocessed, N_newhigh)

        # 回测部分
        signal_df = get_trade_signal(data_preprocessed)
        bt_result, trades_df = backtest_results(
            data_preprocessed, 
            signal_df,
            N_buy,           # 追涨窗口
            N_sell,          # 止损窗口
            enable_chase,    # 是否启用追涨
            enable_stop_loss,# 是否启用止损
            initial_capital=1_000_000
        )

        # 用 'TradeDate' 或索引做时间列
        if 'TradeDate' in data_preprocessed.columns:
            data_preprocessed['date'] = pd.to_datetime(data_preprocessed['TradeDate'], errors='coerce')
        else:
            data_preprocessed['date'] = pd.to_datetime(data_preprocessed.index, errors='coerce')

        data_preprocessed['trade'] = None
        # 合并卖出日期
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[['exit_date']],
            left_on='date',
            right_on='exit_date',
            how='left'
        )
        data_preprocessed['trade'] = np.where(
            data_preprocessed['exit_date'].notna(), 
            'sell', 
            data_preprocessed['trade']
        )

        # 合并买入日期
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[['entry_date']],
            left_on='date',
            right_on='entry_date',
            how='left'
        )
        data_preprocessed['trade'] = np.where(
            data_preprocessed['entry_date'].notna(),
            'buy',
            data_preprocessed['trade']
        )

        # 删除重复
        data_preprocessed = data_preprocessed.drop_duplicates(subset=['date'])
        data_preprocessed.set_index('date', inplace=True)

    except Exception as e:
        print('predict_new_data函数出错:', e)
        if 'trades_df' in locals():
            print("回测结果：", trades_df)
        else:
            print("未生成交易结果")
        raise e

    return data_preprocessed, bt_result, trades_df
'''

#W出现于阴线，D出现于阳线，且盘中要创60日新高
def change_troug_and_peak(df,N_newhigh):
    
    def update_peak_or_trough(df, prediction_col, opposite_col, condition):
        for i, date in enumerate(df.index):
            # 只处理预测值为1的情况
            if df.loc[date, prediction_col] == 1:
                if condition(df, i, date):
                    df.loc[date, prediction_col] = 1  # 保持当前预测
                else:
                    df.loc[date, prediction_col] = 0  # 移除预测
                    # 寻找下一个符合条件的日期，将预测信号转移过去
                    for j in range(i + 1, len(df)):
                        next_date = df.index[j]
                        if condition(df, j, next_date):
                            df.loc[next_date, prediction_col] = 1
                            break
        return df

    # 高点处理：仅在阴线出现卖出信号，且当天 High 创过去60日 Close 新高
    df = update_peak_or_trough(
        df, 
        'Peak_Prediction', 
        'Trough_Prediction', 
        lambda df, i, date: (
            i >= N_newhigh and 
            df.loc[date, 'High'] > df.loc[df.index[i-N_newhigh:i], 'Close'].max() and 
            df.loc[date, 'Close'] < df.loc[date, 'Open']
        )
    )

    # 低点处理：仅在阳线显示低点信号
    df = update_peak_or_trough(
        df, 
        'Trough_Prediction', 
        'Peak_Prediction', 
        lambda df, i, date: df.loc[date, 'Close'] > df.loc[date, 'Open']
    )

    return df





def adjust_probabilities_in_range(df, start_date, end_date):
    """
    将 DataFrame 中指定日期范围内的 'Peak_Probability' 和 'Trough_Probability' 列的值设为 0。

    参数:
      df: 包含预测结果的 DataFrame，其索引为日期。
      start_date: 起始日期（字符串，格式 'YYYY-MM-DD'）。
      end_date: 截止日期（字符串，格式 'YYYY-MM-DD'）。

    返回:
      修改后的 DataFrame。
    """
    # 如果索引不是 datetime 类型，则转换为 datetime 类型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    
    if "Peak_Probability" in df.columns:
        df.loc[mask, "Peak_Prediction"] = 0
        df.loc[mask, "Peak"] = 0
        df.loc[mask, "Peak_Probability"] = 0
    if "Trough_Probability" in df.columns:
        df.loc[mask, "Trough_Prediction"] = 0
        df.loc[mask, "Trough"] = 0
        df.loc[mask, "Trough_Probability"] = 0
    return df


def get_trade_signal(data_preprocessed):
    # 复制数据以避免修改原始 DataFrame
    data_preprocessed = data_preprocessed.copy()

    # 筛选出存在高点或低点预测的行
    signal_df = data_preprocessed[(data_preprocessed['Peak_Prediction'] == 1) | 
                                  (data_preprocessed['Trough_Prediction'] == 1)]
    
    # 对于高点预测的行，设定方向为 'sell'
    signal_df.loc[signal_df['Peak_Prediction'] == 1, 'direction'] = 'sell'
    
    # 对于低点预测的行，设定方向为 'buy'
    signal_df.loc[signal_df['Trough_Prediction'] == 1, 'direction'] = 'buy'
    
    # 仅返回交易方向这一列
    signal_df = signal_df[['direction']]
    

    return signal_df
