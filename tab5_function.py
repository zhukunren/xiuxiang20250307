#tab5_function
import streamlit as st 
import re
import os
import time
import json
from openai import OpenAI
import importlib.util
USER_FACTOR_MAP_FILE = "user_factor_map.json"
BASE_DIR = "user_functions"
def load_user_factor_map(USER_FACTOR_MAP_FILE) -> dict:
    if not os.path.exists(USER_FACTOR_MAP_FILE):
        return {}
    with open(USER_FACTOR_MAP_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)

def save_user_factor_map(data: dict):
    with open(USER_FACTOR_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def register_user_factor(user_id: str, factor_name: str, file_name: str, func_name: str):
    tmp_data = load_user_factor_map(USER_FACTOR_MAP_FILE)
    if user_id not in tmp_data:
        tmp_data[user_id] = {}
    tmp_data[user_id][factor_name] = {
        "file_name": file_name,
        "func_name": func_name
    }
    save_user_factor_map(tmp_data)
    st.session_state.user_factor_map = tmp_data
    st.success(f"自定义因子 '{factor_name}' 已成功保存到文件 {file_name}。")

def save_user_function(user_id: str, code_str: str):
    func_name_match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code_str)
    if not func_name_match:
        raise ValueError("无法从因子代码中解析出函数名，请检查生成的代码格式。")
    func_name = func_name_match.group(1)
    user_dir = os.path.join(BASE_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    file_name = f"{func_name}.py"
    file_path = os.path.join(user_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_str)
    return file_path, func_name
def my_factors(user_id: str):
            st.markdown("### 我的自定义因子列表")
            data = st.session_state.user_factor_map.get(user_id, {})
            if data:
                for f_name, detail in data.items():
                    f_file = detail.get("file_name", "")
                    f_func = detail.get("func_name", "")
                    st.write(f"- **因子名称**: {f_name}, 文件: {f_file}, 函数: {f_func}")
            else:
                st.write("（暂无自定义因子）")
'''                
def display_factor_validation():
    st.subheader("因子验证")
    with st.expander("验证参数", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fv_data_type = st.selectbox("数据类型", ["指数", "股票"], key="fv_type")
            fv_asset_code = st.text_input("标的代码", "000001.SH")
        with col2:
            fv_model_type = st.selectbox("模型类型", ["高点模型", "低点模型"], key="fv_model")
            fv_scoring_method = st.selectbox(
                "评分方法",
                ["pearson", "spearman", "mutual_info", "chi2", "anova_f", "rftree", "mlp"],
                format_func=lambda x: x.upper(),
                key="fv_score"
            )
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            fv_start_date = st.date_input("起始日期", datetime(2021, 1, 1))
        with date_col2:
            fv_end_date = st.date_input("结束日期", datetime.now())
    # 数据处理与评分计算
    try:
        symbol_type = "index" if fv_data_type == "指数" else "stock"
        raw_data = read_day_from_tushare(fv_asset_code, symbol_type)
        # 获取预处理参数（若模型已训练则从 session_state 中获取，否则使用默认值）
        N_val = st.session_state.models.get("N", 30) if "models" in st.session_state else 30
        mixture_depth_val = st.session_state.models.get("mixture_depth", 1) if "models" in st.session_state else 1
        df_all, _ = preprocess_data(raw_data, N_val, mixture_depth_val, mark_labels=True)
        df_data = select_time(df_all, fv_start_date.strftime("%Y%m%d"), fv_end_date.strftime("%Y%m%d"))
        target_col = "Peak" if fv_model_type == "高点模型" else "Trough"
        y = df_data[target_col]
        # 合并选择的因子（系统因子与自定义因子）
        system_keys = st.session_state.get('selected_system_factors', [])
        custom_keys = st.session_state.get('selected_custom_factors', [])
        selected_columns = system_keys + custom_keys
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
            return
        st.session_state.validation_result = score_series
    except Exception as e:
        st.error(f"验证失败: {str(e)}")
        return
'''
def get_generated_code(prompt_str, deepseek_model,retries=3, delay=5):
    """
    让 AI 返回一个【带时间戳】的函数名，并生成相应代码。
    （AI 会在函数名的末尾加 YYYYMMDD_HHMMSS 时间戳）
    """
    system_instruction = "你是专业的证券分析师和机器学习专家。"

    user_prompt = f"""
    我需要基于日线行情构建量化交易数据特征用于机器学习。请生成一个 Python 函数，函数名可自行决定，但名称的末尾必须带上当前日期时间戳（格式形如 YYYYMMDD_HHMMSS）。
    函数接收一个 pandas DataFrame 参数 df（包含列：Open, High, Low, Close, Volume, Amount，索引为日期），
    请根据以下内容为df构建特征列：
    
    - {prompt_str}
    
    函数要求：
    1. 新增的特征列列名与函数名一致；
    2. 对于逐行赋值，请使用 .at[] 或 .loc[]，禁止使用 .iloc[]；
    3. 函数执行完毕后返回修改后的 df；
    4. 仅允许使用 pandas 和 numpy 库及，不得使用其他库；
    5. 只需要返回纯代码，不需要额外解释。
    """

    try:
        client = OpenAI(
            api_key="sk-1e63e70de8e5442594186ee9cf8e9ee6", 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        for attempt in range(retries):
            try:
                print(f"请求生成代码，尝试 {attempt + 1}/{retries}...")
                completion = client.chat.completions.create(
                    model=deepseek_model,  # 或者你所使用的模型
                    messages=[
                        {'role': 'system', 'content': system_instruction},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                print("代码生成成功")
                code_str = completion.choices[0].message.content
                # 去除 Markdown 包裹
                code_str = code_str.strip("```python").strip("```").strip()
                return code_str  # 直接返回AI生成的代码
            except Exception as e:
                print(f"请求失败，错误信息: {e}")
                if attempt < retries - 1:
                    print(f"重试中...{attempt + 1}/{retries}")
                    time.sleep(delay)
                else:
                    print("已达到最大重试次数。")
                    return None
    except Exception as e:
        print(f"调用 OpenAI 失败：{e}")
        return None
def load_user_function(user_id: str, func_file_name: str):
    """
    根据文件名动态加载对应的 Python 模块，并返回其中的函数对象。
    默认函数名与文件名一致（去掉 .py 后缀）。
    """
    file_path = os.path.join(BASE_DIR, str(user_id), func_file_name+'.py')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无法找到文件: {file_path}")
    
    spec = importlib.util.spec_from_file_location("user_factor_mod", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 再次用正则或和你的存储规则去拿“def 函数名”，
    func_name = os.path.splitext(func_file_name)[0]  # 假定一致
    if not hasattr(module, func_name):
        raise AttributeError(f"模块中未找到函数 {func_name} 的定义。")

    return getattr(module, func_name)
def apply_factors_in_sequence(user_id: str, factor_names: list, df,user_factor_map):
    """
    根据因子名称列表，依次加载并执行每个因子对应的函数。
    前一个函数输出的 df，作为后一个函数的输入 df。
    最后返回最终 df。
    """
    if user_id not in user_factor_map:
        raise ValueError(f"用户 {user_id} 在 user_factor_map 中不存在任何因子映射。")

    for factor_name in factor_names:
        if factor_name not in user_factor_map[user_id]:
            raise ValueError(f"用户 {user_id} 未注册因子 '{factor_name}'")

        func_file_name = user_factor_map[user_id][factor_name]['func_name']
        func_obj = load_user_function(user_id, func_file_name)

        print(f"[apply_factors_in_sequence] 执行因子: {factor_name}, 对应函数文件: {func_file_name}")
        df = func_obj(df)  # 调用函数，得到更新后的 df
        #重命名列
        df.rename(columns={f'{func_file_name}': f'{factor_name}'}, inplace=True)

    return df
@st.cache_data(show_spinner=False)
def get_generated_code_cached(prompt_str: str) -> str:
    return get_generated_code(prompt_str, deepseek_model="qwen-turbo-latest")