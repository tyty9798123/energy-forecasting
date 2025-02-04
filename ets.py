# ets_walk_forward_example.py

import numpy as np
import pandas as pd
import time

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import (
    load_data, add_features, add_temp,  # <-- 你自己的前處理函式
    # ... 其他 import ...
)


def build_model_ETS(train_series, trend='add', seasonal='add', seasonal_periods=7):
    """
    建立並訓練 ExponentialSmoothing 模型
    train_series: pandas Series (1D)
    trend: 'add', 'mul' 或 None
    seasonal: 'add', 'mul' 或 None
    seasonal_periods: 季節長度 (如日資料有週期設 7，月資料有年週期設 12)
    """
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fitted_model = model.fit()
    return fitted_model

def evaluate_forecast(all_trues, all_preds):
    """
    對多步預測的所有結果（已經對齊好的陣列）做整體誤差評估。
    all_trues: shape = [num_rolls, horizon]
    all_preds: shape = [num_rolls, horizon]
    """
    # 也可根據需求對「每個 horizon step」分別計算誤差
    # 這裡範例是全部 flatten 在一起，算整體指標
    y_true = all_trues.flatten()
    y_pred = all_preds.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # squared=False → RMSE
    r2 = r2_score(y_true, y_pred)

    # 若 y_true 含 0，計算 MAPE 需避免除以零
    non_zero_idx = (y_true != 0)
    if sum(non_zero_idx) == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

    print(f"[Walk-Forward] MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
    return mae, rmse, r2, mape

def main_ETS_walk_forward():
    """
    使用 ETS 做「步進式（Walk-Forward）多步預測」的範例。
    每次預測 horizon=8 步，直到測試集資料走完為止。
    """

    # 1. 讀取 & 前處理
    df = load_data()       # 讀取自己的資料
    df = add_features(df)  # 若需要額外特徵
    df = add_temp(df)      # 若需要額外特徵
    
    # 2. 選定要預測的欄位
    target_series = df['value']
    
    # 3. 設定 horizon, 例如 8
    horizon = 8
    
    # 4. 簡單分 train / test
    train_ratio = 0.85
    train_size = int(len(target_series) * train_ratio)
    train_data = target_series[:train_size]
    test_data = target_series[train_size:]  # 全部測試區（長度 = N）
    print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")

    # 5. 進行步進式多步預測 (Walk-Forward)
    #    對測試區每個時間點 i (從 train_size 到結束-horizon)，
    #    皆用「過去(含當下)的所有資料」train_series[:i] 重新建模 → 預測下一段 horizon=8 步
    all_preds = []  # shape = [num_rolls, horizon]
    all_trues = []  # shape = [num_rolls, horizon]

    # 因為要預測 i ~ i+horizon-1，所以最遠只能到 len(target_series)-horizon
    for i in range(train_size, len(target_series) - horizon + 1):
        # (a) 以序列前 i 筆資料重新訓練 ETS
        model_ets = build_model_ETS(
            train_series=target_series[:i], 
            trend='add', 
            seasonal='add', 
            seasonal_periods=7
        )
        # (b) 預測未來 horizon 步
        forecast_values = model_ets.forecast(steps=horizon).values

        # (c) 取得真實值：即 target_series[i : i + horizon]
        true_values = target_series[i : i + horizon].values

        # (d) 收集
        all_preds.append(forecast_values)
        all_trues.append(true_values)

    # 轉成 numpy array, shape = [num_rolls, horizon]
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    print(f"all_preds shape = {all_preds.shape}, all_trues shape = {all_trues.shape}")
    # 其中 all_preds.shape[0] 大約 = len(test_data) - horizon + 1

    # 6. 整體誤差評估
    mae, rmse, r2, mape = evaluate_forecast(all_trues, all_preds)

    # 7. 你若需要紀錄結果
    records = []
    records.append(["ETS_walkforward_h8", mae, rmse, r2, mape])
    print("records:", records)

    # 8. 結束
    time.sleep(2)

if __name__ == "__main__":
    main_ETS_walk_forward()
