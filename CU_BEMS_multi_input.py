# main.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from math import sqrt
#from keras.layers import Layer
import keras.backend as K
import seaborn as sns
import datetime
import time


from preprocessing import (load_data, add_features, add_temp, add_emd, create_dataset, 
                           split_time_series_data, split_time_series_data_2parts, scaling, scaling_and_pca, scaling_and_ica, create_dataset_single_step, create_decoder_input)

from models import (build_model, build_model_LSTM, build_model_TCN, build_combined_model,
                    build_model_sensors, build_model_cnn, build_model_cnn_lstm, build_model_tcn_gru,
                    train_model, train_model_2, build_model_TCN_II, build_model_MLP, build_seq2seq_TCN, build_seq2seq_tcn_lstm)

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# def metrics(model, Y_test, X_test_scaled, purpose, forecast_horizon):
#     # 使用模型進行預測
#     predicted = model.predict(X_test_scaled)  # 預測結果形狀: [samples, forecast_horizon]
    
#     # 儲存預測和真實值以供分析
#     save_data(Y_test, predicted, purpose=purpose)
    
#     # 初始化度量指標總和
#     total_mae = 0
#     total_rmse = 0
#     total_r2 = 0
#     total_mape = 0  # 新增 MAPE 總和的變量
    
#     # 計算每個步長的指標
#     for step in range(forecast_horizon):
#         step_mae = mean_absolute_error(Y_test[:, step], predicted[:, step])
#         step_mse = mean_squared_error(Y_test[:, step], predicted[:, step])
#         step_rmse = sqrt(step_mse)
#         step_r2 = r2_score(Y_test[:, step], predicted[:, step])
#         # 計算每一步的 MAPE
#         step_mape = mean_absolute_percentage_error(Y_test[:, step], predicted[:, step])
        
#         total_mae += step_mae
#         total_rmse += step_rmse
#         total_r2 += step_r2
#         total_mape += step_mape  # 累加 MAPE
    
#     # 計算所有預測步長的平均值
#     mae = total_mae / forecast_horizon
#     rmse = total_rmse / forecast_horizon
#     r2 = total_r2 / forecast_horizon
#     mape = total_mape / forecast_horizon  # 平均 MAPE
    
#     print(f'Mean Absolute Error (平均): {mae}')
#     print(f'Root Mean Squared Error (平均): {rmse}')
#     print(f'R^2 Score (平均): {r2}')
#     print(f'Mean Absolute Percentage Error (平均): {mape}')  # 輸出 MAPE
    
#     return mae, rmse, r2, mape

def metrics(model, Y_test, X_test_scaled, purpose):
    """
    計算並輸出 MAE, RMSE, R2, MAPE 等指標。
    可以同時處理單步預測 (samples,) 與多步預測 (samples, horizon)。

    Parameters:
        model: 訓練好的模型，用於預測。
        Y_test: numpy array, shape 可能是 (samples,) 或 (samples, horizon)。
        X_test_scaled: numpy array, 模型需要的輸入 (已縮放)。
        purpose: str, 給 save_data() 用的檔案命名用途。
    """
    # 使用模型進行預測
    predicted = model.predict(X_test_scaled)  # 將得到 (samples,) 或 (samples, forecast_horizon)

    # 檢查形狀，如果是一維 (單步)，就 reshape 成 (samples, 1)
    if len(Y_test.shape) == 1:
        Y_test = Y_test.reshape(-1, 1)
    if len(predicted.shape) == 1:
        predicted = predicted.reshape(-1, 1)
    
    # 這樣就能保證 Y_test, predicted 都是 (samples, forecast_horizon)
    # 取得實際的 horizon
    forecast_horizon = Y_test.shape[1]
    
    # 儲存預測和真實值以供分析 (若你有維度檢查，也可以在這裡再做一次)
    save_data(Y_test, predicted, purpose=purpose)

    # 初始化度量指標總和
    total_mae = 0
    total_rmse = 0
    total_r2 = 0
    total_mape = 0

    # 計算每個步長的指標
    for step in range(forecast_horizon):
        step_mae = mean_absolute_error(Y_test[:, step], predicted[:, step])
        step_mse = mean_squared_error(Y_test[:, step], predicted[:, step])
        step_rmse = sqrt(step_mse)
        step_r2 = r2_score(Y_test[:, step], predicted[:, step])
        step_mape = mean_absolute_percentage_error(Y_test[:, step], predicted[:, step])
        
        total_mae += step_mae
        total_rmse += step_rmse
        total_r2 += step_r2
        total_mape += step_mape
    
    # 計算所有預測步長的平均值
    mae = total_mae / forecast_horizon
    rmse = total_rmse / forecast_horizon
    r2 = total_r2 / forecast_horizon
    mape = total_mape / forecast_horizon
    
    print(f'Mean Absolute Error (平均): {mae}')
    print(f'Root Mean Squared Error (平均): {rmse}')
    print(f'R^2 Score (平均): {r2}')
    print(f'Mean Absolute Percentage Error (平均): {mape}')
    
    return mae, rmse, r2, mape


def metrics_2(model, Y_test, X_test_scaled):
    predicted = model.predict([X_test_scaled,X_test_scaled]).reshape(-1, 1)
    save_data(Y_test, predicted)
    mse = mean_squared_error(Y_test, predicted)
    mae = mean_absolute_error(Y_test, predicted)
    rmse = sqrt(mse)
    r2 = r2_score(Y_test, predicted)
    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R^2 Score: {r2}')
    return mae, rmse, r2

def plot_loss(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_results(model, Y_test, X_test_scaled):
    predicted = model.predict(X_test_scaled).reshape(-1, 1)
    plt.figure(figsize=(10, 4))
    plt.plot(Y_test, label='Real Value')
    plt.plot(predicted, label='Predicted Value', alpha=0.7)
    plt.title('Real vs Predicted Value')
    plt.ylabel('Value')
    plt.xlabel('Time Point')
    plt.legend()
    plt.show()

# def save_data(y_true, y_predict, purpose, filename=None):
#     """
#     Save multi-step prediction results to a CSV file.

#     Parameters:
#         y_true: numpy array, true target values (shape: [samples, forecast_horizon]).
#         y_predict: numpy array, predicted values (shape: [samples, forecast_horizon]).
#         purpose: str, purpose description for the file name.
#         filename: str or None, file name for saving the data.
#     """
#     if filename is None:
#         now = datetime.datetime.now()
#         filename = purpose + "_" + now.strftime("%m-%d_%H_%M") + '.csv'
    
#     # Prepare a DataFrame for multi-step predictions
#     df = pd.DataFrame()
#     forecast_horizon = y_true.shape[1]  # Number of forecast steps
    
#     for step in range(forecast_horizon):
#         df[f'y_true_step_{step+1}'] = y_true[:, step]
#         df[f'y_predict_step_{step+1}'] = y_predict[:, step]
    
#     # Save the DataFrame to a CSV file
#     df.to_csv(f"records/{filename}", index=False)

#     print(f"Data saved to records/{filename}")

def save_data(y_true, y_predict, purpose, filename=None):
    """
    Save multi-step prediction results to a CSV file.

    Parameters:
        y_true: numpy array, true target values (shape: [samples, forecast_horizon]).
        y_predict: numpy array, predicted values (shape: [samples, forecast_horizon]).
        purpose: str, purpose description for the file name.
        filename: str or None, file name for saving the data.
    """
    if filename is None:
        now = datetime.datetime.now()
        filename = purpose + "_" + now.strftime("%m-%d_%H_%M") + '.csv'
    
    # Prepare a DataFrame for multi-step predictions
    df = pd.DataFrame()
    forecast_horizon = y_true.shape[1]  # Number of forecast steps
    
    for step in range(forecast_horizon):
        df[f'y_true_step_{step+1}'] = y_true[:, step]
        df[f'y_predict_step_{step+1}'] = y_predict[:, step]
    
    # Save the DataFrame to a CSV file
    df.to_csv(f"records/{filename}", index=False)

    print(f"Data saved to records/{filename}")


import matplotlib.pyplot as plt
import numpy as np

def plot_results_per_step(y_true, y_predict, forecast_horizon, sample_range=None):
    """
    Plot the true vs. predicted values for each step in multi-step forecasting,
    with support for selecting a range of samples by percentage.

    Parameters:
        y_true: numpy array, true target values (shape: [samples, forecast_horizon]).
        y_predict: numpy array, predicted values (shape: [samples, forecast_horizon]).
        forecast_horizon: int, number of steps in the forecast.
        sample_range: tuple (start_percent, end_percent) or None, 
                      range of samples to display as a percentage (e.g., (50, 60)).
                      If None, all samples are displayed.
    """
    num_samples = y_true.shape[0]  # Total number of samples
    
    # Determine start and end indices based on percentage range
    if sample_range:
        start_idx = int(num_samples * (sample_range[0] / 100))
        end_idx = int(num_samples * (sample_range[1] / 100))
    else:
        start_idx = 0
        end_idx = num_samples  # Default: all samples
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(num_samples, end_idx)
    
    # Subset the data based on the specified range
    y_true_subset = y_true[start_idx:end_idx, :]
    y_predict_subset = y_predict[start_idx:end_idx, :]
    
    # Plot each forecast step
    for step in range(forecast_horizon):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true_subset[:, step], label="True")#, marker='o', linestyle='-', alpha=0.7)
        plt.plot(y_predict_subset[:, step], label="Predicted")#, marker='x', linestyle='--', alpha=0.7)
        plt.title(f"Step {step + 1}: True vs Predicted (Samples {start_idx}-{end_idx})")
        plt.xlabel("Sample Index (Relative to Range)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_results_offset(y_true, y_predict, forecast_horizon, sample_range=None):
    """
    Plot multi-step forecasting with offset so that each step is aligned with its actual time index.

    Parameters:
        y_true: numpy array, shape: [samples, forecast_horizon]
        y_predict: numpy array, shape: [samples, forecast_horizon]
        forecast_horizon: int, number of steps in the forecast
        sample_range: tuple (start_percent, end_percent) or None
    """
    num_samples = y_true.shape[0]
    
    # Determine sample indices
    if sample_range:
        start_idx = int(num_samples * (sample_range[0] / 100))
        end_idx   = int(num_samples * (sample_range[1] / 100))
    else:
        start_idx = 0
        end_idx   = num_samples

    # Boundary check
    start_idx = max(0, start_idx)
    end_idx   = min(num_samples, end_idx)

    # Subset data
    y_true_subset    = y_true[start_idx:end_idx, :]
    y_predict_subset = y_predict[start_idx:end_idx, :]

    plt.figure(figsize=(12, 8))
    
    # 每一個 step 會在時間軸上「右移」 step 個單位
    # 讓第 i 步（預測 t+i）對應到它實際在時間序列的位置
    for step in range(forecast_horizon):
        x_indices = np.arange(start_idx + step, end_idx + step)
        
        plt.plot(
            x_indices, 
            y_true_subset[:, step], 
            label=f"True (Step {step+1})", 
            linestyle='-', marker='o', alpha=0.7
        )
        plt.plot(
            x_indices, 
            y_predict_subset[:, step], 
            label=f"Pred (Step {step+1})", 
            linestyle='--', marker='x', alpha=0.7
        )

    plt.title(f"Offset Plot: True vs. Predicted (Samples {start_idx}-{end_idx})")
    plt.xlabel("Time Index (Offset by Step)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_heatmap(y_true, y_predict, forecast_horizon, sample_range=None):
    """
    Plot a 2D heatmap of the absolute error for each (sample, step) pair.

    Parameters:
        y_true: numpy array, shape = [samples, forecast_horizon]
        y_predict: numpy array, shape = [samples, forecast_horizon]
        forecast_horizon: int, number of forecast steps.
        sample_range: tuple (start_percent, end_percent) or None,
                      e.g., (50, 60) means only plot the middle 10% of samples.
    """
    num_samples = y_true.shape[0]
    
    # 1) Determine start and end indices based on percentage range
    if sample_range:
        start_idx = int(num_samples * (sample_range[0] / 100))
        end_idx   = int(num_samples * (sample_range[1] / 100))
    else:
        start_idx = 0
        end_idx   = num_samples
    
    # 2) Boundary check
    start_idx = max(0, start_idx)
    end_idx   = min(num_samples, end_idx)
    
    # 3) Subset data
    y_true_subset    = y_true[start_idx:end_idx, :]
    y_predict_subset = y_predict[start_idx:end_idx, :]
    
    # 4) Compute absolute error
    errors = np.abs(y_true_subset - y_predict_subset)  # shape: (subset_samples, forecast_horizon)
    
    # 5) Plot heatmap
    plt.figure(figsize=(10, 6))
    # imshow 預設是 (row, col)，所以我們會把「y 軸 = step」、「x 軸 = sample index」
    # 這裡 transpose 一下，讓 row = step, col = sample index
    plt.imshow(errors.T, aspect='auto', cmap='hot', origin='lower')
    
    # 6) Colorbar 方便查看誤差數值
    cbar = plt.colorbar()
    cbar.set_label('Absolute Error')
    
    # 7) 設定座標軸標籤
    plt.title(f"Error Heatmap (Samples {start_idx}-{end_idx}, Steps=1..{forecast_horizon})")
    plt.xlabel("Sample Index (Relative to Range)")
    plt.ylabel("Forecast Step")
    
    # 8) 讓 y 軸刻度對應到 step
    plt.yticks(
        np.arange(forecast_horizon), 
        [f"Step {s+1}" for s in range(forecast_horizon)]
    )
    
    # 9) Tight layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    len_of_scale = len(df.columns)
    df = add_features(df)
    df = add_temp(df)

    records = []
    horizon = 8
    exec_time = 1

    for i in range(exec_time):
        for look_back in [48]:
            print(f"Current Look Back: {look_back}")
            X, y = create_dataset(df.values, look_back, h=horizon)
            #X, y = create_dataset_single_step(df.values, look_back, step=8)
            
            length = len(X)
            for percent in [100]:
                print(f"Percent: {percent}%")
                index_percent = int(length * percent * 0.01)
                new_X = X[:index_percent]
                new_y = y[:index_percent]

                print(f"該批次總長度: {len(new_X)}")
                # 固定測試資料長度
                X_test, Y_test = new_X[-int(length*0.1):], new_y[-int(length*0.1):]
                # 訓練和驗證資料
                X_train, Y_train = new_X[:-int(length*0.1)], new_y[:-int(length*0.1)]
                train_ratio = 0.85
                X_train, Y_train, X_validation, Y_validation = split_time_series_data_2parts(X_train, Y_train, train_ratio)
                n_features = X_test.shape[2]
                # 假設 Y_train, Y_validation, Y_test 已經準備好
 

                print(f"train:{X_train.shape}")
                print(f"validation:{X_validation.shape}")
                print(f"test:{X_test.shape}")

                if (len(X_train)+len(X_validation)+len(X_test)) == len(new_X):
                    print("分割正確.")
                else:
                    print("分割不正確.")
                    break

                X_train_scaled, X_validation_scaled, X_test_scaled = scaling(X_train, X_validation, X_test, len_of_scale)
                # X_train_scaled, X_validation_scaled, X_test_scaled, scaler, pca_model = scaling_and_pca(
                #     X_train, 
                #     X_validation, 
                #     X_test,
                #     n_components=18
                # )
                # X_train_scaled, X_validation_scaled, X_test_scaled, scaler, pca_model = scaling_and_ica(
                #     X_train, 
                #     X_validation, 
                #     X_test,
                #     n_components=18
                # )
                X_dec_train = create_decoder_input(Y_train, target_features=X_train_scaled.shape[2])
                X_dec_validation = create_decoder_input(Y_validation, target_features=X_train_scaled.shape[2])
                X_dec_test = create_decoder_input(Y_test, target_features=X_train_scaled.shape[2])  # 若需要用于推斷階段教師強制

                #model = build_model_LSTM(look_back, X_train_scaled.shape[2], h=horizon)
                #model = build_model_MLP(look_back, X_train_scaled.shape[2], h=horizon)
                #model = build_model_TCN(look_back, X_train_scaled.shape[2], h=horizon) # look back, features, h
                #model = build_model_tcn_gru(look_back, X_train_scaled.shape[2], h=horizon) # look back, features, h
                #model = build_seq2seq_TCN(look_back, X_train_scaled.shape[2], h=horizon) # look back, features, h
                model = build_seq2seq_tcn_lstm(look_back, X_train_scaled.shape[2], 
                                h=horizon,
                                latent_dim=64,
                                nb_filters=32,
                                kernel_size=3)

                #model = build_model_sensors(look_back, n_features, h=horizon)
                #history = train_model(model, X_train_scaled, Y_train, X_validation_scaled, Y_validation)
                history = model.fit(
                    [X_train_scaled, X_dec_train],  # Encoder 和 Decoder 輸入
                    Y_train, 
                    epochs=500,
                    batch_size=128, 
                    validation_data=([X_validation_scaled, X_dec_validation], Y_validation)
                )
                # save results of training data
                mae, rmse, r2, mape = metrics(model, Y_test, [X_test_scaled, X_dec_test], purpose="Test")#, forecast_horizon=horizon)
                #metrics(model, Y_train, X_train_scaled, purpose="Train")#, forecast_horizon=horizon)
                #metrics(model, Y_validation, X_validation_scaled, purpose="Validation")#, forecast_horizon=horizon)

                #plot_results_per_step(Y_test, model.predict(X_test_scaled), forecast_horizon=horizon, sample_range=(85, 100))
                #plot_results_offset(Y_test, model.predict(X_test_scaled), forecast_horizon=horizon, sample_range=(85, 100))
                #plot_error_heatmap(Y_test, model.predict(X_test_scaled), forecast_horizon=horizon, sample_range=(85, 100))
                
                records.append([look_back, percent, mae, rmse, r2, mape])
                print(records)
                time.sleep(95)