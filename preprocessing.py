# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load_data():
    df = pd.read_csv("data/dt_value_SmartBuilding_hourly.csv")
    df.set_index("Date", inplace=True)
    return df

def add_features(df):
    df.index = pd.to_datetime(df.index)
    df['day_of_week'] = df.index.weekday
    df['hour_of_day'] = df.index.hour
    df = pd.get_dummies(df, columns=['day_of_week', 'hour_of_day'])

    holidays = holidays = ['2018-01-01', '2018-01-02', '2018-01-06', '2018-01-07', '2018-01-13', '2018-01-14', '2018-01-16', '2018-01-20', '2018-01-21', '2018-01-27', '2018-01-28', '2018-02-03', '2018-02-04', '2018-02-10', '2018-02-11', '2018-02-14', '2018-02-16', '2018-02-17', '2018-02-18', '2018-02-24', '2018-02-25', '2018-03-01', '2018-03-03', '2018-03-04', '2018-03-10', '2018-03-11', '2018-03-17', '2018-03-18', '2018-03-20', '2018-03-24', '2018-03-25', '2018-03-31', '2018-04-01', '2018-04-06', '2018-04-07', '2018-04-08', '2018-04-12', '2018-04-13', '2018-04-14', '2018-04-15', '2018-04-16', '2018-04-21', '2018-04-22', '2018-04-28', '2018-04-29', '2018-05-01', '2018-05-05', '2018-05-06', '2018-05-12', '2018-05-13', '2018-05-14', '2018-05-19', '2018-05-20', '2018-05-26', '2018-05-27', '2018-05-29', '2018-06-02', '2018-06-03', '2018-06-09', '2018-06-10', '2018-06-16', '2018-06-17', '2018-06-21', '2018-06-23', '2018-06-24', '2018-06-30', '2018-07-01', '2018-07-07', '2018-07-08', '2018-07-14', '2018-07-15', '2018-07-21', '2018-07-22', '2018-07-27', '2018-07-28', '2018-07-29', '2018-07-30', '2018-08-04', '2018-08-05', '2018-08-11', '2018-08-12', '2018-08-13', '2018-08-18', '2018-08-19', '2018-08-25', '2018-08-26', '2018-09-01', '2018-09-02', '2018-09-08', '2018-09-09', '2018-09-15', '2018-09-16', '2018-09-22', '2018-09-23', '2018-09-29', '2018-09-30', '2018-10-06', '2018-10-07', '2018-10-13', '2018-10-14', '2018-10-15', '2018-10-20', '2018-10-21', '2018-10-23', '2018-10-27', '2018-10-28', '2018-10-31', '2018-11-03', '2018-11-04', '2018-11-10', '2018-11-11', '2018-11-17', '2018-11-18', '2018-11-22', '2018-11-24', '2018-11-25', '2018-12-01', '2018-12-02', '2018-12-05', '2018-12-08', '2018-12-09', '2018-12-10', '2018-12-15', '2018-12-16', '2018-12-22', '2018-12-23', '2018-12-24', '2018-12-25', '2018-12-29', '2018-12-30', '2018-12-31', '2019-01-01', '2019-01-05', '2019-01-06', '2019-01-12', '2019-01-13', '2019-01-16', '2019-01-19', '2019-01-20', '2019-01-26', '2019-01-27', '2019-02-02', '2019-02-03', '2019-02-05', '2019-02-06', '2019-02-07', '2019-02-09', '2019-02-10', '2019-02-14', '2019-02-16', '2019-02-17', '2019-02-19', '2019-02-23', '2019-02-24', '2019-03-02', '2019-03-03', '2019-03-09', '2019-03-10', '2019-03-16', '2019-03-17', '2019-03-21', '2019-03-23', '2019-03-24', '2019-03-30', '2019-03-31', '2019-04-06', '2019-04-07', '2019-04-08', '2019-04-12', '2019-04-13', '2019-04-14', '2019-04-15', '2019-04-16', '2019-04-20', '2019-04-21', '2019-04-27', '2019-04-28', '2019-05-01', '2019-05-04', '2019-05-05', '2019-05-06', '2019-05-09', '2019-05-11', '2019-05-12', '2019-05-18', '2019-05-19', '2019-05-20', '2019-05-25', '2019-05-26', '2019-06-01', '2019-06-02', '2019-06-03', '2019-06-08', '2019-06-09', '2019-06-15', '2019-06-16', '2019-06-21', '2019-06-22', '2019-06-23', '2019-06-29', '2019-06-30', '2019-07-06', '2019-07-07', '2019-07-13', '2019-07-14', '2019-07-16', '2019-07-17', '2019-07-20', '2019-07-21', '2019-07-27', '2019-07-28', '2019-07-29', '2019-08-03', '2019-08-04', '2019-08-10', '2019-08-11', '2019-08-12', '2019-08-17', '2019-08-18', '2019-08-24', '2019-08-25', '2019-08-31', '2019-09-01', '2019-09-07', '2019-09-08', '2019-09-14', '2019-09-15', '2019-09-21', '2019-09-22', '2019-09-23', '2019-09-28', '2019-09-29', '2019-10-05', '2019-10-06', '2019-10-12', '2019-10-13', '2019-10-14', '2019-10-19', '2019-10-20', '2019-10-23', '2019-10-26', '2019-10-27', '2019-10-31', '2019-11-02', '2019-11-03', '2019-11-04', '2019-11-05', '2019-11-09', '2019-11-10', '2019-11-11', '2019-11-16', '2019-11-17', '2019-11-23', '2019-11-24', '2019-11-30', '2019-12-01', '2019-12-05', '2019-12-07', '2019-12-08', '2019-12-10', '2019-12-14', '2019-12-15', '2019-12-21', '2019-12-22', '2019-12-24', '2019-12-25', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31']  # 原本的假日清單放這裡
    holidays_set = set(pd.to_datetime(holidays).date)
    def is_holiday(date):
        return 1 if date.date() in holidays_set else 0
    df['holiday'] = df.index.to_series().apply(is_holiday)
    return df

def add_temp(df):
    temp_data = pd.read_pickle('data/df_weather.pickle.gz')
    temp_data = temp_data.append(pd.DataFrame({
        'Avg Temp': [16],
        'Min Temp': [11],
        'Max Temp': [21],
        'Precip': [6]
    }, index=[pd.to_datetime('2020-01-01 00:00:00')]))

    df.index = pd.to_datetime(df.index)
    temp_data.index = pd.to_datetime(temp_data.index)
    temp_data = temp_data.resample('H').ffill()["2018-07-01":]
    temp_data = temp_data[:-1]
    temp_data = temp_data.drop(["Precip"], axis=1)

    merged_data = pd.concat([df, temp_data], axis=1)
    merged_data.to_csv("data/n.csv")
    return merged_data

def add_emd(df):
    imfs = pd.read_csv("data/df_with_imf.csv")
    imfs = imfs.set_index("DateTime")
    imfs = imfs.drop(["value"], axis=1)
    # 刪除IMF_4 ~ IMF_13
    imfs = imfs.drop([f"IMF_{i}" for i in range(4, 14)], axis=1)
    result_df = df.join(imfs, how='inner')
    return result_df

def create_dataset(data, look_back=1, h=1):
    """
    Create dataset for multi-step forecasting.

    Parameters:
        data: numpy array, the input dataset (e.g., time series data).
        look_back: int, the number of past time steps to use as input (X).
        forecast_horizon: int, the number of future time steps to predict (Y).

    Returns:
        X: numpy array, input features with shape (samples, look_back, features).
        Y: numpy array, target labels with shape (samples, forecast_horizon).
    """
    X, Y = [], []
    for i in range(len(data) - look_back - h + 1):
        # Input sequence of length `look_back`
        X.append(data[i:(i + look_back), :])
        # Target sequence of length `forecast_horizon`
        Y.append(data[i + look_back:i + look_back + h, 0])  # Assuming we predict only the first feature
    return np.array(X), np.array(Y)

def create_decoder_input(target_sequence, target_features=36):
    """
    根據目標序列創建 Decoder 輸入，並將特徵維度擴展到 target_features。
    """
    # 如果目標序列是2維的，擴展為三維 (samples, horizon, 1)
    if target_sequence.ndim == 2:
        target_sequence = target_sequence[..., np.newaxis]

    samples, horizon, current_features = target_sequence.shape

    # 如果目前特徵數小於目標特徵數，則進行擴展（例如重複最後一個特徵或填充0）
    if current_features < target_features:
        # 這裡簡單地重複現有特徵以達到 target_features
        repeats = target_features // current_features
        remainder = target_features % current_features
        expanded = np.concatenate([target_sequence] * repeats + [target_sequence[:, :, :remainder]], axis=2)
    else:
        expanded = target_sequence[:, :, :target_features]

    # 使用擴展後的特徵數量
    _, _, n_features = expanded.shape

    # 創建起始標記，其形狀為 (samples, 1, n_features)
    start_token = np.zeros((samples, 1, n_features))

    # 向右移動並在最前面加入 start_token
    decoder_input = np.concatenate([start_token, expanded[:, :-1, :]], axis=1)
    return decoder_input



def create_dataset_single_step(data, look_back=1, step=1):
    """
    Create dataset for single-step forecasting, but the step can be
    1-step ahead, 2-step ahead, ... etc.

    Parameters:
        data: numpy array, the input dataset (time series data).
              shape: (total_samples, features)
        look_back: int, the number of past time steps to use as input (X).
        step: int, which future step to predict (e.g., step=1 means predict t+1,
              step=2 means predict t+2, etc.)

    Returns:
        X: numpy array, shape (samples, look_back, features)
        Y: numpy array, shape (samples,) or (samples, 1)
           (Here we only predict the first feature, but you can modify as needed.)
    """
    X, Y = [], []
    # 迴圈終止條件：確保在最尾端也能取到 step 的那個未來值
    for i in range(len(data) - look_back - step + 1):
        # 收集過去 look_back 個時間點的所有特徵
        X.append(data[i : i + look_back, :])
        # 預測目標：距離現在 step 步之後的第一個 feature
        Y.append(data[i + look_back + step - 1, 0])
    # 轉成 numpy array
    return np.array(X), np.array(Y)


def split_time_series_data(X, Y, train_ratio, validation_ratio, test_ratio):
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)
    test_size = total_size - train_size - validation_size

    X_train = X[:train_size]
    Y_train = Y[:train_size]

    X_validation = X[train_size:train_size + validation_size]
    Y_validation = Y[train_size:train_size + validation_size]

    X_test = X[-test_size:]
    Y_test = Y[-test_size:]

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def split_time_series_data_2parts(X, Y, train_ratio):
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_validation = X[train_size:]
    Y_validation = Y[train_size:]
    return X_train, Y_train, X_validation, Y_validation

def scaling(X_train, X_validation, X_test, num_features=4):
    X_train_scaled = np.copy(X_train)
    X_validation_scaled = np.copy(X_validation)
    X_test_scaled = np.copy(X_test)
    for i in range(num_features):
        scaler = StandardScaler()
        X_train_feature = X_train[:, :, i] 
        X_train_feature_scaled = scaler.fit_transform(X_train_feature.reshape(-1, 1)).reshape(X_train_feature.shape)
        X_train_scaled[:, :, i] = X_train_feature_scaled  
        X_validation_feature = X_validation[:, :, i]
        X_validation_feature_scaled = scaler.transform(X_validation_feature.reshape(-1, 1)).reshape(X_validation_feature.shape)
        X_validation_scaled[:, :, i] = X_validation_feature_scaled
        X_test_feature = X_test[:, :, i]
        X_test_feature_scaled = scaler.transform(X_test_feature.reshape(-1, 1)).reshape(X_test_feature.shape)
        X_test_scaled[:, :, i] = X_test_feature_scaled
    return X_train_scaled, X_validation_scaled, X_test_scaled


def add_gaussian_noise(Y_train, noise_level=0.1, seed=None):
    """
    為 Y_train 加入高斯 (Gaussian) 雜訊。
    
    參數：
    - Y_train     : np.ndarray，原始目標值，形狀不限
    - noise_level : 浮點數，控制雜訊的標準差大小，預設為 0.1
    - seed        : 整數，亂數種子，若需要重複實驗可固定此值；若不需要則保持 None
    
    回傳：
    - Y_train_noisy : np.ndarray，加入雜訊後的目標值
    """
    # 如果想讓每次產生的雜訊都一致，可以設定 seed
    if seed is not None:
        np.random.seed(seed)
    
    # 根據 Y_train 的形狀產生高斯雜訊
    noise = np.random.normal(loc=0.0, scale=noise_level, size=Y_train.shape)
    
    # 加入雜訊
    Y_train_noisy = Y_train + noise
    
    return Y_train_noisy


def scaling_auto(X_train, X_validation, X_test):
    # 复制原始数据以避免修改原数组
    X_train_scaled = np.copy(X_train)
    X_validation_scaled = np.copy(X_validation)
    X_test_scaled = np.copy(X_test)
    
    # 获取特征维度数量
    num_features = X_train.shape[2]
    numeric_feature_indices = []

    # 自动检测数值型特征（非 one-hot 特征）
    for i in range(num_features):
        # 获取训练集中第 i 个特征的所有唯一值
        unique_vals = np.unique(X_train[:, :, i])
        # 如果不全是 0 和 1，则认为是数值型特征
        if not set(unique_vals).issubset({0, 1}):
            numeric_feature_indices.append(i)

    # 对检测到的数值型特征进行标准化
    for i in numeric_feature_indices:
        scaler = StandardScaler()
        
        # 标准化训练集特征
        X_train_feature = X_train[:, :, i]
        X_train_feature_scaled = scaler.fit_transform(
            X_train_feature.reshape(-1, 1)
        ).reshape(X_train_feature.shape)
        X_train_scaled[:, :, i] = X_train_feature_scaled
        
        # 使用训练集拟合的参数标准化验证集特征
        X_validation_feature = X_validation[:, :, i]
        X_validation_feature_scaled = scaler.transform(
            X_validation_feature.reshape(-1, 1)
        ).reshape(X_validation_feature.shape)
        X_validation_scaled[:, :, i] = X_validation_feature_scaled
        
        # 使用训练集拟合的参数标准化测试集特征
        X_test_feature = X_test[:, :, i]
        X_test_feature_scaled = scaler.transform(
            X_test_feature.reshape(-1, 1)
        ).reshape(X_test_feature.shape)
        X_test_scaled[:, :, i] = X_test_feature_scaled

    # 未列入 numeric_feature_indices 的特征（如 one-hot）保持原样
    return X_train_scaled, X_validation_scaled, X_test_scaled

def scaling_and_pca(X_train, X_validation, X_test, 
                    n_components=10, 
                    scaler=None, 
                    pca=None):
    """
    同時對數據進行縮放 + PCA 降維，
    保留最後一維特徵從原本的維度降到 n_components。
    
    參數：
    --------
    X_train : ndarray, shape (n_samples, n_timesteps, n_features)
    X_validation : ndarray, shape (n_samples_val, n_timesteps, n_features)
    X_test : ndarray, shape (n_samples_test, n_timesteps, n_features)
    n_components : int, PCA 最終想保留的特徵維度 (預設 10)
    scaler : Scaler 物件，若傳入 None，則預設使用 StandardScaler()
    pca : PCA 物件，若傳入 None，則會在此函式內 new 一個 PCA(n_components=...)
           如果需要在多處使用相同 PCA，或想查看保留方差，可傳入自定義的 PCA 物件

    回傳：
    --------
    X_train_scaled : ndarray, shape (n_samples, n_timesteps, n_components)
    X_validation_scaled : ndarray, shape (n_samples_val, n_timesteps, n_components)
    X_test_scaled : ndarray, shape (n_samples_test, n_timesteps, n_components)
    scaler : 縮放器物件 (可能用於後續或查看特徵縮放參數)
    pca : PCA 物件 (可用於查看各主成分的解釋度等)
    """
    
    # 1) 如果呼叫者沒提供 scaler，預設用 StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        
    # 2) 如果呼叫者沒提供 pca，則新建一個
    if pca is None:
        pca = PCA(n_components=n_components)
    
    # 取得 shape
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_validation.shape[0]
    n_samples_test = X_test.shape[0]
    
    # =============== 先做 Reshape 成 2D ===============
    #  (samples * timesteps, features)
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_validation.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    # =============== 先做 Scaling ===============
    # fit_transform 用於訓練資料
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    # validation, test 直接 transform
    X_val_scaled_2d = scaler.transform(X_val_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # =============== 接著做 PCA 降維 ===============
    # PCA 對訓練資料擬合 + 轉換
    X_train_pca_2d = pca.fit_transform(X_train_scaled_2d) 
    # validation, test 直接 transform
    X_val_pca_2d = pca.transform(X_val_scaled_2d)
    X_test_pca_2d = pca.transform(X_test_scaled_2d)

    # =============== Reshape 回 3D ===============
    # (samples, timesteps, n_components)
    X_train_scaled = X_train_pca_2d.reshape(n_samples_train, n_timesteps, n_components)
    X_validation_scaled = X_val_pca_2d.reshape(n_samples_val, n_timesteps, n_components)
    X_test_scaled = X_test_pca_2d.reshape(n_samples_test, n_timesteps, n_components)

    return X_train_scaled, X_validation_scaled, X_test_scaled, scaler, pca

from sklearn.decomposition import FastICA

def scaling_and_ica(X_train, X_validation, X_test,
                    n_components=10,
                    scaler=None,
                    ica=None):
    """
    先做縮放，再用 ICA 對資料進行降維 (或成分分解)。
    最終維度會降到 n_components。
    """
    # 1) 如果呼叫者沒提供 scaler，預設用 StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        
    # 2) 如果呼叫者沒提供 ica，就建一個 FastICA
    if ica is None:
        ica = FastICA(n_components=n_components, random_state=0)
    
    # 取得 shape
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_validation.shape[0]
    n_samples_test = X_test.shape[0]
    
    # =============== 先展平成 2D ===============
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_validation.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    # =============== 先做 Scaling ===============
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_val_scaled_2d = scaler.transform(X_val_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # =============== 接著做 ICA ===============
    X_train_ica_2d = ica.fit_transform(X_train_scaled_2d)
    X_val_ica_2d = ica.transform(X_val_scaled_2d)
    X_test_ica_2d = ica.transform(X_test_scaled_2d)

    # =============== Reshape 回 3D ===============
    X_train_ica = X_train_ica_2d.reshape(n_samples_train, n_timesteps, n_components)
    X_val_ica = X_val_ica_2d.reshape(n_samples_val, n_timesteps, n_components)
    X_test_ica = X_test_ica_2d.reshape(n_samples_test, n_timesteps, n_components)

    return X_train_ica, X_val_ica, X_test_ica, scaler, ica