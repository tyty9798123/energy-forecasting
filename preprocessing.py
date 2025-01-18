# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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