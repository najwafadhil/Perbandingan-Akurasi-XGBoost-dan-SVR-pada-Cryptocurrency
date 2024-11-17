import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)


data['date'] = pd.to_datetime(data['date'])


required_lags = ['close_lag_3', 'close_lag_4', 'close_lag_5']
for lag in required_lags:
    if lag not in data.columns:
        lag_num = int(lag.split('_')[-1])
        data[lag] = data['close'].shift(lag_num)


data = data.dropna()


features = ['close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
            'rsi', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'close_diff']
target = 'close'


train_df = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2023-05-17')]
test_df = data[data['date'] > '2023-05-17']


if train_df.shape[0] < 5 or test_df.shape[0] < 5:
    warnings.warn("Data untuk pelatihan atau pengujian tidak mencukupi setelah preprocessing. Model mungkin tidak akan memberikan hasil yang akurat.")

train_features = train_df[features]
test_features = test_df[features]
train_target = train_df[target]
test_target = test_df[target]


if train_features.shape[0] > 0:
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    if test_features.shape[0] > 0:
        test_features = scaler.transform(test_features)
    else:
        warnings.warn("Data uji tidak mencukupi untuk dilakukan scaling.")
else:
    warnings.warn("Data latih tidak mencukupi untuk dilakukan scaling.")


if train_features.shape[0] > 0:
    best_params = {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'min_child_weight': 3,
        'booster': 'gbtree'
    }

    xgb = XGBRegressor(objective='reg:squarederror', **best_params)
    xgb.fit(train_features, train_target, eval_set=[(test_features, test_target)], verbose=False)

    #prediksi
    if test_features.shape[0] > 0:
        xgb_pred = xgb.predict(test_features)

       
        mae = mean_absolute_error(test_target, xgb_pred)
        mse = mean_squared_error(test_target, xgb_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_target - xgb_pred) / (test_target + 1e-6))) * 100
        accuracy = 100 - mape

       
        results = {
            'actual': test_target.reset_index(drop=True),
            'predicted': xgb_pred,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'accuracy': accuracy,
            'best_params': best_params
        }

        
        with open('xgboost_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    