import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data from pickle
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Date Sorting
data['date'] = pd.to_datetime(data['date'])


features = ['close_lag_1', 'close_lag_2', 'rsi', 'rolling_mean_7',
            'rolling_std_7', 'rolling_mean_30', 'close_diff']
target = 'close'

#Split data
train_df = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2023-05-17')]
test_df = data[data['date'] > '2023-05-17']


if train_df.shape[0] < 5 or test_df.shape[0] < 5:
    warnings.warn("Data untuk pelatihan atau pengujian tidak mencukupi setelah preprocessing. Model mungkin tidak akan memberikan hasil yang akurat.")

train_features = train_df[features]
test_features = test_df[features]
train_target = train_df[target]
test_target = test_df[target]

#train
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')


if train_features.shape[0] > 0:
    svr_model.fit(train_features, train_target)

    #prediksi
    if test_features.shape[0] > 0:
        svr_pred = svr_model.predict(test_features)
        svr_pred_exp = np.expm1(svr_pred)
        test_target_exp = np.expm1(test_target)

      
        mae = mean_absolute_error(test_target_exp, svr_pred_exp)
        mse = mean_squared_error(test_target_exp, svr_pred_exp)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_target_exp - svr_pred_exp) / test_target_exp)) * 100
        accuracy = 100 - mape

        results = {
            'actual': test_target_exp.reset_index(drop=True),
            'predicted': svr_pred_exp,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'accuracy': accuracy,
            'best_params': {'C': 100, 'epsilon': 0.1, 'gamma': 'scale'}
        }

        
        with open('svr_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    else:
        warnings.warn("Data uji tidak mencukupi untuk melakukan prediksi.")
else:
    warnings.warn("Data latih tidak mencukupi untuk melatih model SVR.")
