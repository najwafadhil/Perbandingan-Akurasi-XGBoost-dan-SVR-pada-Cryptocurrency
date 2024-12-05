import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile
import warnings
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from fpdf import FPDF
warnings.filterwarnings('ignore')

# Import CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply CSS styling (Make sure 'style.css' exists in the same directory)
if os.path.exists(os.path.join(os.path.dirname(__file__), 'style.css')):
    local_css(os.path.join(os.path.dirname(__file__), 'style.css'))

# Preprocessing Function
def preprocess_data(data):
    data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col], errors='ignore')

    # Feature lowercase
    data.columns = map(str.lower, data.columns)

    # Date
    data['date'] = pd.to_datetime(data['date'], dayfirst=True, errors='coerce')

    # Date Error handling
    data = data.dropna(subset=['date']).reset_index(drop=True)

    # Convert features to numerical
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop NaN
    data = data.dropna().reset_index(drop=True)

    # Normalize 'close'
    data['close'] = (data['close'] / 1e6) + 1e-6
    data['close'] = np.log1p(data['close'])

    # Sort by date
    data = data.sort_values(by='date').reset_index(drop=True)

    # Feature Engineering
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    data['rolling_mean_7'] = data['close'].rolling(window=7, min_periods=1).mean()
    data['rolling_std_7'] = data['close'].rolling(window=7, min_periods=1).std()
    data['rolling_mean_30'] = data['close'].rolling(window=30, min_periods=1).mean()

    # RSI
    delta = data['close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    # Replace zeros in avg_loss to avoid division by zero
    avg_loss.replace(0, np.nan, inplace=True)
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Backfill RSI missing values
    data['rsi'] = data['rsi'].bfill()

    # Add lag features
    data['close_lag_1'] = data['close'].shift(1)
    data['close_lag_2'] = data['close'].shift(2)
    data['close_diff'] = data['close'].diff().fillna(0)

    # Additional lag features for XGBoost
    data['close_lag_3'] = data['close'].shift(3)
    data['close_lag_4'] = data['close'].shift(4)
    data['close_lag_5'] = data['close'].shift(5)

    # Forward-fill remaining NaN values and drop rows with NaN
    data = data.ffill().dropna().reset_index(drop=True)

    if data.shape[0] < 5:
        st.warning("Data setelah preprocessing sangat terbatas. Menggunakan interpolasi untuk menambah jumlah data.")
        data = data.interpolate(method='linear').ffill().bfill().reset_index(drop=True)

    feature_columns = ['close', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rsi',
                       'close_lag_1', 'close_lag_2', 'close_diff', 'close_lag_3', 'close_lag_4', 'close_lag_5']
    if data.shape[0] > 0:
        scaler = MinMaxScaler()
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
    else:
        st.error("Data tidak mencukupi untuk dilakukan scaling. Pastikan dataset memiliki setidaknya satu sampel yang valid.")
        return None

    return data

# SVR Model Function
def run_svr(data):
    # Date Sorting
    data['date'] = pd.to_datetime(data['date'])

    features = ['close_lag_1', 'close_lag_2', 'rsi', 'rolling_mean_7',
                'rolling_std_7', 'rolling_mean_30', 'close_diff']
    target = 'close'

    # Split data
    train_df = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2023-05-17')]
    test_df = data[data['date'] > '2023-05-17']

    if train_df.shape[0] < 5 or test_df.shape[0] < 5:
        st.warning("Data untuk pelatihan atau pengujian tidak mencukupi setelah preprocessing.")
        return None

    train_features = train_df[features]
    test_features = test_df[features]
    train_target = train_df[target]
    test_target = test_df[target]

    # Train
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

    if train_features.shape[0] > 0:
        svr_model.fit(train_features, train_target)

        # Predict
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
            return results
        else:
            st.warning("Data uji tidak mencukupi untuk melakukan prediksi.")
            return None
    else:
        st.warning("Data latih tidak mencukupi untuk melatih model SVR.")
        return None

# XGBoost Model Function
def run_xgboost(data):
    # Date Sorting
    data['date'] = pd.to_datetime(data['date'])

    features = ['close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
                'rsi', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'close_diff']
    target = 'close'

    # Split data
    train_df = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2023-05-17')]
    test_df = data[data['date'] > '2023-05-17']

    if train_df.shape[0] < 5 or test_df.shape[0] < 5:
        st.warning("Data untuk pelatihan atau pengujian tidak mencukupi setelah preprocessing.")
        return None

    train_features = train_df[features]
    test_features = test_df[features]
    train_target = train_df[target]
    test_target = test_df[target]

    # Scaling
    if train_features.shape[0] > 0:
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        if test_features.shape[0] > 0:
            test_features = scaler.transform(test_features)
        else:
            st.warning("Data uji tidak mencukupi untuk dilakukan scaling.")
            return None
    else:
        st.warning("Data latih tidak mencukupi untuk dilakukan scaling.")
        return None

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

        # Predict
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
            return results
        else:
            st.warning("Data uji tidak mencukupi untuk melakukan prediksi.")
            return None
    else:
        st.warning("Data latih tidak mencukupi untuk melatih model XGBoost.")
        return None
# Fungsi untuk membuat laporan PDF
def generate_pdf(results, model_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Judul
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt=f"Laporan Akurasi Prediksi {model_name}", ln=True, align='C')

    # Tambahkan garis horizontal
    pdf.set_draw_color(0, 0, 0)  # Hitam
    pdf.line(10, 25, 200, 25)

    # Tambahkan hasil metrik
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"MAE: {results['mae']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"MSE: {results['mse']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"RMSE: {results['rmse']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"MAPE: {results['mape']:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Akurasi: {results['accuracy']:.2f}%", ln=True)

    # Tambahkan grafik jika ada
    if 'plot_image' in results:
        pdf.ln(10)
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, txt="Grafik Harga Aktual vs Prediksi", ln=True)
        pdf.ln(5)
        pdf.image(results['plot_image'], x=10, y=None, w=180)

    return pdf

# Tombol unduh untuk PDF
def download_pdf(results, model_name):
    pdf = generate_pdf(results, model_name)
    pdf_file = io.BytesIO()
    pdf.output(pdf_file)
    pdf_file.seek(0)
    return pdf_file

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar Navigation
st.sidebar.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://img.icons8.com/ios-filled/50/ffffff/combo-chart.png" width="20" style="margin-right: 10px;">
        <h3 style="margin: 0; color: white; font-size: 18px;">Perbandingan Akurasi Prediksi Cryptocurrency</h3>
    </div>
""", unsafe_allow_html=True)

def set_page(page):
    st.session_state.page = page

st.sidebar.button("Home", on_click=set_page, args=("Home",))
st.sidebar.button("Prediksi", on_click=set_page, args=("Prediksi",))
st.sidebar.button("About", on_click=set_page, args=("About",))

# Home Page
if st.session_state.page == "Home":
    st.markdown("""
    <h4 style="font-size: 28px;">
    Selamat Datang di Aplikasi Perbandingan Akurasi Prediksi XGBoost dan SVR dalam Memprediksi Harga Cryptocurrency
    </h4>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Extreme Gradient Boosting (XGBoost)")
        st.markdown("""
        <p style="text-align: justify;">
        XGBoost adalah salah satu algoritma yang bekerja dengan menggabungkan beberapa model keputusan sederhana (weak learners) untuk membentuk model yang lebih kuat. Proses boosting secara iteratif meningkatkan akurasi dengan mengurangi kesalahan prediksi dari model sebelumnya.
        </p>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("Support Vector Regression (SVR)")
        st.markdown("""
        <p style="text-align: justify;">
        Support Vector Regression (SVR) adalah algoritma supervised learning yang digunakan untuk memprediksi nilai variabel kontinu. SVR menggunakan prinsip yang sama dengan Support Vector Machine (SVM), tujuan dari SVR adalah menemukan hyperplane terbaik yang memprediksi hubungan antara input dan output dengan margin kesalahan minimal.
        </p>
        """, unsafe_allow_html=True)

    st.subheader("Contoh Data CSV")
    st.markdown("""
    <p>
    Untuk memudahkan Anda dalam menggunakan aplikasi ini, kami menyediakan contoh data CSV yang dapat Anda unduh dan gunakan.
    </p>
    """, unsafe_allow_html=True)

    # Reading sample CSV data
    try:
        sample_csv_path = os.path.join(os.path.dirname(__file__), 'Bitcoin.csv')
        with open(sample_csv_path, 'rb') as f:
            sample_csv = f.read()

        # Download button
        st.download_button(
            label='Download Contoh Data CSV',
            data=sample_csv,
            file_name='Bitcoin.csv',
            mime='text/csv'
        )
    except FileNotFoundError:
        st.error("Contoh data CSV tidak ditemukan. Pastikan file 'Bitcoin.csv' ada di direktori aplikasi.")

    # Disclaimer
    st.write("\n" * 3)
    st.markdown(
        """
        <div style="text-align: left;">
            <p style="font-size: 12px;">
            Pada penelitian ini data diambil dari sumber: <a href="https://coinmarketcap.com" target="_blank">coinmarketcap.com</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Prediction Page
elif st.session_state.page == "Prediksi":
    st.title("Perbandingan Akurasi Prediksi Harga Cryptocurrency")
    st.subheader("Upload Dataset Cryptocurrency")
    uploaded_file = st.file_uploader("Upload dataset dalam format CSV", type="csv")

    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file, delimiter=';', encoding='utf-8')
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            st.stop()

        # Preprocess data
        data = preprocess_data(user_data)

        if data is not None and not data.empty:
            st.success("Dataset berhasil diunggah dan diproses.")

            # Run SVR and XGBoost models
            svr_results = run_svr(data)
            xgb_results = run_xgboost(data)

            if svr_results and xgb_results:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Prediksi SVR")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(svr_results['actual'], color='blue', label='Harga Aktual')
                    ax1.plot(svr_results['predicted'], color='green', label='Prediksi SVR')
                    ax1.set_title('Harga Aktual vs Prediksi SVR')
                    ax1.set_xlabel('Hari')
                    ax1.set_ylabel('Harga Close')
                    ax1.legend()
                    st.pyplot(fig1)

                    st.write(f"**MAE:** {svr_results['mae']:.4f}")
                    st.write(f"**MSE:** {svr_results['mse']:.4f}")
                    st.write(f"**RMSE:** {svr_results['rmse']:.4f}")
                    st.write(f"**MAPE:** {svr_results['mape']:.2f}%")
                    st.write(f"**Akurasi:** {svr_results['accuracy']:.2f}%")
                
                    # Simpan grafik ke BytesIO
                    svr_image = io.BytesIO()
                    fig1.savefig(svr_image, format='png')
                    svr_image.seek(0)
                
                    # Tambahkan gambar grafik ke hasil
                    svr_results['plot_image'] = svr_image
                
                    # Tombol unduh sebagai PDF
                    svr_pdf = download_pdf(svr_results, "SVR")
                    st.download_button(
                        label="Download Laporan SVR (PDF)",
                        data=svr_pdf,
                        file_name="svr_results.pdf",
                        mime="application/pdf"
                    )


                with col2:
                    st.subheader("Prediksi XGBoost")
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(xgb_results['actual'], color='blue', label='Harga Aktual')
                    ax2.plot(xgb_results['predicted'], color='red', label='Prediksi XGBoost')
                    ax2.set_title('Harga Aktual vs Prediksi XGBoost')
                    ax2.set_xlabel('Hari')
                    ax2.set_ylabel('Harga Close')
                    ax2.legend()
                    st.pyplot(fig2)

                    st.write(f"**MAE:** {xgb_results['mae']:.4f}")
                    st.write(f"**MSE:** {xgb_results['mse']:.4f}")
                    st.write(f"**RMSE:** {xgb_results['rmse']:.4f}")
                    st.write(f"**MAPE:** {xgb_results['mape']:.2f}%")
                    st.write(f"**Akurasi:** {xgb_results['accuracy']:.2f}%")
                
                    # Simpan grafik ke BytesIO
                    xgb_image = io.BytesIO()
                    fig2.savefig(xgb_image, format='png')
                    xgb_image.seek(0)
                
                    # Tambahkan gambar grafik ke hasil
                    xgb_results['plot_image'] = xgb_image
                
                    # Tombol unduh sebagai PDF
                    xgb_pdf = download_pdf(xgb_results, "XGBoost")
                    st.download_button(
                        label="Download Laporan XGBoost (PDF)",
                        data=xgb_pdf,
                        file_name="xgboost_results.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("Gagal menjalankan model. Pastikan data Anda mencukupi dan benar.")
        else:
            st.error("Data tidak dapat diproses setelah preprocessing.")
    else:
        st.info("Silakan unggah file CSV untuk melanjutkan.")

# About Page
elif st.session_state.page == "About":
    st.title("Tentang Aplikasi")
    st.markdown("""
    <p style="font-size: 16px;">
    Aplikasi ini dibangun untuk membandingkan akurasi algoritma XGBoost dan SVR dalam memprediksi harga cryptocurrency.
    </p>
    <p>
        <a href="https://www.instagram.com/najwafadhil06_" target="_blank" style="text-decoration: none;">
            <i class="fab fa-instagram" style="font-size: 24px; color: #E1306C; margin-right: 10px;"></i> @najwafadhil06_
        </a>
    </p>
    <p>
        <a href="mailto:najwagayo@gmail.com" style="text-decoration: none;">
            <i class="fas fa-envelope" style="font-size: 24px; color: #4285F4; margin-right: 10px;"></i> najwagayo@gmail.com
        </a>
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)
