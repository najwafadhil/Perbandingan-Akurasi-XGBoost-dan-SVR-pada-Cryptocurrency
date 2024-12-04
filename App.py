import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import io
import zipfile
import warnings
from fpdf import FPDF

# Import File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Css
local_css("style.css")

# Pre-processing
def preprocess_data(data):
    data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col], errors='ignore')

    # Feature lowercase
    data.columns = map(str.lower, data.columns)

    # Date 
    data['date'] = pd.to_datetime(data['date'], dayfirst=True, errors='coerce')

    # Date Error handling
    data = data.dropna(subset=['date']).reset_index(drop=True)

    # Convert feature to numerical
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop Nan
    data = data.dropna().reset_index(drop=True)

    # Normalize 'close' 
    data['close'] = (data['close'] / 1e6) + 1e-6
    data['close'] = np.log1p(data['close'])

    # Sort by date
    data = data.sort_values(by='date').reset_index(drop=True)

    # Rekayasa fitur
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

    # Forward-fill remaining NaN values and drop rows with NaN
    data = data.ffill().dropna().reset_index(drop=True)

    if data.shape[0] < 5:
        st.warning("Data setelah preprocessing sangat terbatas. Menggunakan interpolasi untuk menambah jumlah data.")
        data = data.interpolate(method='linear').ffill().bfill().reset_index(drop=True)

    feature_columns = ['close', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rsi', 'close_lag_1', 'close_lag_2', 'close_diff']
    if data.shape[0] > 0:
        scaler = MinMaxScaler()
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
    else:
        st.error("Data tidak mencukupi untuk dilakukan scaling. Pastikan dataset memiliki setidaknya satu sampel yang valid.")
        return None

    return data

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
if "page" not in st.session_state:
    st.session_state.page = "Home"


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

    # Membaca contoh data CSV
    try:
        with open('Bitcoin.csv', 'rb') as f:
            sample_csv = f.read()

        # Tombol unduh
        st.download_button(
            label='Download Contoh Data CSV',
            data=sample_csv,
            file_name='Bitcoin.csv',
            mime='text/csv'
        )
    except FileNotFoundError:
        st.error("Contoh data CSV tidak ditemukan. Pastikan file 'contoh_data.csv' ada di direktori aplikasi.")

    # Tambahkan ruang kosong sebelum disclaimer
    st.write("\n")
    st.write("\n")
    st.write("\n")

    # Tambahkan disclaimer di bagian bawah halaman
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

            with open('preprocessed_data.pkl', 'wb') as f:
                pickle.dump(data, f)

            try:
                subprocess.run(["python", "svr.py"], check=True)
                subprocess.run(["python", "xgboost_model.py"], check=True)

            except subprocess.CalledProcessError as e:
                st.error(f"Error while running the scripts: {e}")

            if os.path.exists('svr_results.pkl') and os.path.exists('xgboost_results.pkl'):
                with open('svr_results.pkl', 'rb') as f:
                    svr_results = pickle.load(f)
                with open('xgboost_results.pkl', 'rb') as f:
                    xgb_results = pickle.load(f)

                col1, col2 = st.columns(2)

                with col1:
                        st.subheader("Prediksi SVR")
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        ax1.plot(svr_results['actual'], color='blue', label='Harga Aktual')
                        ax1.plot(svr_results['predicted'], color='green', label='Prediksi SVR')
                        ax1.set_title('Harga Aktual vs Prediksi SVR')
                        ax1.set_xlabel('Index')
                        ax1.set_ylabel('Harga Close (Skala Tereduksi)')
                        ax1.legend()
                        st.pyplot(fig1)

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
                        ax2.set_xlabel('Index')
                        ax2.set_ylabel('Harga Close (Skala Tereduksi)')
                        ax2.legend()
                        st.pyplot(fig2)

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
                if not os.path.exists('svr_results.pkl'):
                    st.error("SVR results file not found. Please check if the model ran correctly.")
                if not os.path.exists('xgboost_results.pkl'):
                    st.error("XGBoost results file not found. Please check if the model ran correctly.")

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
