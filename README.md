# Demand Forecasting & Stock Recommendation System

Sistem ini dikembangkan sebagai bagian dari skripsi berjudul:

**“Prediksi Permintaan Produk Cat Menggunakan Temporal Fusion Transformer dan Light Gradient Boosting Machine pada PT Tirtakencana Tatawarna”**

Sistem bertujuan untuk membantu divisi logistik dalam memprediksi permintaan produk, memantau kondisi stok, serta memberikan rekomendasi jumlah pemesanan berdasarkan hasil peramalan dan kebijakan stok perusahaan.

---

## 📌 Deskripsi Sistem

Sistem melakukan peramalan permintaan produk cat pada level **SKU dan cabang** menggunakan pendekatan **time series forecasting dengan variabel eksternal**.  
Tiga model diuji dan dibandingkan:

- **Light Gradient Boosting Machine (LightGBM)** – *model utama yang diimplementasikan ke sistem*
- **Temporal Fusion Transformer (TFT)** – model pembanding berbasis deep learning
- **SARIMAX** – model baseline statistik

Berdasarkan hasil evaluasi, **LightGBM menghasilkan error paling stabil dan terendah**, sehingga dipilih sebagai model operasional pada dashboard.

Selain menampilkan hasil forecast, sistem juga menghitung:
- Safety Stock
- Target Stok
- Rekomendasi jumlah pemesanan
- Status stok (aman, risiko kekurangan, potensi kelebihan)

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python  
- **Machine Learning**:
  - LightGBM
  - Temporal Fusion Transformer (TFT)
  - SARIMAX
- **Web Framework**: Streamlit
- **Database**: MySQL
- **Data Processing & Analysis**:
  - pandas
  - numpy
  - scikit-learn
- **Hyperparameter Tuning**:
  - Optuna (LightGBM)
  - Random Search & Bayesian Optimization (TFT)
- **Visualisasi**:
  - matplotlib
  - plotly

---

## 📊 Data yang Digunakan

- Data penjualan bulanan (Januari 2021 – Mei 2024)
- Data per cabang dan SKU
- Variabel eksternal:
  - Event promosi (gathering)
  - Hari libur nasional
  - Curah hujan (cabang tertentu)

> Data asli perusahaan tidak disertakan dalam repository ini.

---

## ⚙️ Alur Sistem

1. **Preprocessing Data**
   - Agregasi data transaksi menjadi bulanan
   - Pembentukan panel data cabang–SKU
   - Penggabungan variabel eksternal

2. **Feature Engineering**
   - Fitur musiman
   - Rolling statistics
   - Informasi cabang dan SKU

3. **Model Training & Evaluation**
   - SARIMAX sebagai baseline
   - LightGBM dan TFT sebagai model pembanding
   - Evaluasi menggunakan RMSE, MAE, MAPE, dan MSE

4. **Forecast Generation**
   - Prediksi permintaan beberapa bulan ke depan
   - Disimpan ke database

5. **Stock Planning & Recommendation**
   - Perhitungan MAX dan Safety Stock
   - Penentuan target stok
   - Rekomendasi jumlah pemesanan
   - Klasifikasi status stok

6. **Dashboard**
   - Visualisasi penjualan aktual vs forecast
   - Filter cabang, SKU, dan periode
   - Tabel detail stok dan rekomendasi
   - Peringatan otomatis kondisi stok

---

## 🖥️ Fitur Dashboard (Streamlit)

- Grafik perbandingan penjualan aktual dan hasil forecast
- Forecast hingga beberapa bulan ke depan
- Analisis kondisi stok per SKU dan cabang
- Rekomendasi jumlah pemesanan
- Filter interaktif (area, cabang, SKU, periode)
- Tampilan ringkasan risiko kekurangan dan kelebihan stok

---

## 📂 Struktur Repository (Ringkas)

