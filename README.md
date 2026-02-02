# Demand Forecasting & Stock Recommendation System

This system was developed as part of an undergraduate thesis entitled:

**“Product Demand Forecasting for Paint Products Using Temporal Fusion Transformer and Light Gradient Boosting Machine at an Indonesian Paint Distribution Company”**

The system aims to support the logistics division in forecasting product demand, monitoring stock conditions, and providing order quantity recommendations based on forecasting results and inventory policies.

---

## 📌 System Description

The system performs demand forecasting for paint products at the **SKU and branch level** using **time series forecasting with external variables**.  
Three models were evaluated and compared:

- **Light Gradient Boosting Machine (LightGBM)** – *the main model implemented in the system*
- **Temporal Fusion Transformer (TFT)** – deep learning–based comparison model
- **SARIMAX** – statistical baseline model

Based on evaluation results, **LightGBM achieved the most stable and lowest forecasting error**, and was therefore selected as the operational model used in the dashboard.

In addition to forecasting results, the system also computes:
- Safety stock
- Target stock
- Order quantity recommendations
- Stock status classification (safe, risk of shortage, potential overstock)

---

## 🛠️ Technologies Used

- **Programming Language**: Python  
- **Machine Learning Models**:
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
- **Visualization**:
  - matplotlib
  - plotly

---

## 📊 Data Description

- Monthly sales data (January 2021 – May 2024)
- Branch-level and SKU-level data
- External variables:
  - Promotional events (gatherings)
  - National holidays
  - Rainfall data (for selected branches)

> The original company data is not included in this repository due to confidentiality.

---

## ⚙️ System Workflow

1. **Data Preprocessing**
   - Aggregation of transaction data into monthly sales
   - Construction of branch–SKU panel data
   - Integration of external variables

2. **Feature Engineering**
   - Seasonal features
   - Rolling statistics
   - Branch and SKU characteristics

3. **Model Training & Evaluation**
   - SARIMAX as a baseline model
   - LightGBM and TFT as comparative models
   - Evaluation using RMSE, MAE, MAPE, and MSE

4. **Forecast Generation**
   - Multi-step demand forecasting
   - Forecast results stored in the database

5. **Stock Planning & Recommendation**
   - Calculation of safety stock and maximum stock
   - Determination of target stock levels
   - Order quantity recommendations
   - Stock status classification

6. **Dashboard**
   - Visualization of actual sales vs forecast results
   - Filters by branch, SKU, and time period
   - Detailed stock and recommendation tables
   - Automated alerts for critical stock conditions

---

## 🖥️ Dashboard Features (Streamlit)

- Actual sales vs forecast comparison charts
- Multi-period demand forecasting
- Stock condition analysis by SKU and branch
- Order quantity recommendations
- Interactive filters (area, branch, SKU, period)
- Summary view of shortage and overstock risks
