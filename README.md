# ML-Based Smart Pharmacy Inventory & Medicine Shortage Prediction System

An enterprise-grade, machine-learning-powered pharmacy inventory, batch & expiry management (FEFO), automated replenishment, and shortage prediction system.

---

## 1. System Overview

This application upgrades the original prototype into a complete, database-backed pharmacy operating system. It marries real-time inventory management with validated time-series ML forecasting models:

* **Persistent SQLite Database:** 12 normalized relational tables with foreign keys and ACID transactions.
* **Batch & Expiry Management (FEFO):** Automatic First-Expiry, First-Out batch selection. Expired batches are locked from dispensing.
* **Commercial Point of Sale (POS):** Real-time checkout, stock deduction, and invoice generation (`INV-YYYYMMDD-XXXX`).
* **ML Demand Forecasting:** Rolling-validation trained models (`HistGradientBoosting`, `RandomForest`, `ExtraTrees`, and `WeeklyProfileBaseline`) predicting 7-day and 30-day demand across 8 ATC therapeutic categories.
* **Smart Shortage Alerts:** Dynamic detection of low stock, out-of-stock items, expiring batches, and ML-predicted category demand deficits.
* **Automated Reorder Engine:** Dynamic reorder point calculation incorporating supplier lead times, safety stocks, and pending purchase orders.
* **Purchase Order Lifecycle:** Multi-item purchase orders with approval workflow, partial inbound receipts, and automatic batch creation.
* **Immutable Audit History:** Every stock receipt, sale, and adjustment is permanently logged with timestamps and user accountability.
* **Comprehensive Reporting:** Exportable CSV reports for Sales, Inventory Valuation, Expiry Losses, and Forecast History.

---

## 2. Architecture & File Structure

```
medicine_shortage_prediction/
├── dashboard.py                  # Main Streamlit application with multi-section navigation
├── forecasting.py                # Time-series ML inference helper
├── train_model.py                # Rolling origin model training & evaluation pipeline
├── predict.py                    # CLI prediction tool
├── requirements.txt              # Project dependencies
├── database/
│   ├── __init__.py
│   ├── db.py                     # SQLite connection, migration, transactions & PBKDF2 hashing
│   └── schema.sql                # 12-table DDL schema with indexes
├── services/
│   ├── __init__.py
│   ├── auth_service.py           # User authentication and role-based access control
│   ├── inventory_service.py      # Medicine catalog and inventory stock management
│   ├── batch_service.py          # Batch tracking, expiry detection, and FEFO allocation
│   ├── sales_service.py          # POS dispensing, stock deduction, invoice generation
│   ├── supplier_service.py       # Supplier records and lead-time tracking
│   ├── purchase_order_service.py # Purchase order workflow and partial inbound receipts
│   ├── alert_service.py          # Real-time alert generation (stockouts, shortages, expiry)
│   ├── reorder_service.py        # Dynamic reorder points and suggested order quantities
│   ├── forecasting_service.py    # Bridge between database stock and ML model inference
│   └── reporting_service.py      # Data export service (Sales, Stock, Expiry, Forecasts)
├── tests/
│   ├── __init__.py
│   └── test_pharmacy_system.py   # Automated unit and integration test suite
├── data/
│   ├── salesdaily.csv            # 5-year daily sales panel (2014-2019)
│   └── pharmacy.db               # SQLite database
├── models/
│   └── demand_models.joblib      # Trained ML models and feature metadata
└── reports/
    └── evaluation.json           # Model evaluation metrics (MAE, WAPE %)
```

---

## 3. Database Schema

The database (`data/pharmacy.db`) contains 12 core tables:
1. `users`: Credentials with PBKDF2-HMAC-SHA256 hashing and roles (`Admin`, `Pharmacist`, `Inventory Manager`).
2. `medicines`: Medicine identity, generic name, ATC group, strength, min threshold, reorder point.
3. `medicine_batches`: Batches with manufacturing & expiry dates, cost price, MRP, and available stock.
4. `sales`: Invoices with timestamps, payment method, customer, and cashier.
5. `sale_items`: Line items mapping medicines and batches to sold quantities and prices.
6. `suppliers`: Suppliers with contact details and delivery lead times (days).
7. `purchase_orders`: Inbound procurement orders with status tracking.
8. `purchase_order_items`: Ordered vs. received quantities and negotiated prices.
9. `stock_movements`: Immutable audit trail of every stock change (`PURCHASE_RECEIPT`, `SALE`, `ADJUSTMENT`).
10. `forecast_history`: Historical snapshots of generated forecasts.
11. `alerts`: Active and resolved alerts with severity levels (`INFO`, `WARNING`, `CRITICAL`).

---

## 4. How to Run the Application

### 1. Prerequisites
Ensure Python 3.10+ (or Python 3.13) is installed.

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```powershell
streamlit run dashboard.py
```
Open **http://localhost:8501** in your browser.

### 4. Run Automated Tests
```powershell
python -m unittest tests/test_pharmacy_system.py
```

---

## 5. Walkthrough of Main Modules

### 1. 📊 Dashboard Overview
High-level KPIs, featured medicine hero banner with demand intelligence, demand vs. stock bar charts, and sales trend curves.

### 2. 📦 Inventory Management
Search, filter by ATC group, register new medicines, and monitor individual stock quantities.

### 3. 🏷️ Batches & Expiry (FEFO)
Track batches, inspect expiring medicines across configurable windows (30/60/90 days), and isolate quarantined expired stock.

### 4. 🛒 Point of Sale (POS)
Select medicine, automatically pick the earliest-expiring active batch via FEFO, validate quantity, and produce an itemized invoice.

### 5. 🔔 Smart Alerts
Real-time severity-ranked alerts for low stock, out-of-stock items, expiring batches, and ML-predicted shortages.

### 6. 📈 Demand Forecasting (ML)
7-day and 30-day forecast estimates across all 8 ATC categories with 10%-90% confidence bands and model estimator details.

### 7. 🤖 Smart Reorder Engine
Calculates reorder points using $(Daily Demand \times Lead Time) + Safety Stock$ and suggests optimal order quantities accounting for pending POs.

### 8. 🏢 Suppliers & 📋 Purchase Orders
Create draft purchase orders, approve, order, and record inbound deliveries (including partial receipts) with automatic batch creation.

### 9. 📑 Reports & Data Export
Instant CSV downloads for Sales, Inventory Valuation, Expiry Losses, and Forecast Summaries.

### 10. 🎯 Model Performance & 📜 Audit Trail
Inspect historical rolling validation scores (MAE, WAPE %) and browse the permanent stock movement log.

---

## 6. Demonstration Credentials

| Username | Password | Role | Description |
| :--- | :--- | :--- | :--- |
| `admin` | `admin123` | Admin | Full system access |
| `pharmacist` | `pharm123` | Pharmacist | Dispensing, stock viewing, alerts |
| `inventory` | `inv123` | Inventory Manager | Reorders, purchase orders, forecasts |

---

## 7. Known Limitations & Real-world Considerations
* **ATC Granularity:** The historical time-series model forecasts demand at the therapeutic category level (e.g. `N02BE`). Constituent drug demand share is calculated proportionately based on inventory balance and sales share.
* **Unrecorded Stockouts:** Sales models only observe fulfilled demand. Integrating supplier stockout records is recommended prior to clinical production deployment.
