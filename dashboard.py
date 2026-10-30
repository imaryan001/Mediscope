"""
ML-Based Smart Pharmacy Inventory & Medicine Shortage Prediction System
Streamlined professional pharmacy management and demand forecasting dashboard.
"""
from pathlib import Path
import html
import json
import re
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Core Database and Services
from database.db import init_db, DB_PATH
from services.auth_service import authenticate, list_users, has_permission
from services.inventory_service import (
    list_medicines, get_medicine, add_medicine, update_medicine,
    deactivate_medicine, get_stock_history, record_stock_adjustment
)
from services.batch_service import (
    list_batches, get_fefo_batches, add_batch, update_batch,
    get_expiring_batches, get_expired_batches, calculate_expiry_loss_summary
)
from services.sales_service import (
    record_sale, list_sales, get_sale_details, get_sales_summary
)
from services.supplier_service import (
    list_suppliers, get_supplier, add_supplier, update_supplier
)
from services.purchase_order_service import (
    list_purchase_orders, get_purchase_order, create_purchase_order,
    update_po_status, receive_po_deliveries
)
from services.forecasting_service import (
    get_model_metadata, get_group_stock_from_db, generate_all_forecasts,
    get_forecast_history, get_combined_sales_series, GROUPS
)
from services.reorder_service import calculate_reorder_recommendations
from services.alert_service import (
    refresh_system_alerts, list_alerts, mark_alert_read, mark_all_alerts_read
)
from services.reporting_service import (
    generate_sales_report, generate_inventory_report, generate_expiry_report,
    generate_po_report, generate_forecast_report
)
from forecasting import DATA_PATH, MODEL_PATH, ROOT, forecast_one, load_bundle, load_history

# Initialize Database and Refresh Alerts on launch
init_db()
refresh_system_alerts()

st.set_page_config(
    page_title="MediScope | Smart Pharmacy & Demand Intelligence",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Editorial Design System CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
:root { 
    --ink: #262824; 
    --muted: #85847e; 
    --paper: #f7f5ef; 
    --line: #e8e4da; 
    --peach: #f8c89e; 
    --green: #244d42; 
    --coral: #c85a38;
}
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1400px; padding-top: 1rem; padding-bottom: 3rem; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Brand Bar */
.brandbar { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 5px 0 14px; margin-bottom: 20px; }
.brand { font-size: 1.15rem; font-weight: 700; letter-spacing: .17em; color: var(--green); display: flex; align-items: center; gap: 8px; }
.brand small { font-size: .65rem; color: #9e9b91; letter-spacing: .22em; display: block; margin-top: 2px; }
.navline { color: #77766e; font-size: .76rem; letter-spacing: .15em; text-transform: uppercase; }

/* Hero Banner */
.hero { position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; min-height: 350px; padding: 36px 6%; border-radius: 6px; background: linear-gradient(112deg, #fcfbf8 0%, #fbf7ef 49%, #f7d7ba 100%); box-shadow: 0 18px 45px rgba(72,57,35,.08); margin-bottom: 24px; border: 1px solid var(--line); }
.hero-copy { position: relative; z-index: 2; max-width: 58%; }
.eyebrow { font-size: .7rem; letter-spacing: .22em; text-transform: uppercase; color: #828176; margin-bottom: 14px; }
.hero h1 { font-family: 'Playfair Display', Georgia, serif; font-weight: 500; font-size: clamp(2.2rem, 4vw, 3.8rem); line-height: 1.05; letter-spacing: -.03em; color: #292a25; margin: 0 0 12px; }
.hero-copy p { max-width: 520px; color: #77776f; font-size: .95rem; line-height: 1.6; }
.hero-stat { margin-top: 20px; font-size: 1.7rem; font-weight: 600; color: #2a342d; }
.hero-stat span { font-size: .75rem; font-weight: 500; color: #96938a; letter-spacing: .08em; margin-left: 8px; }
.hero-art { position: absolute; right: 5%; bottom: -10px; width: 36%; height: 95%; display: flex; align-items: center; justify-content: center; }
.art-disc { position: absolute; width: 290px; height: 290px; border-radius: 50%; background: linear-gradient(145deg, #ffe8d2, #f2b982); right: 8%; top: 10%; opacity: .8; }
.pack { position: relative; z-index: 1; width: 185px; height: 245px; border-radius: 12px; background: linear-gradient(105deg, #fdfcf6 0%, #f2efe5 83%, #ddd8cb 100%); box-shadow: 16px 20px 32px rgba(83,64,38,.16); transform: rotate(-4deg); padding: 20px 16px; display: flex; flex-direction: column; align-items: flex-start; }
.pack-mark { width: 34px; height: 5px; border-radius: 8px; background: #31594d; margin-bottom: 24px; }
.pack-name { font-family: 'Playfair Display', Georgia, serif; font-size: 1.2rem; line-height: 1.1; color: #29463c; max-width: 150px; }
.pack-sub { font-size: .56rem; letter-spacing: .15em; color: #989489; text-transform: uppercase; margin-top: 8px; }
.pack-band { height: 34px; background: linear-gradient(90deg, #de9867, #f1c19a); position: absolute; left: 0; right: 0; top: 56%; opacity: .85; }
.pack-dose { font-size: .6rem; color: #78776e; position: absolute; bottom: 16px; letter-spacing: .12em; }
.capsule { position: absolute; z-index: 2; width: 66px; height: 28px; border-radius: 25px; background: linear-gradient(90deg, #fff 0 50%, #b1c5ad 50%); box-shadow: 0 8px 14px rgba(70,58,43,.15); right: 6%; bottom: 18%; transform: rotate(-32deg); }

/* Medicine Cards */
.medicine-card { border: 1px solid var(--line); background: #fffefa; padding: 16px; border-radius: 4px; min-height: 135px; transition: all .2s ease; margin-bottom: 12px; }
.medicine-card:hover { border-color: #c7b79d; box-shadow: 0 8px 20px rgba(65,56,40,.06); transform: translateY(-2px); }
.med-icon { font-size: 1.6rem; margin-bottom: 8px; }
.med-name { font-family: 'Playfair Display', Georgia, serif; font-size: 1.15rem; color: #30342f; font-weight: 500; }
.med-meta { font-size: .7rem; color: #929087; letter-spacing: .05em; margin-top: 3px; }
.pill { display: inline-block; border: 1px solid #e6dfd2; color: #77756c; padding: 3px 8px; font-size: .6rem; letter-spacing: .12em; text-transform: uppercase; border-radius: 20px; margin-top: 8px; }
.card-forecast { display: flex; justify-content: space-between; align-items: flex-end; border-top: 1px solid #eee9df; margin-top: 10px; padding-top: 8px; font-size: .65rem; letter-spacing: .06em; text-transform: uppercase; color: #858278; }
.card-forecast strong { font-family: 'DM Sans', sans-serif; font-size: .9rem; letter-spacing: 0; color: #31594d; }
.card-risk { font-size: .62rem; font-weight: 700; letter-spacing: .07em; color: #a64d36; margin-top: 4px; }
.card-safe { font-size: .62rem; font-weight: 700; letter-spacing: .07em; color: #386d55; margin-top: 4px; }

/* General Components */
.section-kicker { font-size: .68rem; letter-spacing: .2em; text-transform: uppercase; color: #96938a; }
.section-title { font-family: 'Playfair Display', Georgia, serif; font-size: 1.75rem; font-weight: 500; color: #30332e; margin: 2px 0 14px; }
.note { font-size: .78rem; color: #77756d; background: #f0ede5; padding: 12px 16px; border-left: 3px solid #d8ad84; border-radius: 2px; margin-bottom: 16px; }
div[data-testid="stMetric"] { background: #fffefa; padding: 12px 16px; border: 1px solid var(--line); border-radius: 4px; }
.stButton>button { border-radius: 3px; border: 1px solid #384f43; background: #304c41; color: white; padding: .45rem 1rem; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
.stButton>button:hover { background: #203b31; border-color: #203b31; color: #fff; }

/* Enhanced Sidebar Styling matching Dashboard Theme */
section[data-testid="stSidebar"] {
    background-color: #f2eee5 !important;
    border-right: 1px solid #dfd8cc !important;
}
section[data-testid="stSidebar"] > div {
    background-color: transparent !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #e0d9cd !important;
    margin: 1.1rem 0 !important;
}

/* Hide the extraneous 'Navigation Menu' group label */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
    display: none !important;
}

/* Strictly Equal Navigation Card Buttons */
div[data-testid="stRadio"] {
    margin-top: 4px;
    margin-bottom: 4px;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 8px;
    display: flex;
    flex-direction: column;
    width: 100%;
}
div[role="radiogroup"] label {
    background: #fffefa !important;
    border: 1px solid #e2dbce !important;
    border-radius: 8px !important;
    height: 50px !important;
    min-height: 50px !important;
    max-height: 50px !important;
    padding: 0 16px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 1px 4px rgba(45, 40, 32, 0.04) !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    width: 100% !important;
}
div[role="radiogroup"] label:hover {
    background: #fbf6ec !important;
    border-color: #c4b8a2 !important;
    transform: translateX(3px) !important;
    box-shadow: 0 3px 8px rgba(45, 40, 32, 0.08) !important;
}

/* Hide default radio circle for clean button appearance */
div[role="radiogroup"] input[type="radio"],
div[role="radiogroup"] label > div:first-child:not([data-testid="stMarkdownContainer"]) {
    display: none !important;
}

div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #262824 !important;
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    line-height: 1.2 !important;
}

/* Selected Active Navigation Item (Equal Shape with Rich Green Theme) */
div[role="radiogroup"] label:has(input:checked) {
    background: #244d42 !important;
    border-color: #1b3d34 !important;
    box-shadow: 0 4px 14px rgba(36, 77, 66, 0.26) !important;
}
div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Planning Horizon & Inputs in Sidebar */
.sidebar-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #244d42;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
div[data-testid="stSelectbox"] label p {
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: #3b3933 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #fffefa !important;
    border: 1px solid #d9d2c4 !important;
    border-radius: 7px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #244d42 !important;
    padding: 3px 6px !important;
}
.sidebar-meta {
    font-size: 0.88rem;
    color: #5c5a53;
    line-height: 1.5;
    margin-top: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "user" not in st.session_state:
    st.session_state.user = {"id": 1, "username": "admin", "role": "Admin", "full_name": "System Administrator"}
if "selected_product" not in st.session_state:
    st.session_state.selected_product = "Paracetamol 650mg"
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 30

# Load Model and History safely
try:
    history = load_history()
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load data or model: {exc}")
    st.stop()

# Build Catalog from Database
db_meds = list_medicines()
med_options = [m["name"] for m in db_meds] if db_meds else ["Paracetamol 650mg"]
if st.session_state.selected_product not in med_options and med_options:
    st.session_state.selected_product = med_options[0]

# --- STREAMLINED SIDEBAR NAVIGATION (5 ESSENTIAL SECTIONS) ---
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom:14px;border-bottom:1px solid #ded6c8;margin-bottom:16px">
      <div style="font-size:1.55rem;font-weight:800;letter-spacing:.16em;color:#244d42;line-height:1.1">MEDISCOPE</div>
      <div style="font-size:.78rem;color:#706e65;letter-spacing:.14em;text-transform:uppercase;font-weight:600;margin-top:4px">Smart Pharmacy & Forecasting</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Section Navigation (5 Core Consolidated Modules)
    menu = st.radio(
        "Navigation Menu",
        [
            "📊 Dashboard & Shortages",
            "📦 Inventory & Batches",
            "🛒 Sales & Billing (POS)",
            "🤖 Demand Forecast & Reorders",
            "📑 Reports & Audit Trail"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown('<div class="sidebar-title">⏱️ Planning Horizon</div>', unsafe_allow_html=True)
    horizon = st.selectbox(
        "Forecast Horizon",
        [7, 30],
        format_func=lambda n: f"Next {n} Days",
        key="forecast_horizon"
    )
    st.markdown(f"""
    <div class="sidebar-meta">📅 <b>History:</b> {history.index[0].date()} to {history.index[-1].date()}</div>
    """, unsafe_allow_html=True)
    trained_at = pd.Timestamp(MODEL_PATH.stat().st_mtime, unit="s")
    st.markdown(f"""
    <div class="sidebar-meta">🧠 <b>ML Model:</b> Active ({trained_at:%d %b %Y})</div>
    """, unsafe_allow_html=True)
    
    st.divider()
    # High-contrast User Profile Card
    curr_user = st.session_state.user
    role_color = "#244d42" if curr_user["role"] == "Admin" else ("#c85a38" if curr_user["role"] == "Pharmacist" else "#2a4a6b")
    st.markdown(f"""
    <div style="background:#fffefa;border:1px solid #e1d9cc;border-radius:8px;padding:12px 14px;box-shadow:0 2px 6px rgba(40,35,28,.04)">
      <div style="font-size:.74rem;color:#7a776e;text-transform:uppercase;letter-spacing:.1em;font-weight:700">Signed in as</div>
      <div style="font-size:1.05rem;font-weight:700;color:#242622;margin:2px 0 6px">{curr_user['full_name']}</div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:.82rem;color:#68665e">Role:</span>
        <span style="display:inline-block;background:{role_color};color:#fff;font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:12px;letter-spacing:.08em;text-transform:uppercase">{curr_user['role']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- TOP BRAND BAR ---
st.markdown(f"""
<div class="brandbar">
  <div class="brand">MEDISCOPE <small>PHARMACY INTELLIGENCE SYSTEM</small></div>
  <div class="navline">Section: {menu.split(' ', 1)[-1]}</div>
  <div class="navline">● &nbsp; SQLITE PERSISTENT STORE</div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 1. DASHBOARD & SHORTAGES (Overview + Critical Alerts + Charts)
# =====================================================================
if menu == "📊 Dashboard & Shortages":
    all_meds = list_medicines()
    total_active_meds = len(all_meds)
    total_avail_stock = sum(m["total_available_stock"] for m in all_meds)
    low_stock_meds = sum(1 for m in all_meds if m["total_available_stock"] <= m["min_stock_level"])
    out_of_stock_meds = sum(1 for m in all_meds if m["total_available_stock"] == 0)
    
    expiring_loss_info = calculate_expiry_loss_summary()
    expiring_count = expiring_loss_info["at_risk_units_60d"]
    
    forecasts = generate_all_forecasts(horizon=horizon)
    shortage_risks = sum(1 for fc in forecasts.values() if fc["is_shortage"])
    
    sales_sum = get_sales_summary()
    
    # KPI Metrics
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Active Medicines", f"{total_active_meds}")
    k2.metric("Available Stock", f"{total_avail_stock:,} units")
    k3.metric("Low Stock Alerts", f"{low_stock_meds}", delta=f"{out_of_stock_meds} empty", delta_color="inverse")
    k4.metric("Expiring (60d)", f"{expiring_count} units", delta=f"₹{expiring_loss_info['at_risk_loss_60d']:,.0f}", delta_color="inverse")
    k5.metric("Shortage Risks", f"{shortage_risks} groups", delta=f"{horizon}d horizon", delta_color="inverse")
    k6.metric("Today's Sales", f"₹{sales_sum['today_sales']:,.0f}", delta=f"{sales_sum['today_orders']} orders")
    k7.metric("Monthly Sales", f"₹{sales_sum['month_sales']:,.0f}", delta=f"{sales_sum['month_orders']} orders")
    
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    # Hero Card for Featured Medicine
    def set_featured(med_name):
        st.session_state.selected_product = med_name
        st.session_state.featured_med_select = med_name

    def on_med_change():
        st.session_state.selected_product = st.session_state.featured_med_select

    if "featured_med_select" not in st.session_state:
        st.session_state.featured_med_select = st.session_state.selected_product

    curr_idx = med_options.index(st.session_state.featured_med_select) if st.session_state.featured_med_select in med_options else 0
    featured_name = st.selectbox(
        "⭐ Select featured medicine to display on top banner",
        med_options,
        index=curr_idx,
        key="featured_med_select",
        on_change=on_med_change
    )
    selected_med = next((m for m in all_meds if m["name"] == featured_name), all_meds[0] if all_meds else None)
    st.session_state.selected_product = featured_name
    
    if selected_med:
        group_code = selected_med["category"]
        group_title, group_type = GROUPS.get(group_code, (selected_med["category_name"], ""))
        fc = forecasts.get(group_code, {
            "predicted_quantity": 0, "quantity": 0, "lower_bound": 0, "low": 0,
            "upper_bound": 0, "high": 0, "forecast_date": str(date.today()), "current_stock": 0
        })
        med_stock = selected_med.get("total_available_stock", 0)
        pred_qty = fc.get("predicted_quantity", fc.get("quantity", 0))
        cat_count = max(1, len([m for m in all_meds if m.get("category") == group_code]))
        risk = med_stock < (pred_qty // cat_count)
        status_label = "RESTOCK RECOMMENDED" if risk else "STOCK ADEQUATE"
        status_color = "#c85a38" if risk else "#244d42"
        
        st.markdown(f"""
        <div class="hero">
         <div class="hero-copy">
          <div class="eyebrow">MEDICINE DEMAND INTELLIGENCE &nbsp; / &nbsp; ATC GROUP: {group_code}</div>
          <h1>{html.escape(selected_med['name'])}</h1>
          <p>{html.escape(selected_med['generic_name'] or '')} · {html.escape(group_title)} ({html.escape(group_type)}). Forecast is generated at the ATC category level based on validated ML models.</p>
          <div class="hero-stat">{pred_qty:,}<span>{horizon}-DAY CATEGORY DEMAND</span></div>
          <div style="margin-top:14px;font-size:.74rem;letter-spacing:.08em;color:{status_color};font-weight:700">
            {status_label} &nbsp;·&nbsp; {med_stock:,} UNITS ON HAND ({selected_med['unit_of_measure']})
          </div>
          <div style="font-size:.72rem;color:#85847e;margin-top:8px">Forecast window starting {fc.get('forecast_date', '')} &nbsp;·&nbsp; Category stock: {fc.get('current_stock', 0):,} units &nbsp;·&nbsp; Error range: {fc.get('lower_bound', 0)}–{fc.get('upper_bound', 0)}</div>
         </div>
         <div class="hero-art">
           <div class="art-disc"></div>
           <div class="pack">
             <div class="pack-mark"></div>
             <div class="pack-name">{html.escape(selected_med['name'].split()[0])}</div>
             <div class="pack-sub">{html.escape(group_title)}</div>
             <div class="pack-band"></div>
             <div class="pack-dose">{selected_med['strength'] or 'STANDARD'}</div>
           </div>
           <div class="capsule"></div>
         </div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Charts (Plotly)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title" style="font-size:1.3rem">Demand Forecast vs Current Stock</div>', unsafe_allow_html=True)
        fc_df = pd.DataFrame(list(forecasts.values()))
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Current Stock', x=fc_df['group_name'], y=fc_df['current_stock'], marker_color='#244d42'))
        fig_bar.add_trace(go.Bar(name=f'{horizon}d Predicted Demand', x=fc_df['group_name'], y=fc_df['predicted_quantity'], marker_color='#de9867'))
        fig_bar.update_layout(barmode='group', height=330, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.markdown(f'<div class="section-title" style="font-size:1.3rem">Historical Sales Trend ({group_code})</div>', unsafe_allow_html=True)
        trend_series = history[group_code].tail(120).reset_index()
        trend_series.columns = ["Date", "Sales"]
        fig_line = px.line(trend_series, x="Date", y="Sales", color_discrete_sequence=['#244d42'])
        fig_line.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)

    # Medicine Catalog Cards
    st.markdown('<div class="section-kicker">INVENTORY CATALOG</div><div class="section-title">Active Formulations & Formats</div>', unsafe_allow_html=True)
    search_q = st.text_input("Filter catalog by drug or generic name", placeholder="Search Paracetamol, Diclofenac, Cetirizine...")
    visible_meds = [m for m in all_meds if not search_q or search_q.lower() in m["name"].lower() or search_q.lower() in (m["generic_name"] or "").lower()]
    
    for start in range(0, len(visible_meds), 4):
        cols = st.columns(4)
        for col, med in zip(cols, visible_meds[start:start+4]):
            with col:
                g_code = med["category"]
                g_fc = forecasts.get(g_code, {"predicted_quantity": 0, "quantity": 0})
                g_qty = g_fc.get("predicted_quantity", g_fc.get("quantity", 0))
                is_low = med["total_available_stock"] <= med["min_stock_level"]
                risk_cls = "card-risk" if is_low else "card-safe"
                risk_txt = "LOW STOCK ALERT" if is_low else "STOCK SAFE"
                
                is_selected = (med["name"] == st.session_state.selected_product)
                selected_border = "border: 2px solid #244d42; background: #fbf7ee;" if is_selected else ""
                
                st.markdown(f"""
                <div class="medicine-card" style="{selected_border}">
                  <div class="med-icon">💊</div>
                  <div class="med-name">{html.escape(med['name'])}</div>
                  <div class="med-meta">{html.escape(med['dosage_form'])} · {html.escape(med['strength'] or '')} · {g_code}</div>
                  <span class="pill">{html.escape(med['category_name'])}</span>
                  <div class="card-forecast">
                    <span>Category {horizon}d Demand</span>
                    <strong>{g_qty:,} units</strong>
                  </div>
                  <div class="{risk_cls}">{risk_txt} · STOCK: {med['total_available_stock']:,} {med['unit_of_measure']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                btn_label = "⭐ Currently Featured" if is_selected else "📌 Feature on Top"
                st.button(
                    btn_label,
                    key=f"feat_btn_{med['id']}",
                    use_container_width=True,
                    disabled=is_selected,
                    on_click=set_featured,
                    args=(med["name"],)
                )


# =====================================================================
# 2. INVENTORY & BATCHES (Consolidated Stock & FEFO Tracking)
# =====================================================================
elif menu == "📦 Inventory & Batches":
    st.markdown('<div class="section-kicker">STOCK CONTROL & SHELF LIFE</div><div class="section-title">Medicine Inventory & Batches</div>', unsafe_allow_html=True)
    
    itab1, itab2 = st.tabs(["📦 Medicine Stock Catalog", "🏷️ Batch & Expiry (FEFO)"])
    
    with itab1:
        with st.expander("➕ Register New Medicine", expanded=False):
            with st.form("add_med_form"):
                c1, c2, c3 = st.columns(3)
                med_code = c1.text_input("Medicine Code / SKU *", placeholder="e.g. MED-017")
                name = c2.text_input("Medicine Name *", placeholder="e.g. Pantoprazole 40mg")
                gen_name = c3.text_input("Generic Name", placeholder="e.g. Pantoprazole Sodium")
                
                c4, c5, c6 = st.columns(3)
                cat_opts = list(GROUPS.keys())
                category = c4.selectbox("ATC Group Category *", cat_opts, format_func=lambda c: f"{c} - {GROUPS[c][0]}")
                strength = c5.text_input("Strength", placeholder="e.g. 40mg")
                dosage_form = c6.selectbox("Dosage Form", ["Tablet", "Capsule", "Syrup", "Injection", "Inhaler", "Ointment", "Drops"])
                
                c7, c8 = st.columns(2)
                mfg = c7.text_input("Manufacturer", placeholder="e.g. Cipla")
                uom = c8.selectbox("Unit of Measure", ["Strips (10s)", "Strips (15s)", "Bottle", "Vial", "Canister", "Units"])
                
                c9, c10 = st.columns(2)
                min_lvl = c9.number_input("Minimum Stock Alert Level", min_value=5, value=50, step=5)
                reorder_lvl = c10.number_input("Reorder Trigger Level", min_value=10, value=100, step=10)
                
                submitted = st.form_submit_button("Save Medicine in Database")
                if submitted:
                    if not med_code or not name:
                        st.error("Medicine code and name are mandatory.")
                    else:
                        try:
                            new_id = add_medicine({
                                "medicine_code": med_code, "name": name, "generic_name": gen_name,
                                "category": category, "category_name": GROUPS[category][0],
                                "strength": strength, "dosage_form": dosage_form, "manufacturer": mfg,
                                "unit_of_measure": uom, "min_stock_level": min_lvl, "reorder_level": reorder_lvl
                            })
                            st.success(f"Medicine '{name}' registered successfully with ID #{new_id}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        f1, f2 = st.columns([3, 1])
        search_term = f1.text_input("Search Medicines by code, name, or generic", placeholder="Search...")
        cat_filter = f2.selectbox("Filter by ATC Category", ["All"] + list(GROUPS.keys()))
        
        meds = list_medicines(category=cat_filter, search=search_term)
        if meds:
            st.dataframe(pd.DataFrame(meds)[[
                "medicine_code", "name", "generic_name", "category", "category_name",
                "dosage_form", "strength", "total_available_stock", "expired_stock",
                "min_stock_level", "reorder_level", "supplier_name"
            ]].rename(columns={
                "medicine_code": "SKU", "name": "Medicine", "generic_name": "Generic",
                "category": "ATC", "category_name": "Category", "dosage_form": "Form",
                "strength": "Strength", "total_available_stock": "Available Stock",
                "expired_stock": "Expired Units", "min_stock_level": "Min Alert",
                "reorder_level": "Reorder Level", "supplier_name": "Supplier"
            }), use_container_width=True, hide_index=True)

    with itab2:
        loss_summary = calculate_expiry_loss_summary()
        e1, e2, e3 = st.columns(3)
        e1.metric("Expired Quarantined Stock", f"{loss_summary['expired_units']:,} units", delta=f"₹{loss_summary['expired_loss']:,.2f} Loss", delta_color="inverse")
        e2.metric("At Risk Within 60 Days", f"{loss_summary['at_risk_units_60d']:,} units", delta=f"₹{loss_summary['at_risk_loss_60d']:,.2f} At Risk", delta_color="inverse")
        e3.info("💡 **FEFO Principle:** Dispensing automatically uses the earliest-expiring batch. Expired stock is locked.")
        
        with st.expander("➕ Add New Batch", expanded=False):
            with st.form("add_batch_form"):
                all_m = list_medicines()
                m_map = {m["id"]: f"{m['name']} ({m['medicine_code']})" for m in all_m}
                b_med_id = st.selectbox("Select Medicine *", list(m_map.keys()), format_func=lambda x: m_map[x])
                
                b1, b2, b3 = st.columns(3)
                b_num = b1.text_input("Batch Number *", placeholder="e.g. BATCH-2026-X1")
                p_date = b2.date_input("Purchase Date *", value=date.today())
                exp_date = b3.date_input("Expiry Date *", value=date.today() + timedelta(days=365))
                
                b4, b5, b6 = st.columns(3)
                qty_rec = b4.number_input("Quantity Received *", min_value=1, value=100, step=10)
                cost_price = b5.number_input("Purchase Price (₹)", min_value=0.0, value=15.0, step=1.0)
                sell_price = b6.number_input("Selling Price / MRP (₹)", min_value=0.0, value=25.0, step=1.0)
                
                if st.form_submit_button("Save Batch Record"):
                    try:
                        add_batch({
                            "medicine_id": b_med_id, "batch_number": b_num,
                            "purchase_date": str(p_date), "expiry_date": str(exp_date),
                            "quantity_received": qty_rec, "purchase_price": cost_price,
                            "selling_price": sell_price
                        }, user_id=st.session_state.user["id"])
                        st.success(f"Batch {b_num} created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Expiring and Expired tables
        sub_tab1, sub_tab2 = st.tabs(["⚠️ Expiring Within 90 Days", "🚫 Quarantined Expired Stock"])
        with sub_tab1:
            expiring = get_expiring_batches(days_threshold=90)
            if expiring:
                st.dataframe(pd.DataFrame(expiring)[[
                    "batch_number", "medicine_name", "expiry_date", "days_left",
                    "available_quantity", "purchase_price", "potential_loss"
                ]].rename(columns={
                    "batch_number": "Batch #", "medicine_name": "Medicine", "expiry_date": "Expiry Date",
                    "days_left": "Days Left", "available_quantity": "Stock Qty",
                    "purchase_price": "Unit Cost (₹)", "potential_loss": "Potential Loss (₹)"
                }), use_container_width=True, hide_index=True)
            else:
                st.success("No batches expiring within 90 days.")
                
        with sub_tab2:
            expired = get_expired_batches()
            if expired:
                st.dataframe(pd.DataFrame(expired)[[
                    "batch_number", "medicine_name", "expiry_date", "days_expired_ago",
                    "available_quantity", "loss_amount"
                ]].rename(columns={
                    "batch_number": "Batch #", "medicine_name": "Medicine", "expiry_date": "Expired On",
                    "days_expired_ago": "Days Expired", "available_quantity": "Units Quarantined",
                    "loss_amount": "Total Loss (₹)"
                }), use_container_width=True, hide_index=True)
            else:
                st.success("Zero expired stock.")


# =====================================================================
# 3. SALES & BILLING (POS)
# =====================================================================
elif menu == "🛒 Sales & Billing (POS)":
    st.markdown('<div class="section-kicker">DISPENSING & BILLING</div><div class="section-title">Point of Sale & Invoicing</div>', unsafe_allow_html=True)
    
    stab1, stab2 = st.tabs(["💳 New Sale (POS)", "📜 Past Invoices & Receipts"])
    
    with stab1:
        all_m = list_medicines()
        med_dict = {m["id"]: m for m in all_m}
        
        with st.form("pos_sale_form"):
            p1, p2 = st.columns([2, 1])
            chosen_med_id = p1.selectbox("Select Medicine to Dispense *", list(med_dict.keys()), format_func=lambda x: f"{med_dict[x]['name']} (Avail: {med_dict[x]['total_available_stock']} units)")
            cust_name = p2.text_input("Customer Name", value="Walk-in Customer")
            
            med_selected = med_dict[chosen_med_id]
            fefo_b = get_fefo_batches(chosen_med_id)
            
            p3, p4, p5 = st.columns(3)
            if fefo_b:
                next_batch = fefo_b[0]
                unit_price = float(next_batch["selling_price"] or 30.0)
                p3.info(f"**FEFO Batch:** `{next_batch['batch_number']}` (Expires: {next_batch['expiry_date']})")
            else:
                unit_price = 0.0
                p3.error("NO NON-EXPIRED STOCK AVAILABLE!")
                
            qty = p4.number_input("Quantity *", min_value=1, max_value=max(1, med_selected["total_available_stock"]), value=1, step=1)
            pmt = p5.selectbox("Payment Method", ["Cash", "UPI / QR", "Credit Card", "Debit Card"])
            
            total_bill = qty * unit_price
            st.markdown(f"### Total Billable Amount: **₹{total_bill:,.2f}**")
            
            submit_sale = st.form_submit_button("Confirm Sale & Generate Invoice", use_container_width=True)
            if submit_sale:
                if med_selected["total_available_stock"] < qty:
                    st.error(f"Insufficient stock. Available: {med_selected['total_available_stock']}.")
                elif not fefo_b:
                    st.error("No valid non-expired batch available.")
                else:
                    try:
                        res = record_sale(
                            items=[{"medicine_id": chosen_med_id, "quantity": qty, "unit_price": unit_price}],
                            customer_name=cust_name, payment_method=pmt, user_id=st.session_state.user["id"]
                        )
                        st.success(f"Sale completed successfully! Invoice: **{res['invoice_number']}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transaction failed: {e}")

    with stab2:
        sales_records = list_sales(limit=50)
        if sales_records:
            st.dataframe(pd.DataFrame(sales_records)[[
                "invoice_number", "sale_date", "customer_name", "payment_method",
                "total_items", "total_units_sold", "total_amount", "username"
            ]].rename(columns={
                "invoice_number": "Invoice #", "sale_date": "Timestamp", "customer_name": "Customer",
                "payment_method": "Payment", "total_items": "Items", "total_units_sold": "Units Sold",
                "total_amount": "Total (₹)", "username": "Cashier"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("No recorded sales found.")


# =====================================================================
# 4. DEMAND FORECAST & REORDERS (ML Demand + Reorders + Purchase Orders)
# =====================================================================
elif menu == "🤖 Demand Forecast & Reorders":
    st.markdown('<div class="section-kicker">PREDICTIVE REPLENISHMENT</div><div class="section-title">Demand Forecasting & Reorder Engine</div>', unsafe_allow_html=True)
    
    ftab1, ftab2, ftab3 = st.tabs(["📈 ML Category Forecasts", "🤖 Smart Reorder Recommendations", "📋 Purchase Orders"])
    
    with ftab1:
        st.markdown('<div class="note"><b>ATC Group Forecasting:</b> Evaluates tree ensembles and seasonal baselines on historical sales. Confidence intervals show 10%–90% error bands.</div>', unsafe_allow_html=True)
        fc_results = generate_all_forecasts(horizon=horizon, record_history=True)
        fc_table = pd.DataFrame(list(fc_results.values()))
        
        st.dataframe(fc_table[[
            "group_code", "group_name", "predicted_quantity", "daily_average",
            "lower_bound", "upper_bound", "current_stock", "shortage_gap", "model_name", "risk_level"
        ]].rename(columns={
            "group_code": "ATC Code", "group_name": "Category Name", "predicted_quantity": f"{horizon}d Demand",
            "daily_average": "Daily Avg", "lower_bound": "Lower (10%)", "upper_bound": "Upper (90%)",
            "current_stock": "Current Stock", "shortage_gap": "Deficit Gap", "model_name": "Selected Estimator", "risk_level": "Risk Level"
        }), use_container_width=True, hide_index=True)

    with ftab2:
        st.markdown("""
        <div class="note">
          <b>Formula:</b> <code>Reorder Point = (Daily Demand × Lead Time) + Safety Stock</code>.
          Incorporates lead times and pending supplier deliveries to prevent over-ordering.
        </div>
        """, unsafe_allow_html=True)
        recs = calculate_reorder_recommendations(horizon=horizon)
        st.dataframe(pd.DataFrame(recs)[[
            "medicine_name", "category", "current_stock", "pending_po_qty", "effective_stock",
            "lead_time_days", "reorder_point", "suggested_order_qty", "estimated_order_cost",
            "supplier_name", "risk_level", "reason"
        ]].rename(columns={
            "medicine_name": "Medicine", "category": "Category", "current_stock": "Stock",
            "pending_po_qty": "Pending PO", "effective_stock": "Effective Stock",
            "lead_time_days": "Lead Time (d)", "reorder_point": "Reorder Point",
            "suggested_order_qty": "Suggested Order", "estimated_order_cost": "Est. Cost (₹)",
            "supplier_name": "Supplier", "risk_level": "Priority Status", "reason": "Justification"
        }), use_container_width=True, hide_index=True)

    with ftab3:
        st.markdown("### Active Purchase Orders & Goods Receiving")
        orders = list_purchase_orders()
        if orders:
            st.dataframe(pd.DataFrame(orders)[[
                "po_number", "supplier_name", "order_date", "expected_date",
                "total_ordered_qty", "total_received_qty", "total_amount", "status"
            ]].rename(columns={
                "po_number": "PO #", "supplier_name": "Supplier", "order_date": "Order Date",
                "expected_date": "Expected Date", "total_ordered_qty": "Ordered Qty",
                "total_received_qty": "Received Qty", "total_amount": "Total (₹)", "status": "Status"
            }), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            po_id_sel = st.selectbox("Select PO to Receive Delivery", [o["id"] for o in orders], format_func=lambda x: next(o["po_number"] for o in orders if o["id"] == x))
            sel_po = get_purchase_order(po_id_sel)
            if sel_po and sel_po["header"]["status"] in ("Ordered", "Approved", "Partially Received"):
                with st.form("quick_receive_form"):
                    receipt_inputs = []
                    for it in sel_po["items"]:
                        rem = it["quantity_ordered"] - it["quantity_received"]
                        st.write(f"**{it['medicine_name']}** (Ordered: {it['quantity_ordered']}, Pending: {rem})")
                        rc1, rc2, rc3 = st.columns(3)
                        q_now = rc1.number_input(f"Qty Received Now", min_value=0, max_value=rem, value=rem, key=f"q_{it['id']}")
                        b_num = rc2.text_input(f"Batch Number", value=f"BATCH-PO-{it['id']}", key=f"b_{it['id']}")
                        exp_d = rc3.date_input(f"Expiry Date", value=date.today() + timedelta(days=730), key=f"e_{it['id']}")
                        receipt_inputs.append({"item_id": it["id"], "quantity_received_now": q_now, "batch_number": b_num, "expiry_date": str(exp_d)})
                    if st.form_submit_button("Confirm Inbound Receipt"):
                        try:
                            receive_po_deliveries(po_id_sel, receipt_inputs, user_id=st.session_state.user["id"])
                            st.success("Delivery received and stock updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No purchase orders found.")


# =====================================================================
# 5. REPORTS & AUDIT TRAIL (CSV Reports + Model Evaluation + Audit Log)
# =====================================================================
elif menu == "📑 Reports & Audit Trail":
    st.markdown('<div class="section-kicker">COMPLIANCE & REPORTING</div><div class="section-title">Reports & Audit History</div>', unsafe_allow_html=True)
    
    rtab1, rtab2, rtab3, rtab4 = st.tabs(["📥 Download CSV Reports", "🔔 System Alerts", "🎯 Model Performance Scores", "📜 Immutable Stock Movement Log"])
    
    with rtab1:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("#### Sales Report")
            s_df = generate_sales_report()
            st.download_button("Download Sales CSV", data=s_df.to_csv(index=False), file_name=f"sales_{date.today()}.csv", mime="text/csv", use_container_width=True)
            
            st.markdown("#### Inventory Valuation Report")
            i_df = generate_inventory_report()
            st.download_button("Download Inventory CSV", data=i_df.to_csv(index=False), file_name=f"inventory_{date.today()}.csv", mime="text/csv", use_container_width=True)
            
        with rc2:
            st.markdown("#### Expiry Loss Report")
            e_df = generate_expiry_report(days_threshold=90)
            st.download_button("Download Expiry CSV", data=e_df.to_csv(index=False), file_name=f"expiry_{date.today()}.csv", mime="text/csv", use_container_width=True)
            
            st.markdown("#### Demand Forecasts Report")
            f_df = generate_forecast_report(horizon=horizon)
            st.download_button("Download Forecast CSV", data=f_df.to_csv(index=False), file_name=f"forecasts_{date.today()}.csv", mime="text/csv", use_container_width=True)

    with rtab2:
        st.markdown("### Active & Logged System Alerts")
        alerts_list = list_alerts(limit=50)
        if alerts_list:
            for al in alerts_list:
                is_crit = al["severity"] == "CRITICAL"
                sev_badge_bg = "#fbeae5" if is_crit else "#fef3e7"
                sev_badge_fg = "#c85a38" if is_crit else "#b56928"
                sev_icon = "🚨" if is_crit else "⚠️"
                border_c = "#e8c9c0" if is_crit else "#eeddc9"
                
                c_card, c_btn = st.columns([6, 1])
                with c_card:
                    st.markdown(f"""
                    <div style="background:#fffefa;border:1px solid {border_c};border-left:4px solid {sev_badge_fg};padding:10px 14px;border-radius:4px;display:flex;align-items:center;gap:12px;margin-bottom:8px">
                      <span style="background:{sev_badge_bg};color:{sev_badge_fg};font-weight:700;font-size:.68rem;padding:3px 8px;border-radius:12px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap">
                        {sev_icon} {al['severity']}
                      </span>
                      <span style="font-size:.86rem;color:#262824;line-height:1.4">
                        {html.escape(al['message'])}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_btn:
                    if al["is_read"] == 0:
                        if st.button("Acknowledge", key=f"rep_ack_{al['id']}", use_container_width=True):
                            mark_alert_read(al["id"])
                            st.rerun()
                    else:
                        st.caption("✓ Read")
        else:
            st.success("No active alerts.")

    with rtab3:
        meta = get_model_metadata()
        m1, m2, m3 = st.columns(3)
        m1.metric("Historical Panel Data", f"{meta['dataset_rows']:,} days")
        m2.metric("Date Span", f"{meta['date_start']} to {meta['date_end']}")
        m3.metric("Evaluated Categories", f"{len(meta['groups'])} ATC Groups")
        
        eval_rows = []
        for grp, h_data in meta.get("evaluation", {}).items():
            for horiz, details in h_data.items():
                eval_rows.append({
                    "ATC Category": grp, "Category Name": GROUPS.get(grp, (grp, ""))[0],
                    "Horizon": f"{horiz} Days", "Winner Estimator": details.get("selected_model"),
                    "WAPE Error %": details.get("rolling_validation", {}).get("wape_percent"),
                    "MAE (Units)": details.get("rolling_validation", {}).get("mae"),
                    "Weekly Baseline WAPE %": details.get("weekly_profile_baseline", {}).get("wape_percent")
                })
        if eval_rows:
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

    with rtab3:
        history_logs = get_stock_history(limit=100)
        if history_logs:
            st.dataframe(pd.DataFrame(history_logs)[[
                "created_at", "movement_type", "medicine_name", "medicine_code",
                "batch_number", "quantity_change", "previous_quantity", "new_quantity",
                "reference_id", "reason", "username"
            ]].rename(columns={
                "created_at": "Timestamp", "movement_type": "Movement",
                "medicine_name": "Medicine", "medicine_code": "SKU", "batch_number": "Batch #",
                "quantity_change": "Change", "previous_quantity": "Prev Qty",
                "new_quantity": "New Qty", "reference_id": "Ref #",
                "reason": "Note", "username": "User"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("No stock movements recorded.")

# Footer
st.markdown("""
<div style="margin-top:40px;border-top:1px solid #e8e4da;padding-top:16px;color:#8b887f;font-size:.72rem;letter-spacing:.05em;display:flex;justify-content:space-between">
  <div>MEDISCOPE SMART PHARMACY INVENTORY &amp; SHORTAGE PREDICTION SYSTEM</div>
  <div>SQLITE PERSISTENT STORE &nbsp;·&nbsp; TIME-SERIES ML BACKEND</div>
</div>
""", unsafe_allow_html=True)
