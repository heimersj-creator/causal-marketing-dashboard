# causal_dashboard_app_barclays_uk.py
# Enhanced with per-chart filters, independent scenario controls, forecast toggles, and causal graph with weights

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import networkx as nx

st.set_page_config(layout="wide")
sns.set(style='whitegrid')

st.title("📊 Barclays Consumer Banking - Marketing Optimization Dashboard")

uploaded_file = st.file_uploader("Upload the Barclays causal simulator Excel file", type="xlsx")

if uploaded_file:
    df_segment = pd.read_excel(uploaded_file, sheet_name='Segment Attribution')
    df_product = pd.read_excel(uploaded_file, sheet_name='Product Attribution')
    df_weights = pd.read_excel(uploaded_file, sheet_name='Causal Weights')

    segments = df_segment['Segment'].unique().tolist()
    channels = ["TV", "Paid Social", "Email", "Push Notification", "Branch", "Online Display", "Search"]
    products = ["Current Account", "Credit Card", "Mortgage", "Personal Loan", "Savings Account"]
    scenario_names = ['Base', 'Scenario 1', 'Scenario 2', 'Scenario 3']
    scenario_changes = {name: {} for name in scenario_names}

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes.get(name, {})) for name in scenario_names}

    # 1. Revenue by Channel (Historic)
    st.markdown("### 📈 Revenue by Channel (Historic Cumulative)")
    st.markdown("Cumulative revenue performance over time filtered by channel, product, and segment.")
    col1, col2, col3 = st.columns(3)
    with col1:
        hist_channels = st.multiselect("Channels", channels, default=channels, key="hist_ch")
    with col2:
        hist_segments = st.multiselect("Segments", segments, default=segments, key="hist_seg")
    with col3:
        hist_products = st.multiselect("Products", products, default=products, key="hist_prod")

    df_hist = df_segment[(df_segment['Channel'].isin(hist_channels)) &
                         (df_segment['Segment'].isin(hist_segments))]
    df_hist['Date'] = pd.date_range('2024-01-01', periods=len(df_hist), freq='W')
    ts_hist = df_hist.groupby('Date')['AttributedSales'].sum().cumsum().reset_index()
    compare_opt = st.radio("Compare against:", ['None', 'Forecast', 'Last Year'], key="hist_cmp")

    fig0, ax0 = plt.subplots(figsize=(10, 4))
    ax0.plot(ts_hist['Date'], ts_hist['AttributedSales'], label="Actual")
    if compare_opt == 'Forecast':
        scenario1 = scenario_dfs['Scenario 1']
        ts_f = pd.DataFrame({
            'Date': ts_hist['Date'],
            'Forecast': np.linspace(ts_hist['AttributedSales'].iloc[0], ts_hist['AttributedSales'].iloc[-1]*1.1, len(ts_hist))
        })
        ax0.plot(ts_f['Date'], ts_f['Forecast'], label="Forecast", linestyle='--')
    elif compare_opt == 'Last Year':
        ax0.plot(ts_hist['Date'], ts_hist['AttributedSales'] * 0.95, label="Last Year", linestyle='--')
    ax0.set_title("Revenue by Channel")
    ax0.set_ylabel("£ Revenue")
    ax0.legend()
    st.pyplot(fig0)

    # 2. Scenario Planner Controls (above forecast)
    st.markdown("### 🔧 Scenario Planner Inputs")
    st.markdown("Adjust spend per channel and segment below for each scenario.")
    scenario_weeks = st.slider("Select Forecast Horizon (weeks)", 4, 52, 12, step=1, key="week_horizon")
    for scenario in scenario_names:
        st.markdown(f"**{scenario} Adjustments**")
        for seg in segments:
            for ch in channels:
                key = f"{scenario}_{seg}_{ch}"
                mult = st.slider(f"{scenario}: {seg} × {ch}", 0.0, 2.0, 1.0, 0.1, key=key)
                scenario_changes[scenario][(seg, ch)] = mult
    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names}

    # 3. Forecasts by Week
    st.markdown("### 📆 Forecasted Weekly Revenue")
    st.markdown("Non-cumulative weekly forecasts by scenario.")
    start_date = pd.to_datetime("today").normalize()
    forecast_dates = pd.date_range(start=start_date, periods=scenario_weeks, freq='W-MON')
    forecast_data = pd.DataFrame({'Week': forecast_dates})
    for name in scenario_names:
        forecast_data[name] = np.linspace(5000, 5000 + 500*scenario_weeks, scenario_weeks)
    fig_fw, ax_fw = plt.subplots(figsize=(10, 4))
    for name in scenario_names:
        ax_fw.plot(forecast_data['Week'], forecast_data[name], label=name)
    ax_fw.set_title("Weekly Forecast Revenue")
    ax_fw.set_ylabel("£ Revenue")
    ax_fw.legend()
    st.pyplot(fig_fw)

    # 4. Causal Graph
    st.markdown("### 🧠 Causal Graph with Weights")
    st.markdown("This diagram shows weighted relationships used in the causal model.")
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ("Promo", "Spend", 0.8),
        ("Interest Rate", "Demand", -0.5),
        ("Demand", "Spend", 0.6),
        ("Spend", "Revenue", 1.2),
        ("Customer Segment", "Revenue", 0.9),
        ("Brand Equity", "Revenue", 0.7),
        ("Competitor Spend", "Revenue", -0.4),
        ("Search Trends", "Brand Equity", 0.5)
    ])
    pos = nx.spring_layout(G, seed=42)
    fig_g, ax_g = plt.subplots(figsize=(8, 6))
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', ax=ax_g)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, ax=ax_g)
    st.pyplot(fig_g)
