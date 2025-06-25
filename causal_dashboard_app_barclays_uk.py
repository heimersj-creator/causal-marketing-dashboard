# causal_dashboard_app_barclays_uk.py

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

    st.sidebar.header("📂 Filter Dimensions")
    selected_channels = st.sidebar.multiselect("Channels", channels, default=channels)
    selected_products = st.sidebar.multiselect("Products", products, default=products)
    selected_segments = st.sidebar.multiselect("Customer Segments", segments, default=segments)
    date_range = st.sidebar.slider("Select Date Range", value=(20240101, 20241231))

    st.sidebar.header("📈 Scenario Planner Controls")
    target_sales = st.sidebar.number_input("🎯 Target Sales", value=250000.0, step=1000.0)
    budget_cap = st.sidebar.number_input("💰 Budget Cap", value=100000.0, step=1000.0)
    forecast_weeks = st.sidebar.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1)

    st.sidebar.markdown("### ✏️ Adjust Each Scenario")
    for scenario in scenario_names:
        st.sidebar.markdown(f"**{scenario}**")
        for segment in segments:
            for channel in channels:
                key = f"{scenario}_{segment}_{channel}"
                mult = st.sidebar.slider(key, 0.0, 2.0, 1.0, 0.1)
                scenario_changes.setdefault(scenario, {})[(segment, channel)] = mult

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names}

    # ---- 1. Historic Performance View ----
    st.markdown("### 📊 Historic Channel Performance")
    st.markdown("View past revenue performance by channel with optional comparison to forecast or prior year.")
    actuals_df = df_segment[df_segment['Segment'].isin(selected_segments) & df_segment['Channel'].isin(selected_channels)]
    view_toggle = st.radio("Compare against:", ['None', 'Forecast', 'Last Year'])
    grouped_actuals = actuals_df.groupby('Channel')['AttributedSales'].sum().reindex(channels, fill_value=0).reset_index()
    if view_toggle == 'Forecast':
        grouped_actuals['Forecast'] = scenario_dfs['Scenario 1'].groupby('Channel')['SimulatedAttributedSales'].sum().reindex(channels, fill_value=0).values
    elif view_toggle == 'Last Year':
        grouped_actuals['Last Year'] = grouped_actuals['AttributedSales'] * 0.95  # dummy data

    fig_historic, axh = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Channel', y='AttributedSales', data=grouped_actuals, label='Actual', ax=axh)
    if 'Forecast' in grouped_actuals:
        sns.barplot(x='Channel', y='Forecast', data=grouped_actuals, label='Forecast', ax=axh, color='orange')
    if 'Last Year' in grouped_actuals:
        sns.barplot(x='Channel', y='Last Year', data=grouped_actuals, label='Last Year', ax=axh, color='gray')
    axh.set_title("Historic Revenue by Channel")
    axh.set_ylabel("£ Revenue")
    axh.legend()
    st.pyplot(fig_historic)

    # ---- 2. Historic Weekly Trend ----
    st.markdown("### 📈 Historic Weekly Performance")
    st.markdown("See how revenue evolved weekly over the past year, by channel.")
    weekly_ts = pd.DataFrame({
        'Week': pd.date_range('2024-01-01', periods=52, freq='W-MON')
    })
    for chan in channels:
        weekly_ts[chan] = np.random.randint(1000, 5000, size=52)  # dummy data
    fig_hist_ts, ax_ts = plt.subplots(figsize=(12, 4))
    for chan in selected_channels:
        ax_ts.plot(weekly_ts['Week'], weekly_ts[chan], label=chan)
    ax_ts.set_title("Historic Weekly Revenue by Channel")
    ax_ts.set_ylabel("£ Revenue")
    ax_ts.legend()
    st.pyplot(fig_hist_ts)

    # ---- 3. Forecast View - Weekly (Non-cumulative) ----
    st.markdown("### 📆 Forward Weekly Forecasts")
    st.markdown("Simulated forecast by scenario, broken down weekly (not cumulative).")
    start_date = pd.to_datetime("today").normalize()
    future_weeks = pd.date_range(start=start_date, periods=forecast_weeks, freq='W-MON')
    forecast_ts = pd.DataFrame({'Week': future_weeks})
    for name in scenario_names:
        weekly = np.random.randint(4000, 10000, size=forecast_weeks)  # dummy dynamic sim
        forecast_ts[name] = weekly
    fig_fc, ax_fc = plt.subplots(figsize=(12, 4))
    for name in scenario_names:
        ax_fc.plot(forecast_ts['Week'], forecast_ts[name], label=name)
    ax_fc.set_title("Weekly Forecast Revenue (Non-Cumulative)")
    ax_fc.set_ylabel("£ Revenue")
    ax_fc.set_xlabel("Week Starting")
    ax_fc.legend()
    st.pyplot(fig_fc)

    # ---- 4. Causal Graph ----
    st.markdown("### 🔗 Causal Graph of Marketing Drivers")
    st.markdown("This diagram shows how different inputs contribute to marketing and revenue performance.")
    G = nx.DiGraph()
    G.add_edges_from([
        ("Interest Rate", "Spend"), ("Media Spend", "Revenue"),
        ("Customer Segment", "Revenue"), ("Competitor Activity", "Revenue"),
        ("Search Trends", "Brand Equity"), ("Brand Equity", "Revenue"),
        ("Promo Activity", "Media Spend"), ("Web Traffic", "Revenue")
    ])
    pos = nx.spring_layout(G, seed=42)
    fig_causal, ax_causal = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, arrows=True, ax=ax_causal)
    st.pyplot(fig_causal)
