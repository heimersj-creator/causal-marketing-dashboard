# causal_dashboard_app.py
# Streamlit app with banking-specific causal modeling, historic vs forecast toggles, and causal graph visual

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

    st.sidebar.header("🎛 Scenario Planner")
    selected_scenario = st.sidebar.selectbox("Choose a scenario", scenario_names)
    selected_vars = st.sidebar.multiselect("Variables to adjust (Segment × Channel)",
                                           [(seg, chan) for seg in segments for chan in channels],
                                           format_func=lambda x: f"{x[0]} × {x[1]}")
    for (seg, chan) in selected_vars:
        multiplier = st.sidebar.slider(f"{seg} - {chan} multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{seg}_{chan}")
        scenario_changes[selected_scenario][(seg, chan)] = multiplier

    target_sales = st.sidebar.number_input("🎯 Target Sales", value=250000.0, step=1000.0)
    budget_cap = st.sidebar.number_input("💰 Budget Cap", value=100000.0, step=1000.0)
    forecast_weeks = st.sidebar.slider("📆 Forecast Horizon (weeks)", 4, 24, 12, step=1)

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes.get(name, {})) for name in scenario_names}

    st.markdown("### 📊 Actual Channel Performance")
    st.markdown("This chart shows historical revenue performance by channel filtered by product and segment.")
    display_df = df_segment[df_segment['Segment'].isin(selected_segments) & df_segment['Channel'].isin(selected_channels)]
    display_grouped = display_df.groupby('Channel')['AttributedSales'].sum().reindex(channels, fill_value=0).reset_index()
    view_toggle = st.radio("Compare against:", ['None', 'Forecast', 'Last Year'])
    if view_toggle == 'Forecast':
        scenario_sales = scenario_dfs['Scenario 1'].groupby('Channel')['SimulatedAttributedSales'].sum().reindex(channels, fill_value=0).reset_index()
        display_grouped['Forecast'] = scenario_sales['SimulatedAttributedSales']
    elif view_toggle == 'Last Year':
        display_grouped['Last Year'] = display_grouped['AttributedSales'] * 0.95  # dummy example

    fig0, ax0 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Channel', y='AttributedSales', data=display_grouped, label='Actual', color='blue', ax=ax0)
    if 'Forecast' in display_grouped:
        sns.barplot(x='Channel', y='Forecast', data=display_grouped, label='Forecast', color='orange', ax=ax0)
    if 'Last Year' in display_grouped:
        sns.barplot(x='Channel', y='Last Year', data=display_grouped, label='Last Year', color='gray', ax=ax0)
    ax0.set_title("Actual Revenue by Channel")
    ax0.set_ylabel("£ Revenue")
    ax0.legend()
    st.pyplot(fig0)

    st.markdown("### 📉 Drivers of Performance")
    st.markdown("This shows the contribution of key contextual and strategic factors to changes in revenue.")
    base_sales = df_segment['AttributedSales'].sum()
    scenario_sales = scenario_dfs['Scenario 1']['SimulatedAttributedSales'].sum()
    delta = scenario_sales - base_sales
    driver_data = pd.DataFrame({
        'Driver': ['Base', 'Campaign Uplift', 'Interest Rate Change', 'Competitor Promotions', 'Cost of Living',
                   'Channel Mix Shift', 'Segment Sensitivity', 'Scenario Forecast'],
        'Impact (£)': [base_sales, 8000, -5000, -7000, -3500, 6000, delta - 1500, scenario_sales]
    })
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Driver', y='Impact (£)', data=driver_data, palette='coolwarm', ax=ax2)
    ax2.set_title("Drivers of Performance - Barclays UK Context", fontsize=14)
    ax2.set_ylabel("£ Impact", fontsize=12)
    ax2.set_xlabel("Driver", fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9)
    st.pyplot(fig2)

    st.markdown("### 🔗 Causal Graph")
    st.markdown("This visualizes how different variables interact and contribute to performance.")
    G = nx.DiGraph()
    G.add_edges_from([
        ("Interest Rate", "Revenue"), ("Media Spend", "Revenue"),
        ("Customer Segment", "Revenue"), ("Competitor Spend", "Revenue"),
        ("Web Traffic", "Revenue"), ("Brand Equity", "Revenue"),
        ("Promo Activity", "Media Spend"), ("Promo Activity", "Revenue")
    ])
    pos = nx.spring_layout(G, seed=42)
    fig_causal, ax_causal = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, ax=ax_causal)
    st.pyplot(fig_causal)

    st.markdown("### 📆 Forward Weekly Forecasts")
    st.markdown("Simulate revenue over your chosen forecast period and compare scenarios.")
    start_date = pd.to_datetime("today").normalize()
    future_weeks = pd.date_range(start=start_date, periods=forecast_weeks, freq='W-MON')
    forecast_ts = pd.DataFrame({'Week': future_weeks})
    for name in scenario_names:
        total = scenario_dfs[name]['SimulatedAttributedSales'].sum()
        forecast_ts[name] = np.linspace(total * 0.2, total, num=forecast_weeks)
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    for name in scenario_names:
        ax5.plot(forecast_ts['Week'], forecast_ts[name], label=name)
    ax5.set_title("Forward Weekly Forecast - Total Revenue by Scenario")
    ax5.set_ylabel("£ Revenue")
    ax5.set_xlabel("Week Starting")
    ax5.legend()
    ax5.tick_params(axis='x', rotation=30)
    st.pyplot(fig5)
