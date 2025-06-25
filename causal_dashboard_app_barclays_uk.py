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
    forecast_weeks = st.sidebar.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1)

    st.sidebar.markdown("### ✏️ Scenario Builder")
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

    # 1. Cumulative Historic Performance View
    st.markdown("### 📈 Historic Performance Trend (Cumulative)")
    st.markdown("Shows total historic revenue over time. Use filters above to narrow by product, channel, or segment.")
    df_filtered = df_segment[df_segment['Channel'].isin(selected_channels) &
                             df_segment['Segment'].isin(selected_segments)]
    df_filtered['Date'] = pd.date_range('2024-01-01', periods=len(df_filtered), freq='W')  # dummy date
    historic_ts = df_filtered.groupby(['Date'])['AttributedSales'].sum().cumsum().reset_index()
    fig_h1, ax_h1 = plt.subplots(figsize=(10, 4))
    ax_h1.plot(historic_ts['Date'], historic_ts['AttributedSales'], label='Historic')
    ax_h1.set_title("Cumulative Revenue Trend")
    ax_h1.set_ylabel("£ Revenue")
    ax_h1.set_xlabel("Date")
    st.pyplot(fig_h1)

    # 2. Forecast by Channel with Filters
    st.markdown("### 📊 Forecasted Revenue by Channel")
    st.markdown("Shows forecasted revenue by channel using Scenario 1. Filter by audience, channel, and product.")
    selected_df = scenario_dfs['Scenario 1']
    df_forecast = selected_df[selected_df['Segment'].isin(selected_segments) & selected_df['Channel'].isin(selected_channels)]
    grouped_forecast = df_forecast.groupby('Channel')['SimulatedAttributedSales'].sum().reindex(channels, fill_value=0).reset_index()
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Channel', y='SimulatedAttributedSales', data=grouped_forecast, ax=ax_f1)
    ax_f1.set_title("Forecasted Revenue by Channel")
    ax_f1.set_ylabel("£ Revenue")
    ax_f1.set_xlabel("")
    st.pyplot(fig_f1)

    # 3. Drivers of Performance
    st.markdown("### 📉 Drivers of Performance")
    st.markdown("Breakdown of revenue uplift or decline across contextual drivers relevant to banking.")
    base_sales = df_segment['AttributedSales'].sum()
    scenario_sales = selected_df['SimulatedAttributedSales'].sum()
    delta = scenario_sales - base_sales
    driver_data = pd.DataFrame({
        'Driver': [
            'Base', 'Campaign Uplift', 'Interest Rate', 'Cost of Living',
            'Competitor Activity', 'Segment Shift', 'Forecast'
        ],
        'Impact (£)': [base_sales, 8000, -5000, -4000, -3000, delta - 6000, scenario_sales]
    })
    fig_d, ax_d = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Driver', y='Impact (£)', data=driver_data, palette='coolwarm', ax=ax_d)
    ax_d.set_title("Key Drivers of Performance")
    ax_d.set_ylabel("£ Impact")
    ax_d.set_xlabel("Driver")
    ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=30, ha="right")
    for p in ax_d.patches:
        ax_d.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', fontsize=9)
    st.pyplot(fig_d)

    # 4. Weekly Forecasts (Non-Cumulative)
    st.markdown("### 📆 Weekly Forecasts by Scenario")
    st.markdown("Simulated forward-looking forecast split weekly for each scenario.")
    start_date = pd.to_datetime("today").normalize()
    future_weeks = pd.date_range(start=start_date, periods=forecast_weeks, freq='W-MON')
    forecast_ts = pd.DataFrame({'Week': future_weeks})
    for name in scenario_names:
        total = scenario_dfs[name]['SimulatedAttributedSales'].sum()
        forecast_ts[name] = np.linspace(total * 0.2, total, num=forecast_weeks)
    fig_w, ax_w = plt.subplots(figsize=(10, 4))
    for name in scenario_names:
        ax_w.plot(forecast_ts['Week'], forecast_ts[name], label=name)
    ax_w.set_title("Weekly Forecast Revenue")
    ax_w.set_ylabel("£ Revenue")
    ax_w.set_xlabel("Week Starting")
    ax_w.legend()
    st.pyplot(fig_w)

    # 5. Causal Graph
    st.markdown("### 🧠 Causal Graph - How the Model Works")
    st.markdown("Visual representation of variable relationships driving revenue in the model.")
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ("Promo Activity", "Media Spend", 0.6),
        ("Interest Rate", "Customer Intent", -0.4),
        ("Customer Intent", "Spend", 0.8),
        ("Media Spend", "Spend", 1.0),
        ("Spend", "Revenue", 1.3),
        ("Customer Segment", "Revenue", 0.7),
        ("Competitor Spend", "Revenue", -0.5),
        ("Brand Equity", "Revenue", 0.6)
    ])
    pos = nx.spring_layout(G, seed=42)
    fig_g, ax_g = plt.subplots(figsize=(8, 6))
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, ax=ax_g)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, ax=ax_g)
    st.pyplot(fig_g)
