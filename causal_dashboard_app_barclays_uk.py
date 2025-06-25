# causal_dashboard_app_barclays_uk.py
# Streamlit app with enhanced layout, cleaner charts, restored competitor visuals, and a revenue waterfall

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
    scenario_names = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    scenario_changes = {name: {} for name in scenario_names}

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1, key="horizon")

    st.markdown("### 🔧 Scenario Planner")
    st.markdown("Adjust channel × segment combinations in each scenario.")
    col_scenarios = st.columns(len(scenario_names))
    for idx, name in enumerate(scenario_names):
        with col_scenarios[idx]:
            st.markdown(f"**{name}**")
            for seg in segments:
                for ch in channels:
                    mult = st.slider(f"{seg} × {ch}", 0.0, 2.0, 1.0, 0.1, key=f"{name}_{seg}_{ch}")
                    scenario_changes[name][(seg, ch)] = mult

    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names}

    st.markdown("### 📈 Revenue by Channel (Cumulative)")
    st.markdown("Historic cumulative revenue by channel with filters and comparison.")
    df_segment['Date'] = pd.date_range(start='2024-01-01', periods=len(df_segment), freq='W')
    filtered = df_segment[df_segment['Channel'].isin(channels)]
    historic = filtered.groupby('Date')['AttributedSales'].sum().cumsum().reset_index()
    compare_mode = st.radio("Compare to:", ['None', 'Forecast', 'Last Year'], key="rev_compare")
    fig_rc, ax_rc = plt.subplots(figsize=(10, 4))
    ax_rc.plot(historic['Date'], historic['AttributedSales'], label="Actual")
    if compare_mode == 'Forecast':
        ax_rc.plot(historic['Date'], historic['AttributedSales'] * 1.05, label="Forecast", linestyle='--')
    elif compare_mode == 'Last Year':
        ax_rc.plot(historic['Date'], historic['AttributedSales'] * 0.95, label="Last Year", linestyle='--')
    ax_rc.set_title("Revenue by Channel")
    ax_rc.set_ylabel("£ Revenue")
    ax_rc.tick_params(axis='x', rotation=30)
    ax_rc.legend()
    st.pyplot(fig_rc)

    st.markdown("### 📊 Total Revenue by Channel")
    channel_grouped = df_segment.groupby('Channel')['AttributedSales'].sum().reindex(channels, fill_value=0).reset_index()
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Channel', y='AttributedSales', data=channel_grouped, ax=ax_bar)
    ax_bar.set_title("Total Revenue by Channel")
    ax_bar.set_ylabel("£ Revenue")
    st.pyplot(fig_bar)

    st.markdown("### 📉 Revenue Drivers - Waterfall")
    st.markdown("Waterfall showing additive contribution of key drivers.")
    baseline = 100000
    uplift = 8000
    interest = -5000
    competition = -4000
    segment = 6000
    total = baseline + uplift + interest + competition + segment
    waterfall_data = pd.DataFrame({
        'Driver': ['Baseline', 'Uplift', 'Interest', 'Competition', 'Segment Shift', 'Total'],
        'Value': [baseline, uplift, interest, competition, segment, total]
    })
    fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Driver', y='Value', data=waterfall_data, palette='coolwarm', ax=ax_wf)
    ax_wf.set_title("Revenue Waterfall by Driver")
    for p in ax_wf.patches:
        ax_wf.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2., p.get_height()), ha='center')
    st.pyplot(fig_wf)

    st.markdown("### 📋 Competitor Impact Summary")
    competitors = ['HSBC', 'Lloyds', 'NatWest', 'Santander', 'Monzo', 'Revolut']
    impacts = [-130000, -110000, -85000, -60000, -25000, 10000]
    competitor_df = pd.DataFrame({'Competitor': competitors, 'Impact (£)': impacts})
    fig_comp, ax_comp = plt.subplots(figsize=(10, 3))
    sns.barplot(x='Impact (£)', y='Competitor', data=competitor_df, palette='RdBu', ax=ax_comp)
    ax_comp.set_title("Revenue Impact by Competitor")
    st.pyplot(fig_comp)

    st.markdown("### 🔎 Competitor Impact Breakdown")
    selected_comp = st.selectbox("Select Competitor", competitors)
    comp_factors = pd.DataFrame({
        'Driver': ['Media Spend', 'Promotions', 'Brand Buzz', 'Pricing'],
        'Impact (£)': [-50000, -30000, -20000, -10000] if selected_comp != 'Revolut' else [5000, 3000, 2000, 1000]
    })
    fig_sub, ax_sub = plt.subplots(figsize=(10, 3))
    sns.barplot(x='Impact (£)', y='Driver', data=comp_factors, palette='crest', ax=ax_sub)
    ax_sub.set_title(f"{selected_comp} - Impact Drivers")
    st.pyplot(fig_sub)

    st.markdown("### 🧠 Causal Graph")
    st.markdown("Network graph illustrating weighted causal relationships in the model.")
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
    pos = nx.spring_layout(G, seed=42, k=1.5)
    fig_graph, ax_graph = plt.subplots(figsize=(10, 6))
    nx.draw_networkx(G, pos, ax=ax_graph, node_color='skyblue', node_size=2000, with_labels=True, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_graph)
    st.pyplot(fig_graph)
