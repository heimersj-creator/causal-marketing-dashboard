# causal_dashboard_app_barclays_uk.py
# Simplified scenario planner UI + forecast slider moved inline

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

    # ✅ Fallback if ProductCategory is missing
    if 'ProductCategory' not in df_segment.columns:
        df_segment['ProductCategory'] = 'Unknown'

    channels = ["TV", "Paid Social", "Email", "Push", "Branch", "Display", "Search"]
    df_segment['Channel'] = pd.Categorical(df_segment['Channel'], categories=channels, ordered=True)

    segments = df_segment['Segment'].unique().tolist()
    products = df_segment['ProductCategory'].unique().tolist()
    scenario_names = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    scenario_changes = {name: {} for name in scenario_names}
    adjustment_state = {name: [] for name in scenario_names}

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    # ==== Scenario Planner ====
    st.markdown("### 🔧 Scenario Planner")
    st.markdown("Use the controls below to add specific adjustments to each scenario.")

    for scenario in scenario_names:
        with st.expander(f"{scenario} Adjustments", expanded=False):
            col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
            with col1:
                selected_segment = st.selectbox(f"Select Segment ({scenario})", segments, key=f"{scenario}_seg")
            with col2:
                selected_channel = st.selectbox(f"Select Channel ({scenario})", channels, key=f"{scenario}_chan")
            with col3:
                multiplier = st.slider("Spend Multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{scenario}_mult")
            with col4:
                if st.button(f"Add Adjustment ({scenario})", key=f"{scenario}_add"):
                    scenario_changes[scenario][(selected_segment, selected_channel)] = multiplier
                    adjustment_state[scenario].append((selected_segment, selected_channel, multiplier))

            # Show applied adjustments
            if adjustment_state[scenario]:
                st.markdown("#### Active Adjustments")
                adj_df = pd.DataFrame(adjustment_state[scenario], columns=["Segment", "Channel", "Multiplier"])
                st.dataframe(adj_df, use_container_width=True)

    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names]

    # === Forecast Chart ===
    st.markdown("### 📈 Forecasted Revenue by Scenario")
    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1, key="horizon_slider")

    scenario_totals = pd.DataFrame({
        'Scenario': scenario_names,
        'Revenue (£)': [df['SimulatedAttributedSales'].sum() for df in scenario_dfs.values()]
    })
    fig_scen, ax_scen = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Scenario', y='Revenue (£)', data=scenario_totals, ax=ax_scen)
    ax_scen.set_title("Forecasted Revenue by Scenario")
    for p in ax_scen.patches:
        ax_scen.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2., p.get_height()), ha='center')
    st.pyplot(fig_scen)

    # === Causal Graph ===
    st.markdown("### 🧠 Causal Graph")
    st.markdown("This diagram visualises the structure of how variables like spend, segment, and demand drive revenue.")

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
