# causal_dashboard_app_barclays_uk.py
# Final: Fixes KeyError, restores all visuals, formats revenue, and improves layout

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

st.set_page_config(layout="wide")
sns.set(style='whitegrid')

st.title("📊 Barclays Consumer Banking - Marketing Optimization Dashboard")

uploaded_file = st.file_uploader("Upload the Barclays causal simulator Excel file", type="xlsx")

if uploaded_file:
    df_segment = pd.read_excel(uploaded_file, sheet_name="Segment Attribution")
    df_product = pd.read_excel(uploaded_file, sheet_name="Product Attribution")
    df_weights = pd.read_excel(uploaded_file, sheet_name="Causal Weights")

    # ✅ Fix: ensure ProductCategory exists BEFORE referencing it
    if "ProductCategory" not in df_segment.columns:
        df_segment["ProductCategory"] = "Unknown"

    channels = ["TV", "Paid Social", "Email", "Push", "Branch", "Display", "Search"]
    df_segment["Channel"] = pd.Categorical(df_segment["Channel"], categories=channels, ordered=True)
    segments = df_segment["Segment"].unique().tolist()
    products = df_segment["ProductCategory"].unique().tolist()
    scenario_names = ["Scenario 1", "Scenario 2", "Scenario 3"]
    scenario_changes = {name: {} for name in scenario_names}
    adjustment_state = {name: [] for name in scenario_names}

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df["Segment"] == seg) & (df["Channel"] == chan), "Spend"] *= mult
        df = df.merge(df_weights, on="Channel", how="left")
        df["SimulatedAttributedSales"] = df["Spend"] * df["CausalWeight"]
        return df

    # === Revenue by Channel (Cumulative) ===
    st.markdown("### 📈 Revenue by Channel (Cumulative)")
    df_segment["Date"] = pd.to_datetime(df_segment["Date"])
    selected_channels = st.multiselect("Select Channels", channels, default=channels)
    selected_segments = st.multiselect("Select Segments", segments, default=segments)
    selected_products = st.multiselect("Select Products", products, default=products)

    df_filtered = df_segment[
        df_segment["Channel"].isin(selected_channels) &
        df_segment["Segment"].isin(selected_segments) &
        df_segment["ProductCategory"].isin(selected_products)
    ].sort_values("Date")

    historic = df_filtered.groupby("Date")["AttributedSales"].sum().cumsum().reset_index()
    compare_mode = st.radio("Compare to:", ['None', 'Forecast', 'Last Year'], key="rev_compare")
    fig_rc, ax_rc = plt.subplots(figsize=(10, 4))
    ax_rc.plot(historic["Date"], historic["AttributedSales"] / 1000, label="Actual")
    if compare_mode == "Forecast":
        ax_rc.plot(historic["Date"], historic["AttributedSales"] * 1.1 / 1000, label="Forecast", linestyle='--')
    elif compare_mode == "Last Year":
        ax_rc.plot(historic["Date"], historic["AttributedSales"] * 0.95 / 1000, label="Last Year", linestyle='--')
    ax_rc.set_title("Cumulative Revenue Over Time")
    ax_rc.set_ylabel("£ Revenue (000s)")
    ax_rc.tick_params(axis="x", rotation=30)
    ax_rc.legend()
    st.pyplot(fig_rc)

    # === Total Revenue by Channel ===
    st.markdown("### 📊 Total Revenue by Channel")
    agg = df_filtered.groupby("Channel")["AttributedSales"].sum().reindex(channels, fill_value=0).reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=agg, x="Channel", y="AttributedSales", ax=ax2)
    ax2.set_ylabel("£ Revenue")
    ax2.set_title("Total Revenue by Channel")
    st.pyplot(fig2)

    # === Revenue Waterfall ===
    st.markdown("### 📉 Revenue Drivers - Waterfall")
    channel_totals = df_filtered.groupby("Channel")["AttributedSales"].sum().reindex(channels, fill_value=0)
    macro = {"Interest Rate": -2500, "Competition": -3000, "Segment Shift": 4000}
    waterfall = [("Baseline", 100000)] + list(channel_totals.items()) + list(macro.items())
    total = sum(val for _, val in waterfall)
    waterfall.append(("Total", total))
    wf = pd.DataFrame(waterfall, columns=["Driver", "Value"])
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.barplot(data=wf, x="Driver", y="Value", ax=ax3, palette="coolwarm")
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height()/1000:.0f}", (p.get_x() + p.get_width()/2., p.get_height()), ha="center")
    ax3.set_ylabel("£ Value")
    ax3.set_title("Revenue Waterfall by Driver")
    st.pyplot(fig3)

    # === Competitor Impact Summary ===
    st.markdown("### 📋 Competitor Impact Summary")
    competitors = ["HSBC", "Lloyds", "NatWest", "Santander", "Monzo", "Revolut"]
    impacts = [-130000, -110000, -85000, -60000, -25000, 10000]
    competitor_df = pd.DataFrame({"Competitor": competitors, "Impact (£)": impacts})
    fig_comp, ax_comp = plt.subplots(figsize=(10, 3))
    sns.barplot(data=competitor_df, x="Impact (£)", y="Competitor", palette="RdBu", ax=ax_comp)
    ax_comp.set_title("Revenue Impact by Competitor")
    st.pyplot(fig_comp)

    # === Competitor Breakdown ===
    st.markdown("### 🔎 Competitor Impact Breakdown")
    selected_comp = st.selectbox("Select Competitor", competitors)
    breakdown = pd.DataFrame({
        "Driver": ["Media Spend", "Promotions", "Brand Buzz", "Pricing"],
        "Impact (£)": [-50000, -30000, -20000, -10000] if selected_comp != "Revolut" else [5000, 3000, 2000, 1000]
    })
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=breakdown, x="Impact (£)", y="Driver", palette="crest", ax=ax4)
    ax4.set_title(f"{selected_comp} - Impact Drivers")
    st.pyplot(fig4)

    # === Scenario Planner ===
    st.markdown("### 🔧 Scenario Planner")
    st.markdown("Use the controls below to add specific adjustments. A multiplier of 2.0 means doubling spend.")

    for scenario in scenario_names:
        with st.expander(f"{scenario} Adjustments"):
            col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
            with col1:
                selected_segment = st.selectbox(f"Segment ({scenario})", segments, key=f"{scenario}_seg")
            with col2:
                selected_channel = st.selectbox(f"Channel ({scenario})", channels, key=f"{scenario}_chan")
            with col3:
                multiplier = st.slider("Spend Multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{scenario}_mult")
            with col4:
                if st.button("Add", key=f"{scenario}_add"):
                    scenario_changes[scenario][(selected_segment, selected_channel)] = multiplier
                    adjustment_state[scenario].append((selected_segment, selected_channel, multiplier))

            if adjustment_state[scenario]:
                st.dataframe(pd.DataFrame(adjustment_state[scenario], columns=["Segment", "Channel", "Multiplier"]))

    scenario_dfs = {
        name: simulate(df_segment.copy(), scenario_changes[name])
        for name in scenario_names
    }

    # === Forecast Comparison ===
    st.markdown("### 📈 Forecasted Revenue by Scenario")
    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1, key="horizon_slider")
    totals = pd.DataFrame({
        "Scenario": scenario_names,
        "Revenue (£000s)": [df["SimulatedAttributedSales"].sum()/1000 for df in scenario_dfs.values()]
    })
    baseline = df_segment["AttributedSales"].sum()/1000
    totals.loc[len(totals.index)] = ["Baseline", baseline]
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=totals, x="Scenario", y="Revenue (£000s)", ax=ax5)
    ax5.set_title("Forecast vs Baseline")
    for p in ax5.patches:
        ax5.annotate(f"{p.get_height():,.0f}", (p.get_x()+p.get_width()/2., p.get_height()), ha="center")
    st.pyplot(fig5)

    # === Causal Graph ===
    st.markdown("### 🧠 Causal Graph")
    st.markdown("Visual representation of how variables influence each other.")
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
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    nx.draw_networkx(G, pos, ax=ax6, node_color='skyblue', node_size=2000, with_labels=True, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax6)
    st.pyplot(fig6)
