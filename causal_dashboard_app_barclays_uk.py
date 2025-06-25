# causal_dashboard_app_barclays_uk.py
# Final version with all enhancements and no chart omissions

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
    df_weights = pd.read_excel(uploaded_file, sheet_name="Causal Weights")

    if "ProductCategory" not in df_segment.columns:
        df_segment["ProductCategory"] = "Unknown"

    channels = ["TV", "Paid Social", "Email", "Push", "Branch", "Display", "Search"]
    df_segment["Channel"] = pd.Categorical(df_segment["Channel"], categories=channels, ordered=True)
    segments = df_segment["Segment"].unique().tolist()
    products = df_segment["ProductCategory"].unique().tolist()
    scenario_names = ["Scenario 1", "Scenario 2", "Scenario 3"]
    scenario_changes = {name: {} for name in scenario_names}
    adjustment_state = {name: [] for name in scenario_names}

    df_segment["Date"] = pd.to_datetime(df_segment["Date"])

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df["Segment"] == seg) & (df["Channel"] == chan), "Spend"] *= mult
        df = df.merge(df_weights, on="Channel", how="left")
        df["SimulatedAttributedSales"] = df["Spend"] * df["CausalWeight"]
        return df

    # === Revenue by Channel (Cumulative)
    st.markdown("### 📈 Revenue by Channel (Cumulative)")
    selected_channels = st.multiselect("Select Channels", channels, default=channels)
    selected_segments = st.multiselect("Select Segments", segments, default=segments)
    selected_products = st.multiselect("Select Products", products, default=products)

    df_filtered = df_segment[
        df_segment["Channel"].isin(selected_channels) &
        df_segment["Segment"].isin(selected_segments) &
        df_segment["ProductCategory"].isin(selected_products)
    ]

    ts = df_filtered.sort_values("Date").groupby("Date")["AttributedSales"].sum().cumsum().reset_index()
    compare_mode = st.radio("Compare to:", ['None', 'Forecast', 'Last Year'], key="rev_compare")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(ts["Date"], ts["AttributedSales"] / 1e6, label="Actual")
    if compare_mode == "Forecast":
        ax1.plot(ts["Date"], ts["AttributedSales"] * 1.05 / 1e6, label="Forecast", linestyle='--')
    elif compare_mode == "Last Year":
        ax1.plot(ts["Date"], ts["AttributedSales"] * 0.95 / 1e6, label="Last Year", linestyle='--')
    ax1.set_title("Cumulative Revenue Over Time")
    ax1.set_ylabel("£ Revenue (millions)")
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend()
    st.pyplot(fig1)

    # === Total Revenue by Channel
    st.markdown("### 📊 Total Revenue by Channel")
    agg = df_filtered.groupby("Channel")["AttributedSales"].sum().reindex(channels, fill_value=0).reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=agg, x="Channel", y="AttributedSales", ax=ax2)
    ax2.set_ylabel("£ Revenue")
    ax2.set_title("Total Revenue by Channel")
    ax2.ticklabel_format(style='plain', axis='y')
    st.pyplot(fig2)

    # === Waterfall Chart
    st.markdown("### 📉 Revenue Drivers - Waterfall")
    channel_contrib = df_filtered.groupby("Channel")["AttributedSales"].sum().reindex(channels, fill_value=0) / 1e6
    macro = {"Interest Rate": -0.15, "Competition": -0.2, "Segment Shift": 0.25}
    waterfall = [("Baseline", 1.0)] + list(channel_contrib.items()) + list(macro.items())
    total = sum(val for _, val in waterfall)
    waterfall.append(("Total", total))
    wf = pd.DataFrame(waterfall, columns=["Driver", "Value"])
    fig3, ax3 = plt.subplots(figsize=(16, 5))
    sns.barplot(data=wf, x="Driver", y="Value", palette="coolwarm", ax=ax3)
    ax3.set_ylabel("£ Value (millions)")
    ax3.set_title("Revenue Waterfall by Driver")
    ax3.ticklabel_format(style='plain', axis='y')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.1f}m", (p.get_x() + p.get_width()/2., p.get_height()), ha="center")
    st.pyplot(fig3)

    # === Competitor Summary
    st.markdown("### 📋 Competitor Impact Summary")
    st.markdown("This chart shows total estimated revenue loss or gain from key competitors over the period.")
    competitors = ["HSBC", "Lloyds", "NatWest", "Santander", "Monzo", "Revolut"]
    impacts = [-130000, -110000, -85000, -60000, -25000, 10000]
    df_comp = pd.DataFrame({"Competitor": competitors, "Impact (£)": impacts})
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=df_comp, x="Impact (£)", y="Competitor", palette="RdBu", ax=ax4)
    ax4.set_title("Revenue Impact by Competitor")
    st.pyplot(fig4)

    # === Competitor Breakdown
    st.markdown("### 🔎 Competitor Impact Breakdown")
    selected_comp = st.selectbox("Select Competitor", competitors)
    detail = pd.DataFrame({
        "Driver": ["Media Spend", "Promotions", "Brand Consideration", "Pricing"],
        "Impact (£)": [-50000, -30000, -20000, -10000] if selected_comp != "Revolut" else [5000, 3000, 2000, 1000]
    })
    fig5, ax5 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=detail, x="Impact (£)", y="Driver", palette="crest", ax=ax5)
    ax5.set_title(f"{selected_comp} - Impact Drivers")
    st.pyplot(fig5)

    # === Scenario Planner
    st.markdown("### 🔧 Scenario Planner")
    st.markdown("Adjust segment × channel spend. 2.0 = double spend, 0.0 = no spend.")
    for scenario in scenario_names:
        with st.expander(f"{scenario} Adjustments"):
            col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
            with col1:
                seg = st.selectbox(f"Segment ({scenario})", segments, key=f"{scenario}_seg")
            with col2:
                chan = st.selectbox(f"Channel ({scenario})", channels, key=f"{scenario}_chan")
            with col3:
                mult = st.slider("Multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{scenario}_mult")
            with col4:
                if st.button("Add", key=f"{scenario}_add"):
                    scenario_changes[scenario][(seg, chan)] = mult
                    adjustment_state[scenario].append((seg, chan, mult))

            if adjustment_state[scenario]:
                st.dataframe(pd.DataFrame(adjustment_state[scenario], columns=["Segment", "Channel", "Multiplier"]))

    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names}

    # === Forecast Comparison
    st.markdown("### 📈 Forecasted Revenue by Scenario")
    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1)
    def get_forecast(df, weeks):
        df = df.sort_values("Date")
        return df.groupby("Date")["SimulatedAttributedSales"].sum().head(weeks).sum() / 1e6

    totals = {
        "Scenario": scenario_names,
        "Revenue (£m)": [get_forecast(df, forecast_weeks) for df in scenario_dfs.values()]
    }
    baseline = df_segment.sort_values("Date").groupby("Date")["AttributedSales"].sum().head(forecast_weeks).sum() / 1e6
    df_total = pd.DataFrame(totals)
    df_total.loc[len(df_total.index)] = ["Baseline", baseline]
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_total, x="Scenario", y="Revenue (£m)", ax=ax6)
    ax6.set_title("Forecast vs Baseline")
    for p in ax6.patches:
        ax6.annotate(f"{p.get_height():.1f}m", (p.get_x() + p.get_width()/2., p.get_height()), ha="center")
    st.pyplot(fig6)

    # === Causal Graph
    st.markdown("### 🧠 Causal Graph")
    st.markdown("Causal relationships between drivers and revenue.")
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
    pos = nx.spring_layout(G, seed=42, k=2.2)
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax7, node_color='skyblue', node_size=3000)
    nx.draw_networkx_edges(G, pos, ax=ax7, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=9)
    st.pyplot(fig7)
