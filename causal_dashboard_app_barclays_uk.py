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

uploaded_file = st.file_uploader("Upload the Enriched Causal Simulator Excel file", type="xlsx")

if uploaded_file:
    df_segment = pd.read_excel(uploaded_file, sheet_name="Segment Attribution")
    df_weights = pd.read_excel(uploaded_file, sheet_name="Causal Weights")

    channels = sorted(df_segment["Channel"].unique().tolist())
    segments = sorted(df_segment["Segment"].unique().tolist())
    products = sorted(df_segment["ProductCategory"].unique().tolist())
    customer_types = sorted(df_segment["CustomerType"].unique().tolist())
    scenario_names = ["Scenario 1", "Scenario 2", "Scenario 3"]
    scenario_changes = {name: [] for name in scenario_names}

    all_segments = ["All"] + segments
    all_channels = ["All"] + channels
    all_products = ["All"] + products
    all_customers = ["All"] + customer_types

    df_segment["Date"] = pd.to_datetime(df_segment["Date"])

    def simulate(df, changes):
        df = df.copy()
        df = df.merge(df_weights, on="Channel", how="left")
        for seg, chan, prod, cust, mult in changes:
            mask = (
                (df["Segment"] == seg if seg != "All" else True) &
                (df["Channel"] == chan if chan != "All" else True) &
                (df["ProductCategory"] == prod if prod != "All" else True) &
                (df["CustomerType"] == cust if cust != "All" else True)
            )
            df.loc[mask, "Spend"] *= mult
        df["SimulatedAttributedSales"] = df["Spend"] * df["CausalWeight"]
        return df

    st.markdown("### 📈 Revenue by Channel (Cumulative)")
    st.markdown("""
    Cumulative revenue over time for selected filters.  
    **Use case**: See how revenue is pacing weekly.  
    **Interpretation**: Upward trajectory is expected; plateaus suggest performance gaps.  
    **Action**: Dive into weeks with stagnation or unusual spikes.
    """)

    filters = st.columns(4)
    with filters[0]: selected_channels = st.multiselect("Channels", channels, default=channels)
    with filters[1]: selected_segments = st.multiselect("Segments", segments, default=segments)
    with filters[2]: selected_products = st.multiselect("Products", products, default=products)
    with filters[3]: selected_customers = st.multiselect("Customer Type", customer_types, default=customer_types)

    df_filtered = df_segment[
        df_segment["Channel"].isin(selected_channels) &
        df_segment["Segment"].isin(selected_segments) &
        df_segment["ProductCategory"].isin(selected_products) &
        df_segment["CustomerType"].isin(selected_customers)
    ]
    ts = df_filtered.groupby("Date")["AttributedSales"].sum().cumsum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(ts["Date"], ts["AttributedSales"]/1e6, label="Cumulative Revenue")
    ax1.set_ylabel("£ Revenue (millions)")
    ax1.set_title("Cumulative Revenue Over Time")
    ax1.tick_params(axis='x', rotation=30)
    st.pyplot(fig1)

    st.markdown("### 📈 Revenue by Week")
    st.markdown("""
    Weekly revenue trends.  
    **Use case**: Identify cyclical effects, campaign impact timing.  
    **Interpretation**: Peaks/dips may align with promo cycles or sponsorships.  
    **Action**: Shift budget based on timing performance.
    """)
    ts_weekly = df_filtered.groupby("Date")["AttributedSales"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ts_weekly["Date"], ts_weekly["AttributedSales"]/1e6)
    ax2.set_ylabel("£ Revenue (millions)")
    ax2.set_title("Weekly Revenue")
    ax2.tick_params(axis='x', rotation=30)
    st.pyplot(fig2)

    st.markdown("### 📊 Cumulative Revenue by Channel (Stacked Area)")
    st.markdown("""
    Pacing of cumulative revenue by channel.  
    **Use case**: See mix and evolution over time.  
    **Interpretation**: Growing areas indicate expanding channel influence.  
    **Action**: Rebalance investment toward channels growing slower than expected.
    """)
    area_data = df_filtered.groupby(["Date", "Channel"])["AttributedSales"].sum().unstack().fillna(0).cumsum()
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    area_data = area_data[channels] if set(channels).issubset(area_data.columns) else area_data
    ax3.stackplot(area_data.index, area_data.T / 1e6, labels=area_data.columns)
    ax3.set_ylabel("£ Revenue (millions)")
    ax3.set_title("Cumulative Revenue by Channel")
    ax3.legend(loc="upper left")
    st.pyplot(fig3)

    st.markdown("### 🔧 Scenario Planner")
    st.markdown("""
    Adjust media allocations by audience, channel, product and customer type.  
    **Use case**: Simulate investment decisions.  
    **Interpretation**: Rows define overrides applied in forecast scenarios.  
    **Action**: Build strategies by compounding adjustments.
    """)
    for scenario in scenario_names:
        with st.expander(f"{scenario} Adjustments"):
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 1])
            with col1: seg = st.selectbox(f"Segment ({scenario})", all_segments, key=f"{scenario}_seg")
            with col2: chan = st.selectbox(f"Channel ({scenario})", all_channels, key=f"{scenario}_chan")
            with col3: prod = st.selectbox(f"Product ({scenario})", all_products, key=f"{scenario}_prod")
            with col4: cust = st.selectbox(f"Customer ({scenario})", all_customers, key=f"{scenario}_cust")
            with col5: mult = st.slider("Multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{scenario}_mult")
            with col6:
                if st.button("Add", key=f"{scenario}_add"):
                    scenario_changes[scenario].append((seg, chan, prod, cust, mult))
            if scenario_changes[scenario]:
                df_adj = pd.DataFrame(scenario_changes[scenario], columns=["Segment", "Channel", "Product", "Customer", "Multiplier"])
                st.dataframe(df_adj)

    st.markdown("### 📈 Forecasted Revenue by Scenario")
    st.markdown("""
    Compare projected uplift by scenario.  
    **Use case**: Evaluate plan impact.  
    **Interpretation**: Scenarios vs baseline = expected revenue change.  
    **Action**: Choose top-performing paths.
    """)
    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1)
    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names}
    def forecast(df, weeks): return df.sort_values("Date").groupby("Date")["SimulatedAttributedSales"].sum().head(weeks).sum() / 1e6
    df_total = pd.DataFrame({
        "Scenario": scenario_names,
        "Revenue (£m)": [forecast(df, forecast_weeks) for df in scenario_dfs.values()]
    })
    baseline = forecast(df_segment, forecast_weeks)
    df_total.loc[len(df_total.index)] = ["Baseline", baseline]
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_total, x="Scenario", y="Revenue (£m)", ax=ax4)
    ax4.set_title("Forecast vs Baseline")
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f}m", (p.get_x() + p.get_width()/2., p.get_height()), ha="center")
    st.pyplot(fig4)

    st.markdown("### 🧠 Causal Graph")
    st.markdown("""
    Visual structure of model logic.  
    **Use case**: Understand cause-effect.  
    **Interpretation**: Arrows = direction; weights = influence strength.  
    **Action**: Prioritize nodes with strongest positive or negative weights.
    """)
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ("Promo", "Spend", 0.8), ("Interest Rate", "Demand", -0.5),
        ("Demand", "Spend", 0.6), ("Spend", "Revenue", 1.2),
        ("Customer Segment", "Revenue", 0.9), ("Brand Equity", "Revenue", 0.7),
        ("Competitor Spend", "Revenue", -0.4), ("Search Trends", "Brand Equity", 0.5)
    ])
    pos = nx.spring_layout(G, seed=42, k=2.2)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2800, font_size=9, ax=ax5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=9)
    st.pyplot(fig5)
