import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

st.set_page_config(layout="wide")
sns.set(style='whitegrid')

st.title("📊 Barclays Consumer Banking – Marketing Optimization Dashboard")

uploaded_file = st.file_uploader("Upload the enriched simulator Excel", type="xlsx")

if uploaded_file:
    df_segment = pd.read_excel(uploaded_file, sheet_name="Segment Attribution")
    df_weights = pd.read_excel(uploaded_file, sheet_name="Causal Weights")

    # Standard setup
    channels = sorted(df_segment["Channel"].unique().tolist())
    segments = sorted(df_segment["Segment"].unique().tolist())
    products = sorted(df_segment["ProductCategory"].unique().tolist())
    customer_types = sorted(df_segment["CustomerType"].unique().tolist())

    all_channels = ["All"] + channels
    all_segments = ["All"] + segments
    all_products = ["All"] + products
    all_customers = ["All"] + customer_types

    scenario_names = ["Scenario 1", "Scenario 2", "Scenario 3"]
    scenario_changes = {name: [] for name in scenario_names}

    df_segment["Date"] = pd.to_datetime(df_segment["Date"])

    # Simulation logic
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
    # Filtered view for base visuals
    st.markdown("### 📈 Revenue by Channel (Cumulative)")
    st.markdown("""
    This chart shows cumulative revenue over time by week.  
    **Use case**: Detect pacing trends.  
    **Interpretation**: Upward slope shows revenue momentum.  
    **Action**: Investigate plateaus or dips in growth.
    """)
    f1, f2, f3, f4 = st.columns(4)
    with f1: selected_channels = st.multiselect("Channels", channels, default=channels)
    with f2: selected_segments = st.multiselect("Segments", segments, default=segments)
    with f3: selected_products = st.multiselect("Products", products, default=products)
    with f4: selected_customers = st.multiselect("Customer Type", customer_types, default=customer_types)

    df_filtered = df_segment[
        df_segment["Channel"].isin(selected_channels) &
        df_segment["Segment"].isin(selected_segments) &
        df_segment["ProductCategory"].isin(selected_products) &
        df_segment["CustomerType"].isin(selected_customers)
    ]

    # Cumulative revenue
    ts_cum = df_filtered.groupby("Date")["AttributedSales"].sum().cumsum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(ts_cum["Date"], ts_cum["AttributedSales"] / 1e6)
    ax1.set_title("Cumulative Revenue Over Time")
    ax1.set_ylabel("£ Revenue (millions)")
    ax1.tick_params(axis='x', rotation=30)
    st.pyplot(fig1)

    # Weekly revenue
    st.markdown("### 📉 Revenue by Week")
    st.markdown("""
    This chart shows weekly revenue without accumulation.  
    **Use case**: Spot campaign spikes or seasonal drops.  
    **Interpretation**: Peaks may indicate promotions or events.  
    **Action**: Align high-performing weeks with campaign calendar.
    """)
    ts_week = df_filtered.groupby("Date")["AttributedSales"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ts_week["Date"], ts_week["AttributedSales"] / 1e6)
    ax2.set_title("Weekly Revenue")
    ax2.set_ylabel("£ Revenue (millions)")
    ax2.tick_params(axis='x', rotation=30)
    st.pyplot(fig2)

    # Stacked area by channel
    st.markdown("### 📊 Cumulative Revenue by Channel")
    st.markdown("""
    Cumulative revenue broken down by channel.  
    **Use case**: Understand mix shift over time.  
    **Interpretation**: Flattening areas = underperforming channels.  
    **Action**: Rebalance based on contribution trends.
    """)
    area_data = df_filtered.groupby(["Date", "Channel"])["AttributedSales"].sum().unstack().fillna(0).cumsum()
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    area_data = area_data[channels] if set(channels).issubset(area_data.columns) else area_data
    ax3.stackplot(area_data.index, area_data.T / 1e6, labels=area_data.columns)
    ax3.set_ylabel("£ Revenue (millions)")
    ax3.set_title("Cumulative Revenue by Channel")
    ax3.legend(loc="upper left")
    st.pyplot(fig3)

    # Scenario Planner
    st.markdown("### 🔧 Scenario Planner")
    st.markdown("""
    Modify budget by audience, channel, product, and customer type.  
    **Use case**: Build custom strategy scenarios.  
    **Interpretation**: Each row = one adjustment applied to simulation.  
    **Action**: Add multiple rows to simulate complex shifts.
    """)
    for scenario in scenario_names:
        with st.expander(f"{scenario} Adjustments"):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1: seg = st.selectbox(f"Segment ({scenario})", all_segments, key=f"{scenario}_seg")
            with c2: chan = st.selectbox(f"Channel ({scenario})", all_channels, key=f"{scenario}_chan")
            with c3: prod = st.selectbox(f"Product ({scenario})", all_products, key=f"{scenario}_prod")
            with c4: cust = st.selectbox(f"Customer ({scenario})", all_customers, key=f"{scenario}_cust")
            with c5: mult = st.slider("Multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{scenario}_mult")
            with c6:
                if st.button("Add", key=f"{scenario}_add"):
                    scenario_changes[scenario].append((seg, chan, prod, cust, mult))
            if scenario_changes[scenario]:
                df_adj = pd.DataFrame(scenario_changes[scenario], columns=["Segment", "Channel", "Product", "Customer", "Multiplier"])
                st.dataframe(df_adj)
    # Forecasted Revenue
    st.markdown("### 📈 Forecasted Revenue by Scenario")
    st.markdown("""
    Projected revenue impact by strategy scenario.  
    **Use case**: Evaluate which media changes offer biggest upside.  
    **Interpretation**: Compare scenarios to baseline.  
    **Action**: Use best scenario as media recommendation.
    """)
    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1)
    scenario_dfs = {name: simulate(df_segment.copy(), scenario_changes[name]) for name in scenario_names}

    def get_forecast(df, weeks):
        if "SimulatedAttributedSales" not in df.columns:
            df = df.merge(df_weights, on="Channel", how="left")
            df["SimulatedAttributedSales"] = df["Spend"] * df["CausalWeight"]
        df = df.sort_values("Date")
        return df.groupby("Date")["SimulatedAttributedSales"].sum().head(weeks).sum() / 1e6

    df_total = pd.DataFrame({
        "Scenario": scenario_names,
        "Revenue (£m)": [get_forecast(df, forecast_weeks) for df in scenario_dfs.values()]
    })
    baseline = get_forecast(df_segment, forecast_weeks)
    df_total.loc[len(df_total.index)] = ["Baseline", baseline]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_total, x="Scenario", y="Revenue (£m)", ax=ax4)
    ax4.set_title("Forecast vs Baseline")
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f}m", (p.get_x() + p.get_width()/2., p.get_height()), ha="center")
    st.pyplot(fig4)

    # Competitor Impact Summary
    st.markdown("### 📋 Competitor Impact Summary")
    st.markdown("""
    Shows the estimated revenue impact of major UK banking competitors.  
    **Use case**: Identify where competitor pressure is strongest.  
    **Interpretation**: Negative = lost revenue to rival activity.  
    **Action**: Allocate defense spend or investigate causes.
    """)
    competitors = ["HSBC", "Lloyds", "NatWest", "Santander", "Monzo", "Revolut"]
    impacts = [-130000, -110000, -85000, -60000, -25000, 10000]
    df_comp = pd.DataFrame({"Competitor": competitors, "Impact (£)": impacts})
    fig5, ax5 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=df_comp, x="Impact (£)", y="Competitor", palette="RdBu", ax=ax5)
    ax5.set_title("Revenue Impact by Competitor")
    st.pyplot(fig5)

    # Competitor Breakdown
    st.markdown("### 🔍 Competitor Impact Breakdown")
    st.markdown("""
    View individual competitor's drivers of impact.  
    **Use case**: Understand *why* a brand is winning or losing share.  
    **Interpretation**: Channel/strategy-level breakdown.  
    **Action**: Tailor messaging or counter strategies.
    """)
    selected_comp = st.selectbox("Select Competitor", competitors)
    detail = pd.DataFrame({
        "Driver": ["Media Spend", "Promotions", "Brand Consideration", "Pricing"],
        "Impact (£)": [-50000, -30000, -20000, -10000] if selected_comp != "Revolut" else [5000, 3000, 2000, 1000]
    })
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=detail, x="Impact (£)", y="Driver", palette="crest", ax=ax6)
    ax6.set_title(f"{selected_comp} – Impact Drivers")
    st.pyplot(fig6)

    # Causal Graph
    st.markdown("### 🧠 Causal Model Diagram")
    st.markdown("""
    This diagram visualizes how external and internal drivers affect outcomes.  
    **Use case**: Explainability of uplift & attribution logic.  
    **Interpretation**: Arrows indicate influence direction; weights show strength.  
    **Action**: Focus strategy on strong causal levers.
    """)
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ("Promo", "Spend", 0.8), ("Interest Rate", "Demand", -0.5),
        ("Demand", "Spend", 0.6), ("Spend", "Revenue", 1.2),
        ("Customer Segment", "Revenue", 0.9), ("Brand Equity", "Revenue", 0.7),
        ("Competitor Spend", "Revenue", -0.4), ("Search Trends", "Brand Equity", 0.5)
    ])
    pos = nx.spring_layout(G, seed=42, k=2.2)
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2800, font_size=9, ax=ax7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=9)
    st.pyplot(fig7)
