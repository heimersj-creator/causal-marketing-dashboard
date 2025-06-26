import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import uuid

st.set_page_config(layout="wide")
sns.set(style='whitegrid')

st.title("📊 Barclays Consumer Banking – Marketing Optimization Dashboard")

uploaded_file = st.file_uploader("Upload the enriched simulator Excel", type="xlsx")

if uploaded_file:
    # Load data
    df_segment = pd.read_excel(uploaded_file, sheet_name="Segment Attribution")
    df_weights = pd.read_excel(uploaded_file, sheet_name="Causal Weights")
    df_segment["Date"] = pd.to_datetime(df_segment["Date"])

    # Build selectors
    channels = sorted(df_segment["Channel"].unique().tolist())
    segments = sorted(df_segment["Segment"].unique().tolist())
    products = sorted(df_segment["ProductCategory"].unique().tolist())
    customer_types = sorted(df_segment["CustomerType"].unique().tolist())

    all_channels = ["All"] + channels
    all_segments = ["All"] + segments
    all_products = ["All"] + products
    all_customers = ["All"] + customer_types
    scenario_names = ["Scenario 1", "Scenario 2", "Scenario 3"]

    # Session state for scenario planner
    if "scenario_changes" not in st.session_state:
        st.session_state["scenario_changes"] = {name: [] for name in scenario_names}

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

	# Filter interface
st.markdown("### 🎛️ Filter the Dashboard")

fc1, fc2 = st.columns(2)

with fc1:
		selected_channels = st.multiselect("Channels", channels, default=channels)
		selected_products = st.multiselect("Products", products, default=products)

with fc2:
		selected_segments = st.multiselect("Segments", segments, default=segments)
		selected_customers = st.multiselect("Customer Type", customer_types, default=customer_types)

df_filtered = df_segment[
		df_segment["Channel"].isin(selected_channels) &
		df_segment["Segment"].isin(selected_segments) &
		df_segment["ProductCategory"].isin(selected_products) &
		df_segment["CustomerType"].isin(selected_customers)
	]

    # Chart 1: Cumulative Revenue Over Time
st.markdown("### 📈 Revenue by Channel (Cumulative)")
st.markdown("""
    This chart shows total revenue accumulating week by week.  
    **Use case**: Understand revenue pacing.  
    **Interpretation**: A steady upward slope is expected. Flat lines may show underperformance.  
    **Action**: Drill into slow weeks and align with campaign cadence.
    """)
cum = df_filtered.groupby("Date")["AttributedSales"].sum().cumsum().reset_index()
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(cum["Date"], cum["AttributedSales"] / 1e6)
ax1.set_title("Cumulative Revenue")
ax1.set_ylabel("£ Revenue (millions)")
st.pyplot(fig1)

    # Chart 2: Revenue by Week (Not Cumulative)
st.markdown("### 📉 Revenue by Week")
st.markdown("""
    This view breaks down revenue each week individually.  
    **Use case**: Spot spikes or drops in weekly performance.  
    **Interpretation**: Peaks often align with campaign activity or sponsorship.  
    **Action**: Investigate performance drivers by week.
    """)
weekly = df_filtered.groupby("Date")["AttributedSales"].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(weekly["Date"], weekly["AttributedSales"] / 1e6)
ax2.set_title("Weekly Revenue")
ax2.set_ylabel("£ Revenue (millions)")
st.pyplot(fig2)

    # Chart 3: Stacked Area – Cumulative Revenue by Channel
st.markdown("### 📊 Cumulative Revenue by Channel (Stacked Area)")
st.markdown("""
    This chart shows how each channel contributes to revenue over time.  
    **Use case**: Evaluate mix and shifts in channel contribution.  
    **Interpretation**: Thicker areas = higher contribution.  
    **Action**: Adjust investment in underperforming or over-performing channels.
    """)
area = df_filtered.groupby(["Date", "Channel"])["AttributedSales"].sum().unstack().fillna(0).cumsum()
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.stackplot(area.index, area.T / 1e6, labels=area.columns)
ax3.set_ylabel("£ Revenue (millions)")
ax3.set_title("Cumulative Revenue by Channel")
ax3.legend(loc="upper left")
st.pyplot(fig3)
    
    # Chart 4: Total Revenue by Channel
st.markdown("### 💰 Total Revenue by Channel")
st.markdown("""
    Total revenue generated per channel across the full period.  
    **Use case**: Compare overall performance across media.  
    **Interpretation**: High bars = stronger contributors.  
    **Action**: Rebalance spend toward top channels.
    """)
total = df_filtered.groupby("Channel")["AttributedSales"].sum().reindex(channels, fill_value=0).reset_index()
fig4, ax4 = plt.subplots(figsize=(12, 5))
sns.barplot(data=total, x="Channel", y="AttributedSales", ax=ax4)
ax4.set_ylabel("£ Revenue (millions)")
ax4.set_title("Total Revenue by Channel")
ax4.set_yticklabels([f"{int(y/1e6)}m" for y in ax4.get_yticks()])
ax4.tick_params(axis="x", labelrotation=30, labelsize=9)
ax4.set_xticklabels(ax4.get_xticklabels(), ha="right")
st.pyplot(fig4)

    # Chart 5: Waterfall Chart – Revenue Drivers
st.markdown("### 📉 Revenue Drivers – Waterfall")
st.markdown("""
    This chart shows how each channel and macro factor contributes to revenue.  
    **Use case**: Attribute total revenue to key drivers.  
    **Interpretation**: Negative bars = drag; positive = uplift.  
    **Action**: Adjust strategy to improve low-impact drivers.
    """)
base = 1_000_000
channel_vals = df_filtered.groupby("Channel")["AttributedSales"].sum().reindex(channels, fill_value=0)
macro = {"Interest Rate": -100_000, "Competition": -150_000, "Segment Shift": 120_000}
waterfall = [("Baseline", base)] + list(channel_vals.items()) + list(macro.items())
waterfall.append(("Total", base + sum(v for k, v in waterfall[1:])))
df_wf = pd.DataFrame(waterfall, columns=["Driver", "Value"])
fig5, ax5 = plt.subplots(figsize=(14, 5))
sns.barplot(data=df_wf, x="Driver", y="Value", palette="coolwarm", ax=ax5)
ax5.set_ylabel("£ Value (millions)")
ax5.set_title("Revenue Waterfall by Driver")
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha="right")
    for p in ax5.patches:
        ax5.annotate(f"{p.get_height()/1e6:.1f}m", (p.get_x() + p.get_width()/2., p.get_height()), ha="center")
st.pyplot(fig5)

    # Chart 6: Competitor Impact Summary
    st.markdown("### 📋 Competitor Impact Summary")
    st.markdown("""
    Estimate of total revenue loss or gain attributed to competitors.  
    **Use case**: Identify where Barclays is losing vs. market.  
    **Interpretation**: Negative = revenue loss to competitor; positive = gain.  
    **Action**: Investigate where competitors are pulling ahead.
    """)
    competitors = ["HSBC", "Lloyds", "NatWest", "Santander", "Monzo", "Revolut"]
    competitor_impact = [-130_000, -110_000, -85_000, -60_000, -25_000, 10_000]
    df_comp = pd.DataFrame({"Competitor": competitors, "Impact (£)": competitor_impact})
    fig8, ax8 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=df_comp, x="Impact (£)", y="Competitor", palette="RdBu", ax=ax8)
    ax8.set_title("Revenue Impact by Competitor")
    st.pyplot(fig8)

    # Chart 7: Competitor Breakdown
    st.markdown("### 🔍 Competitor Impact Breakdown")
    st.markdown("""
    Breakdown of selected competitor's influence on revenue.  
    **Use case**: Diagnose strengths/strategies of one competitor.  
    **Interpretation**: Media Spend or Promotions may be leading factors.  
    **Action**: Develop counter-moves or defend share accordingly.
    """)
    selected_comp = st.selectbox("Select Competitor", competitors)
    if selected_comp == "Revolut":
        breakdown = [("Media Spend", 5_000), ("Promotions", 3_000), ("Brand Consideration", 2_000), ("Pricing", 1_000)]
    else:
        breakdown = [("Media Spend", -50_000), ("Promotions", -30_000), ("Brand Consideration", -20_000), ("Pricing", -10_000)]
    df_break = pd.DataFrame(breakdown, columns=["Driver", "Impact (£)"])
    fig9, ax9 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=df_break, x="Impact (£)", y="Driver", palette="crest", ax=ax9)
    ax9.set_title(f"{selected_comp} – Impact Drivers")
    st.pyplot(fig9)
  
    # Chart 8: Scenario Planner
    st.markdown("### 🔧 Scenario Planner")
    st.markdown("""
    Use this section to simulate changes to budget allocation.  
    **Use case**: Create what-if scenarios for spend shifts.  
    **Interpretation**: Each row defines a multiplier for a segment/channel/product/customer combo.  
    **Action**: Add multiple rows to test compounding effects. Use ❌ to remove individual rows or clear all.
    """)

    for scenario in scenario_names:
        with st.expander(f"{scenario} Adjustments"):
            c1, c2, c3, c4, c5, c6 = st.columns(6)

            with c1:
                seg = st.selectbox(f"Segment ({scenario})", all_segments, key=f"{scenario}_seg")
            with c2:
                chan = st.selectbox(f"Channel ({scenario})", all_channels, key=f"{scenario}_chan")
            with c3:
                prod = st.selectbox(f"Product ({scenario})", all_products, key=f"{scenario}_prod")
            with c4:
                cust = st.selectbox(f"Customer ({scenario})", all_customers, key=f"{scenario}_cust")
            with c5:
                mult = st.slider("Multiplier", 0.0, 2.0, 1.0, 0.1, key=f"{scenario}_mult")
            with c6:
                if st.button("Add", key=f"{scenario}_add"):
                    st.session_state["scenario_changes"][scenario].append((seg, chan, prod, cust, mult))
                    st.rerun()

            # Display table of current adjustments
            if st.session_state["scenario_changes"][scenario]:
                st.markdown("#### Current Adjustments")
                df_adj = pd.DataFrame(
                    st.session_state["scenario_changes"][scenario],
                    columns=["Segment", "Channel", "Product", "Customer", "Multiplier"]
                )

                for i, row in df_adj.iterrows():
                    cols = st.columns([3, 3, 3, 3, 1, 1])
                    cols[0].markdown(f"**{row['Segment']}**")
                    cols[1].markdown(f"{row['Channel']}")
                    cols[2].markdown(f"{row['Product']}")
                    cols[3].markdown(f"{row['Customer']}")
                    cols[4].markdown(f"x{row['Multiplier']:.1f}")

                    # Generate unique delete key
                    delete_key = f"{scenario}_del_{i}_{uuid.uuid4()}"
                    if cols[5].button("❌", key=delete_key):
                        st.session_state["scenario_changes"][scenario].pop(i)
                        st.rerun()

                # Clear All button
                if st.button(f"🗑 Clear All ({scenario})", key=f"{scenario}_clear_all"):
                    st.session_state["scenario_changes"][scenario] = []
                    st.rerun()
   
   # Chart 9: Forecasted Revenue by Scenario
    st.markdown("### 📈 Forecasted Revenue by Scenario")
    st.markdown("""
    This chart compares revenue projections under each scenario.  
    **Use case**: Evaluate which combination delivers best return.  
    **Interpretation**: Taller bars = more effective strategy.  
    **Action**: Choose a scenario for investment or test further.
    """)
    forecast_weeks = st.slider("📆 Forecast Horizon (weeks)", 4, 52, 12, step=1)

    scenario_dfs = {
        name: simulate(df_segment.copy(), st.session_state["scenario_changes"][name])
        for name in scenario_names
    }

    def get_forecast(df, weeks):
        df = df.sort_values("Date")
        if "SimulatedAttributedSales" not in df.columns:
            df = df.merge(df_weights, on="Channel", how="left")
            df["SimulatedAttributedSales"] = df["Spend"] * df["CausalWeight"]
        return df.groupby("Date")["SimulatedAttributedSales"].sum().head(weeks).sum() / 1e6

    scenario_results = pd.DataFrame({
        "Scenario": scenario_names,
        "Revenue (£m)": [get_forecast(df, forecast_weeks) for df in scenario_dfs.values()]
    })
    baseline = get_forecast(df_segment, forecast_weeks)
    scenario_results.loc[len(scenario_results)] = ["Baseline", baseline]

    fig7, ax7 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=scenario_results, x="Scenario", y="Revenue (£m)", ax=ax7)
    ax7.set_title("Forecast vs Baseline")
    for p in ax7.patches:
        ax7.annotate(f"{p.get_height():.1f}m", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center')
    st.pyplot(fig7)

	# Chart 10: Causal Graph – Marketing Influence Model
	st.markdown("### 🧠 Causal Graph – Marketing Influence Model")
	st.markdown("""
	This network diagram visualizes causal relationships between marketing inputs and revenue.  
	**Use case**: Explain which variables are driving impact and how they're connected.  
	**Interpretation**: Arrows show direction; weights show strength.  
	**Action**: Focus on the most influential drivers to optimize performance.
	""")

	# Define graph
	G = nx.DiGraph()
	edges = [
		("Promo", "Spend", 0.8),
		("Interest Rate", "Demand", -0.5),
		("Demand", "Spend", 0.6),
		("Spend", "Revenue", 1.2),
		("Customer Segment", "Revenue", 0.9),
		("Brand Equity", "Revenue", 0.7),
		("Competitor Spend", "Revenue", -0.4),
		("Search Trends", "Brand Equity", 0.5),
		("Search Trends", "Demand", 0.6)
	]
	G.add_weighted_edges_from(edges)

	# Improve layout and visibility
	pos = nx.spring_layout(G, seed=42, k=2.8)

	# Draw nodes with larger labels
	fig10, ax10 = plt.subplots(figsize=(10, 7))
	nx.draw_networkx_nodes(G, pos, ax=ax10, node_color='skyblue', node_size=5000)
	nx.draw_networkx_edges(G, pos, ax=ax10, arrowstyle='-|>', arrowsize=18, edge_color='black', connectionstyle='arc3,rad=0.1')
	nx.draw_networkx_labels(G, pos, ax=ax10, font_size=10, font_weight='bold', verticalalignment='center')

	# Draw weights clearly
	edge_labels = {(u, v): f"{d:.1f}" for u, v, d in G.edges(data='weight')}
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

	ax10.set_title("Causal Influence Network", fontsize=14)
	ax10.axis('off')
	st.pyplot(fig10)
