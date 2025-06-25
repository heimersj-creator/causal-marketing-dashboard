# causal_dashboard_app.py
# Streamlit app with optimized layout for Barclays UK consumer bank

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    selected_channel = st.sidebar.selectbox("Channel", channels)
    selected_product = st.sidebar.selectbox("Product", products)
    selected_segment = st.sidebar.selectbox("Segment", segments)
    spend_multiplier = st.sidebar.slider("Spend multiplier", 0.0, 2.0, 1.0, 0.1)
    if st.sidebar.button("Apply Scenario Change"):
        scenario_changes[selected_scenario][(selected_segment, selected_channel)] = spend_multiplier
        st.sidebar.success(f"Applied: {selected_segment} × {selected_channel} → Spend x {spend_multiplier}")

    target_sales = st.sidebar.number_input("🎯 Target Sales", value=250000.0, step=1000.0)
    budget_cap = st.sidebar.number_input("💰 Budget Cap", value=100000.0, step=1000.0)

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    scenario_dfs = {name: simulate(df_segment, scenario_changes.get(name, {})) for name in scenario_names}

    st.subheader("📊 Weekly Revenue Performance")
    selected_df = scenario_dfs['Scenario 1']
    weekly_perf = selected_df[selected_df['Segment'].isin(selected_segments) &
                              selected_df['Channel'].isin(selected_channels)]
    # Ensure all channels are represented
    revenue_by_channel = weekly_perf.groupby('Channel')['SimulatedAttributedSales'].sum().reindex(channels, fill_value=0).reset_index()
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    sns.barplot(x='Channel', y='SimulatedAttributedSales', data=revenue_by_channel, ax=ax1)
    ax1.set_title("Forecasted Revenue by Channel")
    ax1.set_ylabel("£ Revenue")
    ax1.set_xlabel("")
    st.pyplot(fig1)

    st.subheader("📉 Drivers of Performance")
    base_sales = df_segment['AttributedSales'].sum()
    scenario_sales = selected_df['SimulatedAttributedSales'].sum()
    delta = scenario_sales - base_sales
    driver_data = pd.DataFrame({
        'Driver': ['Base', 'Campaign Uplift', 'Interest Rate Change', 'Competitor Promotions', 'Barclays Media Shift', 'Scenario Forecast'],
        'Value': [base_sales, 8000, -5000, -7000, delta, scenario_sales]
    })
    driver_data['Cumulative'] = driver_data['Value'].cumsum()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Driver', y='Value', data=driver_data, palette='coolwarm', ax=ax2)
    ax2.set_title("Drivers of Performance - Barclays UK Context", fontsize=14)
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_xlabel("Driver", fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():,.0f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9)
    st.pyplot(fig2)

    st.subheader("📋 Competitor Impact Breakdown")
    competitors = ['HSBC', 'Lloyds', 'NatWest', 'Santander', 'Monzo', 'Revolut']
    reasons = ['Media Spend', 'Promo Push', 'New Product', 'Rate Cut', 'Brand Buzz', 'Referral Bonus']
    competitor_effect = pd.DataFrame({
        'Competitor': competitors,
        'Impact (£)': [-130000, -110000, -85000, -60000, -25000, 10000],
        'Driver': reasons
    })
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    sns.barplot(x='Impact (£)', y='Competitor', hue='Driver', data=competitor_effect, ax=ax3)
    ax3.set_title("Estimated Revenue Impact by UK Competitors & Cause")
    st.pyplot(fig3)

    st.subheader("🔮 Scenario Forecast Comparison")
    comparison = pd.DataFrame({
        'Scenario': scenario_names,
        'Revenue (£)': [scenario_dfs[name]['SimulatedAttributedSales'].sum() for name in scenario_names]
    })
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Scenario', y='Revenue (£)', data=comparison, palette='Blues', ax=ax4)
    ax4.set_title("Forecasted Revenue by Scenario")
    ax4.set_ylabel("£ Revenue")
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():,.0f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9)
    st.pyplot(fig4)
