
# causal_dashboard_app.py
# Streamlit app version of the causal AI marketing simulator

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
sns.set(style='whitegrid')

st.title("📊 Causal Marketing Decision Support Dashboard")

# Upload Excel workbook
uploaded_file = st.file_uploader("Upload the causal simulator Excel file", type="xlsx")
if uploaded_file:
    df_segment = pd.read_excel(uploaded_file, sheet_name='Segment Attribution')
    df_product = pd.read_excel(uploaded_file, sheet_name='Product Attribution')
    df_weights = pd.read_excel(uploaded_file, sheet_name='Causal Weights')

    segments = df_segment['Segment'].unique().tolist()
    channels = df_segment['Channel'].unique().tolist()

    scenario_names = ['Base', 'Scenario 1', 'Scenario 2', 'Scenario 3']
    scenario_changes = {name: {} for name in scenario_names}

    st.sidebar.header("🎛 Scenario Controls")
    selected_scenario = st.sidebar.selectbox("Choose a scenario", scenario_names)
    selected_segment = st.sidebar.selectbox("Select a segment", segments)
    selected_channel = st.sidebar.selectbox("Select a channel", channels)
    spend_multiplier = st.sidebar.slider("Spend multiplier", 0.0, 2.0, 1.0, 0.1)

    if st.sidebar.button("Apply to scenario"):
        scenario_changes[selected_scenario][(selected_segment, selected_channel)] = spend_multiplier
        st.sidebar.success(f"Applied: {selected_segment} × {selected_channel} → Spend x {spend_multiplier}")

    # Target and budget inputs
    target_sales = st.sidebar.number_input("🎯 Target Sales", value=100000.0, step=1000.0)
    budget_cap = st.sidebar.number_input("💰 Budget Cap", value=50000.0, step=1000.0)

    def simulate(df, changes):
        df = df.copy()
        for (seg, chan), mult in changes.items():
            df.loc[(df['Segment'] == seg) & (df['Channel'] == chan), 'Spend'] *= mult
        df['SimulatedAttributedSales'] = df['Spend'] * df['CausalWeight']
        df['ROI'] = df['SimulatedAttributedSales'] / df['Spend']
        return df

    # Simulate all scenarios
    scenario_dfs = {name: simulate(df_segment, scenario_changes.get(name, {})) for name in scenario_names}

    # Show scenario comparison
    st.subheader("📈 Scenario Performance Comparison")
    results = pd.DataFrame({
        "Scenario": scenario_names,
        "Attributed Sales": [df['SimulatedAttributedSales'].sum() for df in scenario_dfs.values()],
        "Total Spend": [df['Spend'].sum() for df in scenario_dfs.values()]
    })
    results["Sales Gap"] = target_sales - results["Attributed Sales"]
    results["Budget Remaining"] = budget_cap - results["Total Spend"]
    st.dataframe(results)

    fig, ax = plt.subplots()
    sns.barplot(data=results, x="Scenario", y="Attributed Sales", ax=ax)
    ax.axhline(target_sales, color='red', linestyle='--', label='Target')
    ax.set_title("Total Attributed Sales by Scenario")
    ax.legend()
    st.pyplot(fig)

    # ROI Leaderboard
    st.subheader("📊 ROI Leaderboard (Scenario 1)")
    df1 = scenario_dfs['Scenario 1'].copy()
    df1['ROI'] = df1['SimulatedAttributedSales'] / df1['Spend']
    top_roi = df1.sort_values(by='ROI', ascending=False).head(10)
    st.dataframe(top_roi[['Segment', 'Channel', 'Spend', 'SimulatedAttributedSales', 'ROI']])

    # Marginal Efficiency (vs Base)
    st.subheader("📉 Marginal ROI: Scenario 1 vs Base")
    base = scenario_dfs['Base']
    df1['DeltaSpend'] = df1['Spend'] - base['Spend']
    df1['DeltaSales'] = df1['SimulatedAttributedSales'] - base['AttributedSales']
    df1['MarginalROI'] = df1['DeltaSales'] / df1['DeltaSpend']
    marginal = df1.sort_values(by='MarginalROI', ascending=False).head(10)
    st.dataframe(marginal[['Segment', 'Channel', 'DeltaSpend', 'DeltaSales', 'MarginalROI']])

    # Budget Optimisation
    st.subheader("💡 Budget-Constrained Optimisation")
    df_opt = df1.sort_values(by='ROI', ascending=False).copy()
    spent = 0
    allocation = []

    for _, row in df_opt.iterrows():
        if spent + row['Spend'] <= budget_cap:
            allocation.append(row)
            spent += row['Spend']

    alloc_df = pd.DataFrame(allocation)
    st.dataframe(alloc_df[['Segment', 'Channel', 'Spend', 'SimulatedAttributedSales', 'ROI']])
    st.success(f"Total Spend: {spent:.2f} of {budget_cap:.2f}")
