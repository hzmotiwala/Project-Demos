import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Function to load data (modify the path to where your dataset is located)

uploaded_file = st.file_uploader("snowflake_energy.csv")

dataframe = pd.read_csv(uploaded_file)
@st.cache_data
df = pd.read_csv(uploaded_file, parse_dates=['Date'])

# @st.cache_data
# def load_data():
#     data = pd.read_csv('snowflake_energy.csv', parse_dates=['Date'])
#     return data
# df = load_data()

# Sidebar for statistical summary
# st.sidebar.header('Statistical Summary')
# selected_metrics_summary = st.sidebar.multiselect('Select metrics for summary:', ['Energy_3d', 'Energy_10d', 'Energy_50d', 'Temp', 'Precip'], default=['Energy_3d', 'Energy_50d'])
# st.sidebar.write(df[selected_metrics_summary].describe())

# Main page
st.title('Energy Usage Dashboard')

# Date range selector
start_date, end_date = st.select_slider(
    'Select date range:',
    options=df['Date'].dt.date.unique(),
    value=(df['Date'].dt.date.min(), df['Date'].dt.date.max())
)
filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

# Metrics selection for visualization
selected_metrics = st.multiselect('Select metrics to display:', ['Energy_3d', 'Energy_10d', 'Energy_50d', 'Temp', 'Precip', 'Temp_7d_avg', 'Precip_7d_avg'], default=['Energy_3d', 'Energy_50d'])

# Plotting the selected metrics with anomaly highlighting
fig = px.line(filtered_df, x='Date', y=selected_metrics, title='Selected Metrics Over Time')
anomalies = filtered_df[filtered_df['Anomaly_Flag'] == 1]
for metric in selected_metrics:
    if 'Anomaly_Flag' in filtered_df.columns:
        fig.add_scatter(x=anomalies['Date'], y=anomalies[metric], mode='markers', marker_symbol='x', marker_color='red', name=f'Anomalies in {metric}')
st.plotly_chart(fig)

# 30-Day Moving Average of Energy Usage
fig_30d_ma = px.line(filtered_df, x='Date', y='30d_MA', title='30-Day Moving Average of Energy Usage')
st.plotly_chart(fig_30d_ma)

# Temperature and Precipitation Visualizations
fig_temp = px.line(filtered_df, x='Date', y=['Temp', 'Temp_7d_avg'], title='Temperature Trends')
st.plotly_chart(fig_temp)

fig_precip = px.line(filtered_df, x='Date', y=['Precip', 'Precip_7d_avg'], title='Precipitation Trends')
st.plotly_chart(fig_precip)

# Anomaly Detection Section
st.write("Anomaly Detection")
st.dataframe(anomalies[['Date', 'Energy_3d', 'Energy_10d', 'Energy_50d', 'Anomaly_Flag']])
