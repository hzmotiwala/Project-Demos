import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Function to load data (modify the path to where your dataset is located)
@st.cache_data
def load_data():
    data = pd.read_csv('snowflake_presentation/snowflake_energy.csv', parse_dates=['Date'])
    return data
df = load_data()

# uploaded_file = st.file_uploader("snowflake_energy.csv")
# dataframe = pd.read_csv(uploaded_file)
# df = pd.read_csv(uploaded_file, parse_dates=['Date'])

# Main page
st.title('Energy Usage Dashboard')

# Date range selector
start_date, end_date = st.select_slider(
    'Select date range:',
    options=df['Date'].dt.date.unique(),
    value=(pd.to_datetime('2023-07-01').date(), df['Date'].dt.date.max())
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

# Delta Comparison Chart
fig = px.line(filtered_df, x='Date', y=['Energy_delta_50d_3d', 'Energy_delta_50d_10d'],
              title='Energy Delta Comparisons')
st.plotly_chart(fig)

# DAY OF WEEK 7 Day Chart
latest_7_days = filtered_df.sort_values(by='Date', ascending=False).head(7)

# Determine the order of the weekdays based on the last day in the latest 7 days
last_day = latest_7_days['Date'].max().weekday()
weekday_order = [(last_day - i) % 7 for i in range(7)]
weekday_order = [pd.Timestamp(2021, 1, 4).weekday() + day for day in weekday_order]  # Monday is 0
weekday_order = [pd.Timestamp(2021, 1, 4 + day).day_name() for day in weekday_order]

# Sort the latest_7_days DataFrame by this new order
latest_7_days['Weekday'] = pd.Categorical(latest_7_days['Weekday'], categories=weekday_order, ordered=True)
latest_7_days = latest_7_days.sort_values('Weekday')

# Now, we need to aggregate this data by 'Weekday' for the bar chart
grouped = latest_7_days.groupby('Weekday', as_index=False).mean()

# Plotting with Plotly
fig = px.bar(grouped, x='Weekday', y=['Energy_3d', '8w_DayOfWeek_Avg'],
             barmode='group', title='Energy Usage Last 7 Days DOW')
# Display the plot
st.plotly_chart(fig)

# Temperature and Precipitation Visualizations
fig_temp = px.line(filtered_df, x='Date', y=['Temp', 'Temp_7d_avg'], title='Temperature Trends')
st.plotly_chart(fig_temp)

fig_precip = px.line(filtered_df, x='Date', y=['Precip', 'Precip_7d_avg'], title='Precipitation Trends')
st.plotly_chart(fig_precip)

anomalies['Date'] = anomalies['Date'].dt.date
anomalies['Energy_delta_50d_3d'] = (df['Energy_delta_50d_3d'] * 100).round(1).astype(str) + '%'
anomalies['Energy_delta_50d_10d'] = (df['Energy_delta_50d_10d'] * 100).round(1).astype(str) + '%'
anomalies = anomalies.sort_values(by='Date', ascending=False)
anomalies = anomalies.set_index('Date')


# Anomaly Detection Section
st.write("Anomaly Detection")
st.dataframe(anomalies[['Anomaly_Flag','Energy_3d','Energy_10d','Energy_50d','Energy_delta_50d_3d','Energy_delta_50d_10d','Temp','Temp_7d_avg']])

