import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import calendar
import numpy as np
import io

# Set page config
st.set_page_config(layout="wide", page_title="Logistics Analytics Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["IF Order Trends", "Transportation Trips Trends"])

with tab1:
    st.header("IF Order Trends Analysis")

    # File uploader for IF Order data
    uploaded_file = st.file_uploader("Upload IF Orders CSV file", type=['csv'], key="if_orders_uploader")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Convert date column to datetime
            df['Created Date'] = pd.to_datetime(df['Created Date'], format='%d-%m-%Y')

            # Get current date and month/year
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year

            # Get last month
            if current_month == 1:
                last_month = 12
                last_year = current_year - 1
            else:
                last_month = current_month - 1
                last_year = current_year

            # Filter current and last month data
            current_month_data = df[(df['Created Date'].dt.month == current_month) &
                                    (df['Created Date'].dt.year == current_year)].copy()

            last_month_data = df[(df['Created Date'].dt.month == last_month) &
                                 (df['Created Date'].dt.year == last_year)].copy()

            # Month-over-month comparison
            st.subheader("Month-over-Month Comparison")
            col1, col2, col3 = st.columns(3)

            current_month_total = current_month_data['Order Number'].nunique()
            last_month_total = last_month_data['Order Number'].nunique()

            col1.metric("Current Month Total Orders", current_month_total)
            col2.metric("Last Month Total Orders", last_month_total)

            change_percent = ((current_month_total - last_month_total) / last_month_total * 100
                              if last_month_total > 0 else 0)
            col3.metric("Change (%)", f"{change_percent:.2f}%",
                        delta_color="inverse" if change_percent < 0 else "normal")

            # Week-wise comparison
            st.subheader("Week-wise Comparison")

            def get_week(date):
                day = date.day
                if day <= 7:
                    return "Week 1 (1-7)"
                elif day <= 14:
                    return "Week 2 (8-14)"
                elif day <= 21:
                    return "Week 3 (15-21)"
                else:
                    return "Week 4 (22-31)"

            current_month_data.loc[:, 'Week'] = current_month_data['Created Date'].apply(get_week)
            last_month_data.loc[:, 'Week'] = last_month_data['Created Date'].apply(get_week)

            current_week_counts = current_month_data.groupby('Week')['Order Number'].nunique().reset_index()
            current_week_counts['Month'] = 'Current Month'

            last_week_counts = last_month_data.groupby('Week')['Order Number'].nunique().reset_index()
            last_week_counts['Month'] = 'Last Month'

            week_comparison = pd.concat([current_week_counts, last_week_counts])

            fig = px.bar(week_comparison, x='Week', y='Order Number', color='Month',
                         barmode='group', title='Week-wise Order Comparison',
                         labels={'Order Number': 'Number of Orders'})
            st.plotly_chart(fig, use_container_width=True)

            # Month-to-date comparison
            st.subheader("Month-to-Date Comparison")

            min_date_current = current_month_data['Created Date'].min().date()
            max_date_current = current_month_data['Created Date'].max().date()
            default_end_date = min(current_date.date(), max_date_current)

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date",
                                           value=datetime(current_year, current_month, 1).date(),
                                           min_value=datetime(current_year, current_month, 1).date(),
                                           max_value=default_end_date,
                                           key="if_start_date")  # Unique key
            with col2:
                end_date = st.date_input("End date",
                                         value=default_end_date,
                                         min_value=datetime(current_year, current_month, 1).date(),
                                         max_value=default_end_date,
                                         key="if_end_date")  # Unique key

            if start_date > end_date:
                st.error("Error: End date must be after start date.")
            else:
                current_mtd = current_month_data[
                    (current_month_data['Created Date'].dt.date >= start_date) &
                    (current_month_data['Created Date'].dt.date <= end_date)
                ].copy()

                last_mtd = last_month_data[
                    (last_month_data['Created Date'].dt.day >= 1) &
                    (last_month_data['Created Date'].dt.day <= end_date.day)
                ].copy()
                last_mtd.loc[:, 'Day'] = last_mtd['Created Date'].dt.day

                col1, col2, col3 = st.columns(3)

                current_mtd_total = current_mtd['Order Number'].nunique()
                col1.metric(f"Current Month ({start_date.strftime('%d-%b')} to {end_date.strftime('%d-%b')}) Orders",
                            current_mtd_total)

                last_mtd_total = last_mtd['Order Number'].nunique()
                last_month_name = calendar.month_name[last_month]
                col2.metric(f"Last Month ({last_month_name} 1-{end_date.day}) Orders", last_mtd_total)

                mtd_change_percent = ((current_mtd_total - last_mtd_total) / last_mtd_total * 100
                                      if last_mtd_total > 0 else 0)
                col3.metric("Change (%)", f"{mtd_change_percent:.2f}%",
                            delta_color="inverse" if mtd_change_percent < 0 else "normal")

                # Customer-wise comparison table
                st.subheader("Customer-wise MTD Comparison")
                
                # Group by customer for current and last month
                current_customer = current_mtd.groupby('Customer Name')['Order Number'].nunique().reset_index()
                current_customer.columns = ['Customer Name', 'Current Month Orders']
                
                last_customer = last_mtd.groupby('Customer Name')['Order Number'].nunique().reset_index()
                last_customer.columns = ['Customer Name', 'Last Month Orders']
                
                # Merge the data
                customer_comparison = pd.merge(current_customer, last_customer, on='Customer Name', how='outer').fillna(0)
                
                # Calculate differences and growth percentages
                customer_comparison['Difference'] = customer_comparison['Current Month Orders'] - customer_comparison['Last Month Orders']
                customer_comparison['Growth %'] = np.where(
                    customer_comparison['Last Month Orders'] > 0,
                    (customer_comparison['Difference'] / customer_comparison['Last Month Orders']) * 100,
                    0
                )
                
                # Add grand total row
                grand_total = pd.DataFrame({
                    'Customer Name': ['TOTAL'],
                    'Current Month Orders': [customer_comparison['Current Month Orders'].sum()],
                    'Last Month Orders': [customer_comparison['Last Month Orders'].sum()],
                    'Difference': [customer_comparison['Difference'].sum()],
                    'Growth %': [
                        (customer_comparison['Difference'].sum() / customer_comparison['Last Month Orders'].sum() * 100
                        if customer_comparison['Last Month Orders'].sum() > 0 else 0)
                    ]
                })
                
                customer_comparison = pd.concat([customer_comparison, grand_total], ignore_index=True)
                
                # Format the table
                def color_negative_red(val):
                    color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                    return f'color: {color}'
                
                styled_comparison = customer_comparison.style \
                    .applymap(color_negative_red, subset=['Difference', 'Growth %']) \
                    .format({
                        'Current Month Orders': '{:.0f}',
                        'Last Month Orders': '{:.0f}',
                        'Difference': '{:.0f}',
                        'Growth %': '{:.1f}%'
                    }) \
                    .background_gradient(cmap='Blues', subset=['Current Month Orders', 'Last Month Orders'])
                
                st.dataframe(styled_comparison, use_container_width=True)

                # Daily trend chart with customer breakdown
                st.subheader(f"Daily Order Trend ({start_date.strftime('%d-%b')} to {end_date.strftime('%d-%b')})")
                
                # Create tabs for different views
                daily_tab1, daily_tab2 = st.tabs(["Overall Daily Trend", "Customer-wise Daily Trend"])

                with daily_tab1:
                    # Original daily trend (overall)
                    daily_current = current_mtd.groupby(current_mtd['Created Date'].dt.date)['Order Number']\
                                           .nunique().reset_index()
                    daily_current['Month'] = 'Current Month'

                    daily_last = last_mtd.groupby('Day')['Order Number'].nunique().reset_index()
                    daily_last['Month'] = 'Last Month'
                    daily_last['Created Date'] = [f"Day {d}" for d in daily_last['Day']]

                    daily_comparison = pd.concat([
                        daily_current.rename(columns={'Created Date': 'Date'}),
                        daily_last.rename(columns={'Created Date': 'Date'})
                    ])

                    fig_daily = px.line(daily_comparison, x='Date', y='Order Number', color='Month',
                                   markers=True,
                                   title=f'Daily Order Comparison: {start_date.strftime("%d-%b")} to {end_date.strftime("%d-%b")}',
                                   labels={'Order Number': 'Number of Orders', 'Date': 'Date/Day'})
                    st.plotly_chart(fig_daily, use_container_width=True)

                with daily_tab2:
                    # Customer-wise daily trend
                    daily_customer = current_mtd.groupby([current_mtd['Created Date'].dt.date, 'Customer Name'])['Order Number']\
                                            .nunique().unstack().fillna(0)
                    
                    # Sort customers by total orders (descending) and take top N for better visualization
                    top_customers = current_mtd['Customer Name'].value_counts().head(10).index.tolist()
                    daily_customer = daily_customer[top_customers]
                    
                    fig_customer_daily = px.bar(daily_customer, 
                                             x=daily_customer.index, 
                                             y=daily_customer.columns,
                                             title=f'Customer-wise Daily Orders: {start_date.strftime("%d-%b")} to {end_date.strftime("%d-%b")}',
                                             labels={'value': 'Number of Orders', 'Created Date': 'Date', 'variable': 'Customer'},
                                             barmode='stack')
                    
                    fig_customer_daily.update_layout(legend_title_text='Customers',
                                                  xaxis_title='Date',
                                                  yaxis_title='Number of Orders')
                    
                    st.plotly_chart(fig_customer_daily, use_container_width=True)
                    
                    # Show raw data option
                    if st.checkbox("Show customer-wise daily data", key="if_customer_daily_data"):
                        st.dataframe(daily_customer)

            # Customer-wise order trends
            st.subheader("Customer-wise Order Trends")

            customer_monthly = df.groupby(['Customer Name', df['Created Date'].dt.to_period('M')])['Order Number']\
                                 .nunique().unstack().fillna(0).astype(int)

            months = customer_monthly.columns.sort_values(ascending=False)
            current_month_col = months[0]
            last_month_col = months[1] if len(months) > 1 else None

            if last_month_col is not None:
                customer_monthly['Difference'] = customer_monthly[current_month_col] - customer_monthly[last_month_col]
                customer_monthly['%Growth'] = np.where(
                    customer_monthly[last_month_col] > 0,
                    (customer_monthly[current_month_col] - customer_monthly[last_month_col]) /
                    customer_monthly[last_month_col] * 100,
                    0
                )
                customer_monthly['%Growth'] = customer_monthly['%Growth'].round(1)

            grand_total = customer_monthly.sum().to_frame().T
            grand_total.index = ['Grand Total']

            # Calculate correct %Growth for Grand Total
            if last_month_col is not None:
                total_current = grand_total[current_month_col].values[0]
                total_last = grand_total[last_month_col].values[0]
                if total_last > 0:
                    grand_total['%Growth'] = ((total_current - total_last) / total_last * 100).round(1)
                else:
                    grand_total['%Growth'] = 0
                grand_total['Difference'] = total_current - total_last

            customer_monthly = pd.concat([customer_monthly, grand_total])

            customer_monthly = customer_monthly.sort_values(current_month_col, ascending=False)

            def style_negative_positive(val):
                color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                return f'color: {color}'

            styled_df = customer_monthly.style \
                .applymap(style_negative_positive, subset=['Difference', '%Growth']) \
                .format("{:.0f}", subset=[col for col in customer_monthly.columns if col not in ['%Growth']]) \
                .format("{:.1f}%", subset=['%Growth']) \
                .background_gradient(cmap='Blues',
                                     subset=[col for col in customer_monthly.columns if col not in ['Difference', '%Growth']])

            st.dataframe(styled_df, use_container_width=True)

            # Optional raw data
            if st.checkbox("Show raw data", key="if_raw_data"):
                st.subheader("Raw Data")
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error processing data: {e}")

with tab2:
    st.header("Transportation Trips Trends Analysis")

    # File uploader for Transportation Trips data
    uploaded_file_trips = st.file_uploader("Upload Transportation Trips CSV file", type=['csv'], key="trips_uploader")
    
    if uploaded_file_trips is not None:
        try:
            # Read the uploaded file
            df_trips = pd.read_csv(uploaded_file_trips)
            
            # Convert date column to datetime
            df_trips['Scheduled At'] = pd.to_datetime(df_trips['Scheduled At'], format='%d-%b-%Y')

            # Filter only completed trips
            completed_trips = df_trips[df_trips['Trip Status'] == 'Completed'].copy()
            
            # Get current date and month/year
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year

            # Get last month
            if current_month == 1:
                last_month = 12
                last_year = current_year - 1
            else:
                last_month = current_month - 1
                last_year = current_year

            # Filter current and last month data
            current_month_data = completed_trips[(completed_trips['Scheduled At'].dt.month == current_month) &
                                               (completed_trips['Scheduled At'].dt.year == current_year)].copy()

            last_month_data = completed_trips[(completed_trips['Scheduled At'].dt.month == last_month) &
                                            (completed_trips['Scheduled At'].dt.year == last_year)].copy()

            # Month-over-month comparison
            st.subheader("Month-over-Month Comparison")
            col1, col2, col3 = st.columns(3)

            current_month_total = current_month_data['Trip_Number'].nunique()
            last_month_total = last_month_data['Trip_Number'].nunique()

            col1.metric("Current Month Completed Trips", current_month_total)
            col2.metric("Last Month Completed Trips", last_month_total)

            change_percent = ((current_month_total - last_month_total) / last_month_total * 100
                            if last_month_total > 0 else 0)
            col3.metric("Change (%)", f"{change_percent:.2f}%",
                      delta_color="inverse" if change_percent < 0 else "normal")

            # Week-wise comparison
            st.subheader("Week-wise Comparison")

            def get_week(date):
                day = date.day
                if day <= 7:
                    return "Week 1 (1-7)"
                elif day <= 14:
                    return "Week 2 (8-14)"
                elif day <= 21:
                    return "Week 3 (15-21)"
                else:
                    return "Week 4 (22-31)"

            current_month_data.loc[:, 'Week'] = current_month_data['Scheduled At'].apply(get_week)
            last_month_data.loc[:, 'Week'] = last_month_data['Scheduled At'].apply(get_week)

            current_week_counts = current_month_data.groupby('Week')['Trip_Number'].nunique().reset_index()
            current_week_counts['Month'] = 'Current Month'

            last_week_counts = last_month_data.groupby('Week')['Trip_Number'].nunique().reset_index()
            last_week_counts['Month'] = 'Last Month'

            week_comparison = pd.concat([current_week_counts, last_week_counts])

            fig = px.bar(week_comparison, x='Week', y='Trip_Number', color='Month',
                       barmode='group', title='Week-wise Trip Comparison',
                       labels={'Trip_Number': 'Number of Trips'})
            st.plotly_chart(fig, use_container_width=True)

            # Month-to-date comparison
            st.subheader("Month-to-Date Comparison")

            min_date_current = current_month_data['Scheduled At'].min().date()
            max_date_current = current_month_data['Scheduled At'].max().date()
            default_end_date = min(current_date.date(), max_date_current)

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date",
                                         value=datetime(current_year, current_month, 1).date(),
                                         min_value=datetime(current_year, current_month, 1).date(),
                                         max_value=default_end_date,
                                         key="trips_start_date")  # Unique key
            with col2:
                end_date = st.date_input("End date",
                                       value=default_end_date,
                                       min_value=datetime(current_year, current_month, 1).date(),
                                       max_value=default_end_date,
                                       key="trips_end_date")  # Unique key

            if start_date > end_date:
                st.error("Error: End date must be after start date.")
            else:
                current_mtd = current_month_data[
                    (current_month_data['Scheduled At'].dt.date >= start_date) &
                    (current_month_data['Scheduled At'].dt.date <= end_date)
                ].copy()

                last_mtd = last_month_data[
                    (last_month_data['Scheduled At'].dt.day >= 1) &
                    (last_month_data['Scheduled At'].dt.day <= end_date.day)
                ].copy()
                last_mtd.loc[:, 'Day'] = last_mtd['Scheduled At'].dt.day

                col1, col2, col3 = st.columns(3)

                current_mtd_total = current_mtd['Trip_Number'].nunique()
                col1.metric(f"Current Month ({start_date.strftime('%d-%b')} to {end_date.strftime('%d-%b')}) Trips",
                          current_mtd_total)

                last_mtd_total = last_mtd['Trip_Number'].nunique()
                last_month_name = calendar.month_name[last_month]
                col2.metric(f"Last Month ({last_month_name} 1-{end_date.day}) Trips", last_mtd_total)

                mtd_change_percent = ((current_mtd_total - last_mtd_total) / last_mtd_total * 100
                                    if last_mtd_total > 0 else 0)
                col3.metric("Change (%)", f"{mtd_change_percent:.2f}%",
                          delta_color="inverse" if mtd_change_percent < 0 else "normal")

                # Customer-wise comparison table
                st.subheader("Customer-wise MTD Comparison")
                
                # Group by customer for current and last month
                current_customer = current_mtd.groupby('Customer')['Trip_Number'].nunique().reset_index()
                current_customer.columns = ['Customer', 'Current Month Trips']
                
                last_customer = last_mtd.groupby('Customer')['Trip_Number'].nunique().reset_index()
                last_customer.columns = ['Customer', 'Last Month Trips']
                
                # Merge the data
                customer_comparison = pd.merge(current_customer, last_customer, on='Customer', how='outer').fillna(0)
                
                # Calculate differences and growth percentages
                customer_comparison['Difference'] = customer_comparison['Current Month Trips'] - customer_comparison['Last Month Trips']
                customer_comparison['Growth %'] = np.where(
                    customer_comparison['Last Month Trips'] > 0,
                    (customer_comparison['Difference'] / customer_comparison['Last Month Trips']) * 100,
                    0
                )
                
                # Add grand total row
                grand_total = pd.DataFrame({
                    'Customer': ['TOTAL'],
                    'Current Month Trips': [customer_comparison['Current Month Trips'].sum()],
                    'Last Month Trips': [customer_comparison['Last Month Trips'].sum()],
                    'Difference': [customer_comparison['Difference'].sum()],
                    'Growth %': [
                        (customer_comparison['Difference'].sum() / customer_comparison['Last Month Trips'].sum() * 100
                        if customer_comparison['Last Month Trips'].sum() > 0 else 0)
                    ]
                })
                
                customer_comparison = pd.concat([customer_comparison, grand_total], ignore_index=True)
                
                # Format the table
                def color_negative_red(val):
                    color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                    return f'color: {color}'
                
                styled_comparison = customer_comparison.style \
                    .applymap(color_negative_red, subset=['Difference', 'Growth %']) \
                    .format({
                        'Current Month Trips': '{:.0f}',
                        'Last Month Trips': '{:.0f}',
                        'Difference': '{:.0f}',
                        'Growth %': '{:.1f}%'
                    }) \
                    .background_gradient(cmap='Blues', subset=['Current Month Trips', 'Last Month Trips'])
                
                st.dataframe(styled_comparison, use_container_width=True)

                # Daily trend chart with customer breakdown
                st.subheader(f"Daily Trip Trend ({start_date.strftime('%d-%b')} to {end_date.strftime('%d-%b')})")
                
                # Create tabs for different views
                daily_tab1, daily_tab2 = st.tabs(["Overall Daily Trend", "Customer-wise Daily Trend"])

                with daily_tab1:
                    # Original daily trend (overall)
                    daily_current = current_mtd.groupby(current_mtd['Scheduled At'].dt.date)['Trip_Number']\
                                           .nunique().reset_index()
                    daily_current['Month'] = 'Current Month'

                    daily_last = last_mtd.groupby('Day')['Trip_Number'].nunique().reset_index()
                    daily_last['Month'] = 'Last Month'
                    daily_last['Scheduled At'] = [f"Day {d}" for d in daily_last['Day']]

                    daily_comparison = pd.concat([
                        daily_current.rename(columns={'Scheduled At': 'Date'}),
                        daily_last.rename(columns={'Scheduled At': 'Date'})
                    ])

                    fig_daily = px.line(daily_comparison, x='Date', y='Trip_Number', color='Month',
                                   markers=True,
                                   title=f'Daily Trip Comparison: {start_date.strftime("%d-%b")} to {end_date.strftime("%d-%b")}',
                                   labels={'Trip_Number': 'Number of Trips', 'Date': 'Date/Day'})
                    st.plotly_chart(fig_daily, use_container_width=True)

                with daily_tab2:
                    # Customer-wise daily trend
                    daily_customer = current_mtd.groupby([current_mtd['Scheduled At'].dt.date, 'Customer'])['Trip_Number']\
                                            .nunique().unstack().fillna(0)
                    
                    # Sort customers by total trips (descending) and take top N for better visualization
                    top_customers = current_mtd['Customer'].value_counts().head(10).index.tolist()
                    daily_customer = daily_customer[top_customers]
                    
                    fig_customer_daily = px.bar(daily_customer, 
                                             x=daily_customer.index, 
                                             y=daily_customer.columns,
                                             title=f'Customer-wise Daily Trips: {start_date.strftime("%d-%b")} to {end_date.strftime("%d-%b")}',
                                             labels={'value': 'Number of Trips', 'Scheduled At': 'Date', 'variable': 'Customer'},
                                             barmode='stack')
                    
                    fig_customer_daily.update_layout(legend_title_text='Customers',
                                                  xaxis_title='Date',
                                                  yaxis_title='Number of Trips')
                    
                    st.plotly_chart(fig_customer_daily, use_container_width=True)
                    
                    # Show raw data option
                    if st.checkbox("Show customer-wise daily data", key="trips_customer_daily_data"):
                        st.dataframe(daily_customer)

            # Customer-wise trip trends
            st.subheader("Customer-wise Trip Trends")

            customer_monthly = completed_trips.groupby(['Customer', completed_trips['Scheduled At'].dt.to_period('M')])['Trip_Number']\
                                 .nunique().unstack().fillna(0).astype(int)

            months = customer_monthly.columns.sort_values(ascending=False)
            current_month_col = months[0]
            last_month_col = months[1] if len(months) > 1 else None

            if last_month_col is not None:
                customer_monthly['Difference'] = customer_monthly[current_month_col] - customer_monthly[last_month_col]
                customer_monthly['%Growth'] = np.where(
                    customer_monthly[last_month_col] > 0,
                    (customer_monthly[current_month_col] - customer_monthly[last_month_col]) /
                    customer_monthly[last_month_col] * 100,
                    0
                )
                customer_monthly['%Growth'] = customer_monthly['%Growth'].round(1)

            grand_total = customer_monthly.sum().to_frame().T
            grand_total.index = ['Grand Total']

            # Calculate correct %Growth for Grand Total
            if last_month_col is not None:
                total_current = grand_total[current_month_col].values[0]
                total_last = grand_total[last_month_col].values[0]
                if total_last > 0:
                    grand_total['%Growth'] = ((total_current - total_last) / total_last * 100).round(1)
                else:
                    grand_total['%Growth'] = 0
                grand_total['Difference'] = total_current - total_last

            customer_monthly = pd.concat([customer_monthly, grand_total])

            customer_monthly = customer_monthly.sort_values(current_month_col, ascending=False)

            def style_negative_positive(val):
                color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                return f'color: {color}'

            styled_df = customer_monthly.style \
                .applymap(style_negative_positive, subset=['Difference', '%Growth']) \
                .format("{:.0f}", subset=[col for col in customer_monthly.columns if col not in ['%Growth']]) \
                .format("{:.1f}%", subset=['%Growth']) \
                .background_gradient(cmap='Blues',
                                   subset=[col for col in customer_monthly.columns if col not in ['Difference', '%Growth']])

            st.dataframe(styled_df, use_container_width=True)

            # Hub-wise comparison (additional feature)
            st.subheader("Hub-wise Comparison")
            
            # Let user select a customer first
            selected_customer = st.selectbox("Select Customer for Hub-wise Analysis", 
                                           options=completed_trips['Customer'].unique(),
                                           index=0,
                                           key="hub_customer_select")
            
            # Filter data for selected customer
            customer_data = completed_trips[completed_trips['Customer'] == selected_customer]
            
            # Current and last month hub data
            current_hub_data = customer_data[(customer_data['Scheduled At'].dt.month == current_month) &
                                           (customer_data['Scheduled At'].dt.year == current_year)]
            
            last_hub_data = customer_data[(customer_data['Scheduled At'].dt.month == last_month) &
                                        (customer_data['Scheduled At'].dt.year == last_year)]
            
            # Group by hub
            current_hub = current_hub_data.groupby('Hub')['Trip_Number'].nunique().reset_index()
            current_hub.columns = ['Hub', 'Current Month Trips']
            
            last_hub = last_hub_data.groupby('Hub')['Trip_Number'].nunique().reset_index()
            last_hub.columns = ['Hub', 'Last Month Trips']
            
            # Merge the data
            hub_comparison = pd.merge(current_hub, last_hub, on='Hub', how='outer').fillna(0)
            
            # Calculate differences and growth percentages
            hub_comparison['Difference'] = hub_comparison['Current Month Trips'] - hub_comparison['Last Month Trips']
            hub_comparison['Growth %'] = np.where(
                hub_comparison['Last Month Trips'] > 0,
                (hub_comparison['Difference'] / hub_comparison['Last Month Trips']) * 100,
                0
            )
            
            # Add grand total row
            grand_total_hub = pd.DataFrame({
                'Hub': ['TOTAL'],
                'Current Month Trips': [hub_comparison['Current Month Trips'].sum()],
                'Last Month Trips': [hub_comparison['Last Month Trips'].sum()],
                'Difference': [hub_comparison['Difference'].sum()],
                'Growth %': [
                    (hub_comparison['Difference'].sum() / hub_comparison['Last Month Trips'].sum() * 100
                    if hub_comparison['Last Month Trips'].sum() > 0 else 0)
                ]
            })
            
            hub_comparison = pd.concat([hub_comparison, grand_total_hub], ignore_index=True)
            
            # Format the table
            styled_hub_comparison = hub_comparison.style \
                .applymap(color_negative_red, subset=['Difference', 'Growth %']) \
                .format({
                    'Current Month Trips': '{:.0f}',
                    'Last Month Trips': '{:.0f}',
                    'Difference': '{:.0f}',
                    'Growth %': '{:.1f}%'
                }) \
                .background_gradient(cmap='Blues', subset=['Current Month Trips', 'Last Month Trips'])
            
            st.dataframe(styled_hub_comparison, use_container_width=True)
            
            # Optional raw data
            if st.checkbox("Show raw trips data", key="trips_raw_data"):
                st.subheader("Raw Data")
                st.dataframe(df_trips)

        except Exception as e:
            st.error(f"Error processing trips data: {e}")