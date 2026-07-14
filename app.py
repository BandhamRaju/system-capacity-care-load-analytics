# ==========================================================
# SYSTEM CAPACITY & CARE LOAD ANALYTICS
# Streamlit Dashboard
# Author : Bandham Raju
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from prophet import Prophet

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="Healthcare Capacity Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------

st.markdown(
    """
<style>

.main{
    background-color:#0E1117;
}

h1,h2,h3,h4{
    color:white;
}

div[data-testid="metric-container"]{
    background-color:#1b2631;
    padding:15px;
    border-radius:12px;
    border:1px solid #2c3e50;
}

.sidebar .sidebar-content{
    background:#111827;
}

</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------


@st.cache_data
def load_data():

    df = pd.read_csv("Outputs/cleaned_healthcare_data.csv", parse_dates=["Date"])

    return df


df = load_data()

# ----------------------------------------------------------
# LOAD FORECAST
# ----------------------------------------------------------


@st.cache_data
def load_forecast():

    forecast = pd.read_csv("Outputs/forecast_90_days.csv", parse_dates=["Date"])

    return forecast


forecast_df = load_forecast()

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/hospital-3.png", width=80)

st.sidebar.title("Healthcare Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "EDA",
        "Comparative Analysis",
        "Load Pressure",
        "Capacity Utilization",
        "Peak Demand",
        "Operational Efficiency",
        "Forecasting",
        "Downloads",
        "About",
    ],
)

st.sidebar.divider()

st.sidebar.write("Dataset")

st.sidebar.success("HHS Unaccompanied Children")

st.sidebar.write("Records")

st.sidebar.info(f"{len(df):,}")

st.sidebar.write("Project")

st.sidebar.success("Data Science")

# ----------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------


def dashboard_page():

    st.title("🏥 Healthcare Capacity Dashboard")

    st.markdown("""
    This dashboard provides operational insights into healthcare
    capacity, care load, forecasting and system performance.
    """)
    st.markdown("""
        ### 📋 Project Overview

        This dashboard analyzes healthcare system capacity for Unaccompanied Children (UC) by examining historical care trends, operational workload, forecasting future demand, and evaluating healthcare resource utilization. The objective is to support evidence-based planning and improve healthcare capacity management through data-driven insights.
        """)
    st.markdown("""
        ### 🎯 Dashboard Objectives

        - Monitor healthcare demand over time.
        - Compare CBP Custody and HHS Care populations.
        - Analyze monthly workload patterns.
        - Understand demand distribution.
        - Forecast future healthcare capacity requirements.
        - Support operational and strategic decision-making.
        """)

    st.divider()

    # --------------------------------------------------
    # Date Filter
    # --------------------------------------------------

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    if start_date > end_date:
        st.error("Start Date cannot be greater than End Date.")
        st.stop()

    dashboard_df = df[
        (df["Date"] >= pd.to_datetime(start_date))
        & (df["Date"] <= pd.to_datetime(end_date))
    ]

    # ---------------- KPI ----------------

    avg_children = int(dashboard_df["Total Children Under Care"].mean())

    avg_cbp = int(dashboard_df["Children in CBP custody"].mean())

    avg_hhs = int(dashboard_df["Children in HHS Care"].mean())

    avg_pressure = dashboard_df["Net Intake Pressure"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Children Under Care", f"{avg_children:,}")

    k2.metric("CBP Custody", f"{avg_cbp:,}")

    k3.metric("HHS Care", f"{avg_hhs:,}")

    k4.metric("Net Intake Pressure", f"{avg_pressure:.2f}")

    st.info("""
        ### 📊 Key Performance Indicators

        The KPI cards provide a real-time summary of the selected period. They display the average number of children under care, CBP Custody population, HHS Care population, and Net Intake Pressure, enabling quick assessment of healthcare system capacity.
        """)


    st.divider()
    st.success("""
        ### 📌 Dashboard Navigation

        • Review the KPIs for an overall system summary.

        • Explore historical healthcare demand trends.

        • Compare CBP Custody with HHS Care populations.

        • Analyze monthly workload patterns and demand distribution.

        • Switch to the Forecast tab to view future healthcare demand predictions.
        """)

    st.subheader("Healthcare Demand Trend")

    fig = px.line(
        dashboard_df,
        x="Date",
        y="Total Children Under Care",
        markers=True,
        title="Total Children Under Care",
    )

    fig.update_layout(template="plotly_dark", height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CBP vs HHS Care")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dashboard_df["Date"],
            y=dashboard_df["Children in CBP custody"],
            name="CBP Custody",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dashboard_df["Date"],
            y=dashboard_df["Children in HHS Care"],
            name="HHS Care",
        )
    )

    fig.update_layout(template="plotly_dark", height=500, title="CBP vs HHS Care Load")

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Monthly Healthcare Load")

    monthly = (
        dashboard_df.set_index("Date")
        .resample("ME")["Total Children Under Care"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        monthly,
        x="Date",
        y="Total Children Under Care",
        color="Total Children Under Care",
    )

    fig.update_layout(template="plotly_dark", height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution")

    fig = px.histogram(
        dashboard_df,
        x="Total Children Under Care",
        nbins=30,
        color_discrete_sequence=["orange"],
    )

    fig.update_layout(template="plotly_dark", height=450)

    st.plotly_chart(fig, use_container_width=True)
    # ----------------------------------------------------------


# EDA PAGE
# ----------------------------------------------------------


def eda_page():

    st.title("📊 Exploratory Data Analysis")

    st.markdown("""
    This section presents the major exploratory analyses performed on the
    healthcare dataset.
    """)

    graphs = [
        (
            "EDA 1 • Daily Children Under Care",
            "Graphs/eda_01_daily_total_children_under_care.png",
        ),
        ("EDA 2 • Rolling Average", "Graphs/eda_02_7day_rolling_average.png"),
        (
            "EDA 3 • Monthly Average Care Load",
            "Graphs/eda_03_monthly_average_total_children_under_care.png",
        ),
        ("EDA 4 • Growth Rate", "Graphs/eda_04_monthly_net_intake_pressure.png"),
        (
            "EDA 5 • Daily Care Load Volatility",
            "Graphs/eda_05_care_load_volatility_index.png",
        ),
        (
            "EDA 6 • Monthly Care Load Volatility",
            "Graphs/eda_06_monthly_care_load_volatility.png",
        ),
        (
            "EDA 7 • Monthly Net Intake Pressure",
            "Graphs/eda_07_monthly_net_intake_pressure.png",
        ),
        (
            "EDA 8 • Monthly Backlog Accumulation Rate",
            "Graphs/eda_08_monthly_backlog_accumulation_rate.png",
        ),
        (
            "EDA 9 • Monthly Discharge Offset Ratio",
            "Graphs/eda_09_monthly_discharge_offset_ratio.png",
        ),
        ("EDA 10 • Correlation Heatmap", "Graphs/eda_10_correlation_heatmap.png"),
        ("EDA 11 • CBP vs HHS Care Load", "Graphs/eda_11_cbp_vs_hhs_care_load.png"),
        (
            "EDA 12 • Distribution",
            "Graphs/eda_12_distribution_total_children_under_care.png",
        ),
    ]

    descriptions = {
        "EDA 1 • Daily Children Under Care": """
        Displays the daily trend of children under care throughout the study period. The chart highlights fluctuations in healthcare demand, identifies periods of rapid growth and decline, and provides a baseline understanding of overall system workload over time.
        """,
        "EDA 2 • Rolling Average": """
        Shows the 7-day rolling average of children under care to smooth daily fluctuations. This visualization reveals long-term demand patterns, making it easier to identify sustained increases or decreases in healthcare utilization.
        """,
        "EDA 3 • Monthly Average Care Load": """
        Presents the average number of children under care for each month. Monthly aggregation removes daily noise and helps identify seasonal variations, long-term trends, and shifts in healthcare capacity requirements.
        """,
        "EDA 4 • Growth Rate": """
        Illustrates the monthly Net Intake Pressure, representing the difference between admissions and discharges. Positive values indicate increasing operational demand, while negative values suggest improved discharge efficiency and reduced system pressure.
        """,
        "EDA 5 • Daily Care Load Volatility": """
        Measures day-to-day variability in healthcare demand. Higher volatility indicates unstable operational conditions requiring rapid resource adjustments, whereas lower volatility reflects more predictable system behavior.
        """,
        "EDA 6 • Monthly Care Load Volatility": """
        Shows the monthly average volatility of healthcare demand. This helps evaluate operational stability over longer periods and identifies months with unusually high fluctuations in patient load.
        """,
        "EDA 7 • Monthly Net Intake Pressure": """
        Displays the monthly balance between incoming and outgoing children within the healthcare system. This metric helps assess whether operational capacity is expanding, stabilizing, or experiencing sustained pressure.
        """,
        "EDA 8 • Monthly Backlog Accumulation Rate": """
        Represents the monthly rate at which pending cases accumulate within the system. Higher backlog rates indicate increasing operational strain and may signal insufficient capacity to process incoming demand efficiently.
        """,
        "EDA 9 • Monthly Discharge Offset Ratio": """
        Shows the relationship between successful discharges and new admissions. Higher values indicate stronger discharge performance, while declining values may suggest reduced operational efficiency and growing care demand.
        """,
        "EDA 10 • Correlation Heatmap": """
        Visualizes the relationships among major healthcare metrics using Pearson correlation coefficients. Strong positive correlations indicate variables that increase together, whereas negative correlations reveal inverse operational relationships.
        """,
        "EDA 11 • CBP vs HHS Care Load": """
        Compares children remaining in CBP custody with those transferred into HHS care over time. This analysis highlights patient flow between agencies and helps evaluate transfer efficiency and overall system coordination.
        """,
        "EDA 12 • Distribution": """
        Displays the statistical distribution of daily children under care. The histogram illustrates the frequency of different care-load levels, identifies common operating ranges, and reveals potential outliers or skewness within the dataset.
        """,
    }

    for title, image in graphs:

        st.subheader(title)

        st.image(image, use_container_width=True)

        st.info(descriptions[title])

        st.divider()


# ----------------------------------------------------------
# COMPARATIVE ANALYSIS
# ----------------------------------------------------------


def comparative_page():

    st.title("📈 Comparative Time Analysis")

    st.markdown("""
    Compare healthcare demand across different years.
    """)

    yearly = df.copy()

    yearly["Year"] = yearly["Date"].dt.year

    yearly["Month"] = yearly["Date"].dt.strftime("%b")

    order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    yearly["Month"] = pd.Categorical(yearly["Month"], categories=order, ordered=True)

    compare = (
        yearly.groupby(["Year", "Month"])["Total Children Under Care"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        compare,
        x="Month",
        y="Total Children Under Care",
        color="Year",
        markers=True,
        title="Year-wise Monthly Care Load Comparison",
    )

    fig.update_layout(template="plotly_dark", height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.info("""

The comparison illustrates how healthcare demand changed
between different years and highlights seasonal behaviour.

""")


# ----------------------------------------------------------
# LOAD PRESSURE
# ----------------------------------------------------------


def load_pressure_page():

    st.title("📊 Load Pressure Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Total Children Under Care"], name="Children Under Care"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Net Intake Pressure"] * 20, name="Net Intake Pressure"
        )
    )

    fig.update_layout(
        template="plotly_dark", height=600, title="Healthcare Load Pressure"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""

Higher pressure indicates increased operational demand
requiring greater healthcare capacity.

""")


# ----------------------------------------------------------
# CAPACITY UTILIZATION
# ----------------------------------------------------------


def capacity_page():

    st.title("🏥 Capacity Utilization")

    capacity = df.copy()

    capacity["Utilization"] = (
        capacity["Children in HHS Care"] / capacity["Total Children Under Care"]
    ) * 100

    fig = px.line(
        capacity, x="Date", y="Utilization", title="Healthcare Capacity Utilization"
    )

    fig.update_layout(template="plotly_dark", height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.info("""

Higher utilization indicates efficient placement
into HHS healthcare facilities.

""")
    # ----------------------------------------------------------


# PEAK DEMAND ANALYSIS
# ----------------------------------------------------------


def peak_page():

    st.title("📈 Peak Demand Analysis")

    top_days = df.nlargest(15, "Total Children Under Care").sort_values("Date")

    fig = px.bar(
        top_days,
        x="Date",
        y="Total Children Under Care",
        color="Total Children Under Care",
        title="Top 15 Peak Healthcare Demand Days",
    )

    fig.update_layout(template="plotly_dark", height=550)

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Peak demand periods identify dates when healthcare resources
experienced maximum operational workload.
""")


# ----------------------------------------------------------
# OPERATIONAL EFFICIENCY
# ----------------------------------------------------------


def operational_page():

    st.title("⚙ Operational Efficiency")

    monthly = (
        df.set_index("Date")[
            [
                "Children apprehended and placed in CBP custody",
                "Children transferred out of CBP custody",
            ]
        ]
        .resample("ME")
        .sum()
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["Date"],
            y=monthly["Children apprehended and placed in CBP custody"],
            name="Admissions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["Date"],
            y=monthly["Children transferred out of CBP custody"],
            name="Transfers",
        )
    )

    fig.update_layout(
        template="plotly_dark", height=600, title="Monthly Admissions vs Transfers"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
Balanced admissions and transfers indicate efficient patient
movement through the healthcare system.
""")


# ----------------------------------------------------------
# FORECASTING
# ----------------------------------------------------------


def forecasting_page():

    st.title("🤖 Machine Learning Forecasting")

    tab1, tab2, tab3 = st.tabs(
        ["Linear Regression", "Prophet Forecast", "Forecast Data"]
    )

    # ======================================================
    # TAB 1
    # ======================================================

    with tab1:

        st.subheader("Linear Regression")

        try:

            lr_model = joblib.load("Models/linear_regression_model.pkl")

            X = np.arange(len(df)).reshape(-1, 1)

            prediction = lr_model.predict(X)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["Date"], y=df["Total Children Under Care"], name="Actual"
                )
            )

            fig.add_trace(go.Scatter(x=df["Date"], y=prediction, name="Prediction"))

            fig.update_layout(
                template="plotly_dark", height=600, title="Actual vs Predicted"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.info("""
                ### Business Interpretation

                This visualization compares the actual daily healthcare demand with predictions generated using a Linear Regression model.

                The model identifies the long-term direction of healthcare demand by fitting a single trend line across the historical observations. While it captures the overall downward trend in capacity utilization, it does not accurately represent sudden fluctuations, seasonal effects, or structural changes within the system.

                This graph demonstrates that Linear Regression provides a simple baseline forecasting approach but is limited when modeling complex healthcare demand patterns.
                """)

        except Exception:

            st.warning("Linear Regression model not found.")

    # ======================================================
    # TAB 2
    # ======================================================

    with tab2:

        st.subheader("Prophet Forecast")

        try:

            prophet = joblib.load("Models/prophet_forecast_model.pkl")

            future = prophet.make_future_dataframe(periods=90)

            forecast = prophet.predict(future)

            fig1 = prophet.plot(forecast)

            st.pyplot(fig1)

            st.info("""
                ### Business Interpretation

                The Prophet forecasting model estimates future healthcare demand by learning long-term trends and recurring seasonal patterns from historical data.

                The black points represent observed historical values, while the blue line represents the model's estimated trend. The shaded region indicates the confidence interval of the prediction.

                Compared with Linear Regression, Prophet adapts more effectively to changing demand patterns and is better suited for operational forecasting and healthcare capacity planning.
                """)


            st.subheader("Trend Components")

            fig2 = prophet.plot_components(forecast)

            st.pyplot(fig2)
            st.info("""
                ### Business Interpretation

                This decomposition separates the forecast into its underlying components.

                • Trend illustrates the long-term increase or decrease in healthcare demand.

                • Weekly Seasonality identifies systematic demand variation across different days of the week.

                • Yearly Seasonality highlights recurring annual demand cycles.

                Understanding these components helps decision-makers distinguish long-term structural changes from normal seasonal fluctuations and supports more informed operational planning.
                """)
            
        except Exception:

            st.warning("Prophet model not found.")

    # ======================================================
    # TAB 3
    # ======================================================

    with tab3:

        st.subheader("Forecast Table")

        st.dataframe(forecast_df, use_container_width=True)

        st.info("""
            ### Business Interpretation

            The forecast table provides future projected healthcare demand generated by the forecasting model.

            Key columns include:

            • Date – Forecast period

            • Predicted Demand – Expected healthcare capacity requirement

            • Lower Bound – Conservative estimate

            • Upper Bound – Optimistic estimate

            These projections support workforce planning, capacity management, and proactive resource allocation for healthcare administrators.
            """)

        csv = forecast_df.to_csv(index=False)

        st.download_button(
            "⬇ Download Forecast", csv, "forecast_90_days.csv", "text/csv"
        )

        st.metric("Forecast Days", len(forecast_df))

        st.metric("Average Forecast", round(forecast_df["Forecast"].mean(), 2))
    
    
# ----------------------------------------------------------

    # DOWNLOADS PAGE
# ----------------------------------------------------------


def downloads_page():

    st.title("📥 Downloads")

    st.markdown("""
Download project outputs generated during the analysis.
""")

    # -----------------------------
    # Cleaned Dataset
    # -----------------------------

    try:

        with open("Outputs/cleaned_healthcare_data.csv", "rb") as file:

            st.download_button(
                label="⬇ Download Cleaned Dataset",
                data=file,
                file_name="cleaned_healthcare_data.csv",
                mime="text/csv",
            )

    except:

        st.error("Cleaned dataset not found.")

    # -----------------------------
    # Forecast Dataset
    # -----------------------------

    try:

        with open("Outputs/forecast_90_days.csv", "rb") as file:

            st.download_button(
                label="⬇ Download Forecast Dataset",
                data=file,
                file_name="forecast_90_days.csv",
                mime="text/csv",
            )

    except:

        st.error("Forecast file not found.")

    # -----------------------------
    # Executive Summary
    # -----------------------------

    try:

        with open("Outputs/executive_summary.csv", "rb") as file:

            st.download_button(
                label="⬇ Download Executive Summary",
                data=file,
                file_name="executive_summary.csv",
                mime="text/csv",
            )

    except:

        st.error("Executive Summary not found.")


# ----------------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------------


def about_page():

    st.title("ℹ About Project")

    st.markdown("""
# System Capacity & Care Load Analytics

This project analyzes operational healthcare capacity
for Unaccompanied Children.

The project demonstrates an end-to-end Data Science workflow
covering:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Comparative Time Analysis
- Load Pressure Analysis
- Capacity Utilization
- Machine Learning
- Forecasting
- Business Insights
""")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📂 Dataset")

        st.write("Healthcare.gov")

        st.write("Historical Daily Data")

        st.write(f"Records : {len(df):,}")

    with col2:

        st.subheader("🛠 Technology")

        st.write("Python")

        st.write("Pandas")

        st.write("Plotly")

        st.write("Prophet")

        st.write("Scikit-Learn")

        st.write("Streamlit")

    st.divider()

    st.subheader("📊 KPIs")

    st.markdown("""

- Total Children Under Care

- Net Intake Pressure

- Care Load Volatility

- Backlog Accumulation Rate

- Capacity Utilization

- Discharge Offset Ratio

""")

    st.divider()

    st.subheader("🎯 Project Objective")

    st.info("""

Provide healthcare administrators with
historical analytics,
capacity monitoring,
operational insights,
and future forecasting.

""")


# ----------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------

if page == "Dashboard":

    dashboard_page()

elif page == "EDA":

    eda_page()

elif page == "Comparative Analysis":

    comparative_page()

elif page == "Load Pressure":

    load_pressure_page()

elif page == "Capacity Utilization":

    capacity_page()

elif page == "Peak Demand":

    peak_page()

elif page == "Operational Efficiency":

    operational_page()

elif page == "Forecasting":

    forecasting_page()

elif page == "Downloads":

    downloads_page()

elif page == "About":

    about_page()

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.divider()

st.markdown(
    """
<center>

### 🏥 System Capacity & Care Load Analytics

Developed using

Python • Streamlit • Plotly • Prophet • Scikit-Learn

© 2026 Bandham Raju

</center>
""",
    unsafe_allow_html=True,
)
