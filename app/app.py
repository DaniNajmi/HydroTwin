import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="HydroTwin Dashboard",
    page_icon="💧",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/hydro_twin_app_data.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # If timestamp was saved as index, recover it
    if "timestamp" not in df.columns:
        possible_time_cols = [
            col for col in df.columns
            if "time" in col.lower() or "date" in col.lower() or "unnamed" in col.lower()
        ]

        if possible_time_cols:
            df = df.rename(columns={possible_time_cols[0]: "timestamp"})
        else:
            st.error("No timestamp column found. Available columns are:")
            st.write(df.columns.tolist())
            st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df.sort_values(["pond_id", "timestamp"]).reset_index(drop=True)

df = load_data()

st.title("HydroTwin Monitoring System")
st.caption(
    "Interactive aquaponics monitoring system for prediction, anomaly detection, "
    "state classification, and trust scoring."
    "Developed by Daniela Najmias Lang"
)

with st.expander("How to use this dashboard", expanded=True):
    st.markdown(
        """
        This dashboard shows how HydroTwin monitors an aquaponics system.

        **1. Select a pond**  
        Use the sidebar to choose which pond/system you want to inspect.

        **2. Adjust the residual threshold**  
        The threshold controls how sensitive the system is to prediction errors.
        - Lower threshold = more sensitive, more anomalies flagged
        - Higher threshold = stricter, fewer anomalies flagged

        **3. Compare anomaly layers**  
        - **Residual anomalies** show when observed ammonia does not match what the model expected.
        - **Isolation Forest anomalies** show unusual system behavior across multiple variables.
        - When both appear together, the system has stronger evidence of abnormal behavior.

        **4. Read the system state**  
        HydroTwin translates technical outputs into operational categories such as Normal, Sensor Drift, Biological Warning, Biological Danger, or Critical.

        **5. Use the trust score**  
        The trust score summarizes how reliable the current system reading appears.
        A score near 1 means high confidence. A lower score means the system should be inspected.
        """
    )

# Sidebar
st.sidebar.header("System Controls")

ponds = sorted(df["pond_id"].unique())
selected_pond = st.sidebar.selectbox("Select Pond", ponds)

threshold_q = st.sidebar.slider(
    "Residual anomaly threshold",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)

show_sensor = st.sidebar.checkbox("Show residual anomalies", True)
show_iso = st.sidebar.checkbox("Show Isolation Forest anomalies", True)
show_states = st.sidebar.checkbox("Show system states", True)

df_pond = df[df["pond_id"] == selected_pond].copy().reset_index(drop=True)

# Interactive threshold recalculation
threshold = df_pond["residual_abs"].quantile(threshold_q)
df_pond["interactive_sensor_flag"] = df_pond["residual_abs"] > threshold

# KPI cards
latest = df_pond.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current State", latest["state"])
col2.metric("Trust Score", f"{latest['trust_smooth']:.2f}")
col3.metric("Residual Threshold", f"{threshold:.3f}")
col4.metric("Residual Flags", int(df_pond["interactive_sensor_flag"].sum()))

# Explanation
st.info(
    """
    HydroTwin compares expected ammonia behavior against observed ammonia. 
    Large residuals suggest potential sensor-side issues, while Isolation Forest 
    captures unusual multivariate system behavior. Together, these signals classify 
    system state and generate an operator-facing trust score.
    """
)

with st.expander("What do these metrics mean?"):
    st.markdown(
        """
        **Current State**  
        The latest operational classification for the selected pond.

        **Trust Score**  
        A 0–1 reliability score based on prediction error and anomaly flags.

        **Residual Threshold**  
        The current cutoff used to decide whether a prediction error is unusually large.

        **Residual Flags**  
        Number of readings where observed ammonia differed strongly from expected ammonia.

        **Actual Ammonia**  
        The measured ammonia value from the system.

        **Predicted Ammonia**  
        The ammonia value Hydro-Twin expected based on learned biological patterns.

        **Residual**  
        The difference between actual and predicted ammonia. Large residuals may indicate sensor drift or unusual biological behavior.
        """
    )
# Actual vs predicted
st.subheader("Actual vs Predicted Ammonia")

fig_pred = px.line(
    df_pond,
    x="timestamp",
    y=["actual", "predicted"],
    labels={"value": "Ammonia", "timestamp": "Time", "variable": "Signal"},
    title="Observed vs Expected Ammonia"
)

st.plotly_chart(fig_pred, use_container_width=True)

# Anomaly detection
st.subheader("Anomaly Detection Layer")

fig_anom = px.line(
    df_pond,
    x="timestamp",
    y="actual",
    title="Ammonia with Anomaly Overlays",
    labels={"actual": "Ammonia", "timestamp": "Time"}
)

if show_sensor:
    sensor_df = df_pond[df_pond["interactive_sensor_flag"]]
    fig_anom.add_scatter(
        x=sensor_df["timestamp"],
        y=sensor_df["actual"],
        mode="markers",
        name="Residual anomaly"
    )

if show_iso:
    iso_df = df_pond[df_pond["iso_flag"] == True]
    fig_anom.add_scatter(
        x=iso_df["timestamp"],
        y=iso_df["actual"],
        mode="markers",
        name="Isolation Forest anomaly"
    )

st.plotly_chart(fig_anom, use_container_width=True)

# State classification
if show_states:
    st.subheader("System State Classification")

    fig_state = px.scatter(
        df_pond,
        x="timestamp",
        y="actual",
        color="state",
        title="HydroTwin System State Over Time",
        labels={"actual": "Ammonia", "timestamp": "Time", "state": "System State"},
        hover_data=["predicted", "residual_abs", "trust_smooth"]
    )

    st.plotly_chart(fig_state, use_container_width=True)

# Trust score
st.subheader("Trust Score")

fig_trust = px.line(
    df_pond,
    x="timestamp",
    y="trust_smooth",
    title="Smoothed Trust Score Over Time",
    labels={"trust_smooth": "Trust Score", "timestamp": "Time"}
)

st.plotly_chart(fig_trust, use_container_width=True)

# Decision panel
st.subheader("Current Operational Interpretation")

current_state = latest["state"]
current_trust = latest["trust_smooth"]

if current_state == "Normal":
    st.success(f"System appears stable. Trust score: {current_trust:.2f}")

elif "Sensor" in current_state:
    st.warning(
        f"Possible sensor-side issue detected. Trust score: {current_trust:.2f}. "
        "Recommended action: inspect or recalibrate sensor."
    )

elif "Biological" in current_state:
    st.warning(
        f"Biological risk signal detected. Trust score: {current_trust:.2f}. "
        "Recommended action: inspect water quality and system conditions."
    )

elif "CRITICAL" in current_state:
    st.error(
        f"Critical condition detected. Trust score: {current_trust:.2f}. "
        "Recommended action: immediate inspection of both sensor reliability and biological condition."
    )

else:
    st.info(f"Current state: {current_state}. Trust score: {current_trust:.2f}.")
