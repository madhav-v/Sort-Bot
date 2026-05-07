# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta
import time

# ================== CONFIG ==================
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "events.csv")
CATEGORIES = ["Plastic", "Burnable", "Cans", "Bottles", "Others"]

# Ensure data folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)

# Create sample CSV if missing
if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=["ts", "category", "confidence", "source", "note"])
    df_init.to_csv(DATA_FILE, index=False)

# ================== PAGE STYLE ==================
st.set_page_config(
    page_title="SortBot Dashboard", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
body { background-color: #0a0a0f; color: #eee; }
[data-testid="stMetricValue"] { color: #c084fc !important; }
h1, h2, h3, h4 { color: #c084fc; }
.sidebar .sidebar-content { background: #111 !important; }
div[data-testid="stMetric"] {
    background-color: #1a1a2e;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #c084fc;
}
</style>
""",
    unsafe_allow_html=True,
)

# ================== AUTO-REFRESH MECHANISM ==================
# Add a refresh button and auto-refresh option
st.sidebar.markdown("### 🔄 Refresh Settings")
col_refresh1, col_refresh2 = st.sidebar.columns([1, 1])
with col_refresh1:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col_refresh2:
    auto_refresh = st.checkbox("Auto-refresh", value=True)

refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 2)

if auto_refresh:
    # Auto-refresh at specified rate
    time.sleep(refresh_rate)
    st.cache_data.clear()
    st.rerun()


# ================== LOAD DATA (NO CACHING FOR REAL-TIME) ==================
def load_data():
    """Load data without caching for real-time updates"""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, parse_dates=["ts"], on_bad_lines="skip")
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(
                columns=["ts", "category", "confidence", "source", "note"]
            )
    return pd.DataFrame(columns=["ts", "category", "confidence", "source", "note"])


df_full = load_data()

# ================== REAL-TIME KPI DATA (NO CACHING) ==================
if os.path.exists(DATA_FILE):
    try:
        df_rt = pd.read_csv(
            DATA_FILE, usecols=["ts", "category", "confidence"], parse_dates=["ts"]
        )
        df_rt = df_rt.dropna(subset=["ts", "category"])
        # Ensure timezone awareness
        if len(df_rt) > 0:
            if df_rt["ts"].dt.tz is None:
                df_rt["ts"] = df_rt["ts"].dt.tz_localize("UTC")
            else:
                df_rt["ts"] = df_rt["ts"].dt.tz_convert("UTC")
    except Exception as e:
        st.error(f"Error loading real-time data: {e}")
        df_rt = pd.DataFrame(columns=["ts", "category", "confidence"])
else:
    df_rt = pd.DataFrame(columns=["ts", "category", "confidence"])

# Time window selector
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
minutes = st.sidebar.slider("Time window (minutes)", 1, 240, 60)

if len(df_rt) > 0:
    now = pd.Timestamp.now(tz="UTC")
    window_start = now - pd.Timedelta(minutes=minutes)
    df_rt = df_rt[df_rt["ts"] >= window_start]

# ================== HEADER ==================
st.title("🤖 SortBot Live Sorting Dashboard")
st.markdown("Tracking item sorting activity in **real time**.")

# Show last update time
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ================== KPI CARDS ==================
st.markdown("### 📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)

total_events = len(df_rt)
c1.metric("Events (window)", f"{total_events:,}")

events_per_min = total_events / max(1, minutes)
c2.metric("Avg / min", f"{events_per_min:.2f}")

if len(df_rt) > 0:
    last = df_rt.iloc[-1]
    c3.metric(
        "Last event",
        f"{last['category']}",
        f"{float(last['confidence']):.2f} confidence",
    )
else:
    c3.metric("Last event", "No data")

if len(df_rt) > 0:
    most = df_rt["category"].value_counts().idxmax()
    count_most = df_rt["category"].value_counts().max()
    c4.metric("Most frequent", most, f"{count_most} items")
else:
    c4.metric("Most frequent", "—")

st.markdown("---")

# ================== CHARTS ==================
if len(df_full) > 0:
    # Ensure timezone awareness for full dataset
    if df_full["ts"].dt.tz is None:
        df_full["ts"] = df_full["ts"].dt.tz_localize("UTC")
    else:
        df_full["ts"] = df_full["ts"].dt.tz_convert("UTC")

    if len(df_rt) > 0:
        now = pd.Timestamp.now(tz="UTC")
        window_start = now - pd.Timedelta(minutes=minutes)
        df_window = df_full[df_full["ts"] >= window_start]
    else:
        df_window = df_full

    # Bar chart
    counts = (
        df_window["category"].value_counts().reindex(CATEGORIES).fillna(0).astype(int)
    )
    fig_bar = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Category", "y": "Count"},
        title="📦 Sorted Items by Category",
        text=counts.values,
    )
    fig_bar.update_traces(marker_color="#c084fc", textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )

    # Time series
    if len(df_window) > 0:
        df_ts = (
            df_window.set_index("ts")
            .resample("1min")
            .size()
            .rename("count")
            .reset_index()
        )
        fig_ts = px.area(
            df_ts, x="ts", y="count", title="📈 Events Over Time (per minute)"
        )
        fig_ts.update_traces(line_color="#c084fc", fillcolor="rgba(192, 132, 252, 0.3)")
    else:
        fig_ts = px.area(title="📈 Events Over Time (per minute)")

    fig_ts.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
else:
    counts = pd.Series([0] * 5, index=CATEGORIES)
    fig_bar = px.bar(title="📦 Sorted Items by Category")
    fig_ts = px.area(title="📈 Events Over Time")

colA, colB = st.columns([2, 1])

with colA:
    st.plotly_chart(fig_bar, use_container_width=True)
    st.plotly_chart(fig_ts, use_container_width=True)

with colB:
    # Pie chart
    fig_pie = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.4,
        title="🥧 Category Distribution",
    )
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Recent events table
    st.markdown("### 📋 Recent Events")
    if len(df_rt) > 0:
        df_show = df_rt.sort_values("ts", ascending=False).head(20).copy()
        df_show["ts"] = (
            df_show["ts"].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_show["confidence"] = df_show["confidence"].round(2)
        st.dataframe(df_show, height=360, use_container_width=True)
    else:
        st.info("No events recorded yet. Start sorting items!")

# ================== SIDEBAR INFO ==================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.info(
    f"""
**Total Events:** {len(df_full):,}  
**Window Events:** {len(df_rt):,}  
**Categories:** {len(CATEGORIES)}  
**Data File:** `events.csv`
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Category Legend")
colors_legend = {
    "Plastic": "🔵",
    "Burnable": "🟠",
    "Cans": "⚪",
    "Bottles": "🟢",
    "Others": "⚫",
}
for cat, emoji in colors_legend.items():
    st.sidebar.markdown(f"{emoji} **{cat}**")

# ================== FOOTER ==================
st.markdown("---")
st.caption(
    "SortBot Dashboard • Real-time waste sorting visualization • Powered by Streamlit"
)
