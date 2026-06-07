"""
app_dashboard.py
================
Streamlit Dashboard — AI-Based Smart Traffic Management System
Developed by: Akshit Rai (225811350) — B.Tech IT, MIT Bengaluru, May 2026

Upgrades in this version:
  - API Key header sent with every request
  - Congestion prediction panel (ML model output)
  - CSV download button
  - Database record counter in sidebar
  - Cleaner layout and prediction cards
"""

import time
import random
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_BASE  = "http://127.0.0.1:8000"
ANALYTICS_URL = f"{BACKEND_BASE}/api/traffic/analytics"
UPDATE_URL    = f"{BACKEND_BASE}/api/traffic/update"
EXPORT_URL    = f"{BACKEND_BASE}/api/traffic/export/csv"
API_KEY       = "traffic2026"
HEADERS       = {"X-API-Key": API_KEY}
LANE_NAMES    = ["North", "South", "East", "West"]

CONGESTION_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, val in [("emergency_lane", None), ("emergency_triggered", False), ("auto_refresh", True)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def fetch_analytics() -> dict:
    try:
        resp = requests.get(ANALYTICS_URL, headers=HEADERS, timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}

def post_emergency_trigger(lane_name: str):
    try:
        requests.post(
            f"{BACKEND_BASE}/api/traffic/trigger_emergency",
            json={"lane": lane_name},
            headers=HEADERS,
            timeout=3,
        )
    except Exception:
        pass

def post_clear_emergency():
    try:
        requests.post(f"{BACKEND_BASE}/api/traffic/clear_emergency", headers=HEADERS, timeout=3)
    except Exception:
        pass

def seconds_remaining(green_duration: int, last_updated_str: str) -> int:
    try:
        from datetime import timezone
        updated_at = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
        elapsed    = (datetime.now(timezone.utc) - updated_at).total_seconds()
        return max(0, int(green_duration - elapsed))
    except Exception:
        return green_duration

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/6/6a/Manipal_Institute_of_Technology_logo.png",
        width=120,
    )
    st.markdown("## 🚦 AI Traffic Manager")
    st.markdown(
        "**Thesis Project**  \n"
        "Akshit Rai · 225811350  \n"
        "B.Tech IT · MIT Bengaluru  \n"
        "May 2026"
    )
    st.divider()

    st.session_state.auto_refresh = st.toggle("Auto-Refresh Dashboard", value=st.session_state.auto_refresh)
    refresh_rate = st.slider("Refresh interval (s)", 1, 10, 2)

    st.divider()
    st.markdown("### 🚨 Emergency Simulation")
    emg_lane_choice = st.selectbox("Target Lane", LANE_NAMES, index=0)

    if st.button("🚑 Trigger Emergency Vehicle", type="primary", use_container_width=True):
        st.session_state.emergency_lane      = emg_lane_choice
        st.session_state.emergency_triggered = True
        post_emergency_trigger(emg_lane_choice)
        st.success(f"Emergency injected into **{emg_lane_choice}** lane!")

    if st.session_state.emergency_triggered:
        if st.button("✅ Clear Emergency", use_container_width=True):
            st.session_state.emergency_lane      = None
            st.session_state.emergency_triggered = False
            post_clear_emergency()
            st.success("Emergency cleared — signals reset to normal.")

    st.divider()
    st.markdown("### 📥 Export Data")
    if st.button("⬇️ Download Signal Log CSV", use_container_width=True):
        try:
            r = requests.get(EXPORT_URL, headers=HEADERS, timeout=5)
            st.download_button(
                label="💾 Save CSV File",
                data=r.content,
                file_name="traffic_signal_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            st.error("Could not fetch CSV — is the backend running?")

    st.divider()
    st.markdown("### 🔗 Backend Status")
    health_ok  = False
    db_records = 0
    try:
        r = requests.get(f"{BACKEND_BASE}/api/traffic/health", headers=HEADERS, timeout=2)
        if r.status_code == 200:
            health_ok  = True
            db_records = r.json().get("db_records", 0)
    except Exception:
        pass

    if health_ok:
        st.success("✅ FastAPI backend online")
        st.caption(f"🗄️ SQLite records: **{db_records:,}**")
    else:
        st.error("❌ Backend offline")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🚦 AI-Based Smart Traffic Management & Congestion Prediction</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:grey;'>Real-time adaptive signal optimization · "
    "YOLOv8 · FastAPI · SQLite · Scikit-learn · Streamlit</p>",
    unsafe_allow_html=True,
)

if st.session_state.emergency_triggered:
    st.error(
        f"🚨 EMERGENCY ACTIVE — **{st.session_state.emergency_lane}** lane → 60s GREEN · "
        f"All other lanes preempted to 10s"
    )

st.divider()

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
data = fetch_analytics()

if data and data.get("current_lanes"):
    lanes_raw  = data["current_lanes"]
    kpis       = data.get("performance_kpis", {})
    srv_info   = data.get("server_info", {})
    recent_log = data.get("recent_log", [])
    predictions = data.get("predictions", {})
else:
    lanes_raw = [
        {
            "name": n, "vehicle_count": random.randint(1,12),
            "density_score": round(random.uniform(10,90),1),
            "congestion_level": random.choice(["Low","Medium","High"]),
            "emergency_flag": False, "vehicle_types": {"car":3,"truck":1},
            "green_duration": random.choice([15,30,45]),
            "preempted": False, "signal_phase": "GREEN",
            "last_updated": datetime.utcnow().isoformat()+"Z",
        } for n in LANE_NAMES
    ]
    kpis = {}; srv_info = {}; recent_log = []; predictions = {}
    st.warning("⚠️ Backend offline — showing demo data.")

df_lanes = pd.DataFrame(lanes_raw)

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
st.markdown("## 📊 Intersection KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🚗 Total Vehicles",    int(df_lanes["vehicle_count"].sum()) if not df_lanes.empty else 0)
k2.metric("📈 Avg Density",       f"{round(df_lanes['density_score'].mean(),1) if not df_lanes.empty else 0}%")
k3.metric("🟢 Avg Green Time",    f"{kpis.get('avg_green_duration_s', 0)}s")
k4.metric("🚨 Emergency Events",  kpis.get("emergency_activations", 0))
k5.metric("🗄️ DB Records",        f"{srv_info.get('total_updates',0) * 4:,}")
st.divider()

# ---------------------------------------------------------------------------
# Lane table + countdown
# ---------------------------------------------------------------------------
st.markdown("## 🛣️ Live Lane Status & Signal Timings")
col_table, col_timers = st.columns([3, 2])

with col_table:
    st.markdown("### Lane Detail Table")
    rows = []
    for lane in lanes_raw:
        cong = lane.get("congestion_level","Low")
        rows.append({
            "Lane":           lane.get("name"),
            "Vehicles":       lane.get("vehicle_count"),
            "Density Score":  f"{lane.get('density_score',0)}%",
            "Congestion":     f"🔴 {cong}" if cong=="High" else (f"🟡 {cong}" if cong=="Medium" else f"🟢 {cong}"),
            "Green Time (s)": lane.get("green_duration"),
            "Status":         "⏸ Preempted" if lane.get("preempted") else "✅ Normal",
            "Emergency":      "🚑 YES" if lane.get("emergency_flag") else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with col_timers:
    st.markdown("### 🟢 Green Signal Countdown")
    for lane in lanes_raw:
        green_dur = lane.get("green_duration", 15)
        last_upd  = lane.get("last_updated", datetime.utcnow().isoformat()+"Z")
        remaining = seconds_remaining(green_dur, last_upd)
        pct       = remaining / green_dur if green_dur > 0 else 0.0
        emg_flag  = lane.get("emergency_flag", False)
        pre_flag  = lane.get("preempted", False)
        icon      = "🚑" if emg_flag else ("⏸" if pre_flag else "🟢")
        st.markdown(f"{icon} **{lane.get('name')}** — {remaining}s / {green_dur}s")
        st.progress(pct)

st.divider()

# ---------------------------------------------------------------------------
# Density gauges
# ---------------------------------------------------------------------------
st.markdown("## 🔍 Per-Lane Density Gauges")
gcols = st.columns(4)
for i, lane in enumerate(lanes_raw):
    score = lane.get("density_score", 0.0)
    cong  = lane.get("congestion_level", "Low")
    color = CONGESTION_COLORS.get(cong, "#22c55e")
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": lane.get("name"), "font": {"size": 14}},
        gauge={
            "axis":  {"range": [0,100]},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,35],  "color": "#d1fae5"},
                {"range": [35,65], "color": "#fef3c7"},
                {"range": [65,100],"color": "#fee2e2"},
            ],
        },
        number={"suffix": "%", "font": {"size": 18}},
    ))
    fig.update_layout(height=200, margin=dict(t=30,b=10,l=10,r=10))
    gcols[i].plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# ML Prediction panel
# ---------------------------------------------------------------------------
st.markdown("## 🤖 ML Congestion Predictions (5 Minutes Ahead)")
st.caption("Linear Regression model trained on historical signal data from SQLite database.")

if predictions:
    pred_cols = st.columns(4)
    for i, lane_name in enumerate(LANE_NAMES):
        pred = predictions.get(lane_name, {})
        pred_density = pred.get("predicted_density")
        pred_cong    = pred.get("predicted_congestion", "Unknown")
        confidence   = pred.get("confidence", "—")
        horizon      = pred.get("horizon", "—")

        color = CONGESTION_COLORS.get(pred_cong, "#94a3b8")
        icon  = "🔴" if pred_cong=="High" else ("🟡" if pred_cong=="Medium" else "🟢")

        with pred_cols[i]:
            st.markdown(f"**{lane_name} Lane**")
            if pred_density is not None:
                st.metric(
                    label=f"{icon} Predicted Congestion",
                    value=pred_cong,
                    delta=f"{pred_density}% density",
                )
            else:
                st.info("Collecting data...")
            st.caption(f"📊 {confidence}")
            st.caption(f"⏱ {horizon}")
else:
    st.info("Predictions will appear after the backend collects enough data (10+ signal events).")

st.divider()

# ---------------------------------------------------------------------------
# Vehicle type chart
# ---------------------------------------------------------------------------
st.markdown("## 🚌 Vehicle Type Distribution per Lane")
vtype_rows = []
for lane in lanes_raw:
    for vtype, count in lane.get("vehicle_types",{}).items():
        vtype_rows.append({"Lane": lane.get("name"), "Type": vtype.capitalize(), "Count": count})

if vtype_rows:
    fig_bar = px.bar(
        pd.DataFrame(vtype_rows), x="Lane", y="Count", color="Type",
        barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
        title="Vehicle Composition by Lane",
    )
    fig_bar.update_layout(height=320, margin=dict(t=40,b=20,l=10,r=10))
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Awaiting vehicle data...")

st.divider()

# ---------------------------------------------------------------------------
# Signal allocation + congestion distribution
# ---------------------------------------------------------------------------
st.markdown("## 🟢 Signal Green-Time Allocation")
ac1, ac2 = st.columns([2,3])

with ac1:
    fig_pie = go.Figure(go.Pie(
        labels=[l.get("name") for l in lanes_raw],
        values=[l.get("green_duration",0) for l in lanes_raw],
        hole=0.45,
        marker=dict(colors=["#22c55e","#3b82f6","#f59e0b","#ef4444"]),
    ))
    fig_pie.update_layout(title_text="Green Duration Share", height=300, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with ac2:
    dist = kpis.get("density_distribution", {"High":0,"Medium":0,"Low":0})
    fig_dens = go.Figure(go.Bar(
        x=list(dist.keys()), y=list(dist.values()),
        marker_color=["#ef4444","#f59e0b","#22c55e"],
        text=[f"{v}%" for v in dist.values()], textposition="outside",
    ))
    fig_dens.update_layout(title_text="Historical Congestion Distribution (%)",
                           height=300, margin=dict(t=40,b=20,l=10,r=10))
    st.plotly_chart(fig_dens, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Recent signal log
# ---------------------------------------------------------------------------
st.markdown("## 📋 Recent Signal Log — Live from SQLite Database")
if recent_log:
    log_rows = []
    for entry in recent_log:
        log_rows.append({
            "Timestamp":  str(entry.get("timestamp",""))[:19].replace("T"," "),
            "Lane":       entry.get("lane"),
            "Vehicles":   entry.get("vehicle_count"),
            "Density":    f"{entry.get('density_score',0)}%",
            "Congestion": entry.get("congestion_level"),
            "Green (s)":  entry.get("green_duration"),
            "Preempted":  "Yes" if entry.get("preempted") else "No",
            "Emergency":  "🚑 Yes" if entry.get("emergency_flag") else "—",
        })
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
else:
    st.info("No signal events yet. Start traffic_detection.py to begin streaming data.")

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
if srv_info:
    fc1, fc2, fc3 = st.columns(3)
    fc1.caption(f"🕒 Server Time (UTC): {str(srv_info.get('server_time_utc',''))[:19]}")
    fc2.caption(f"⏱ Uptime: {srv_info.get('uptime_seconds',0)}s")
    fc3.caption(f"🗄️ Database: {srv_info.get('db_path','traffic_logs.db')}")

st.caption(
    "AI-Based Smart Traffic Management and Congestion Prediction System · "
    "Akshit Rai 225811350 · B.Tech IT · MIT Bengaluru · May 2026"
)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if st.session_state.auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()