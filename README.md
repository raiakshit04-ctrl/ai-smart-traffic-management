# 🚦 AI-Based Smart Traffic Management \& Congestion Prediction System

**Developer:** Akshit Rai (225811350)  
**Programme:** B.Tech Information Technology  
**Institution:** Manipal Institute of Technology, Bengaluru  
**Year:** May 2026

\---

## Project Overview

A real-time intelligent traffic management system that uses computer vision and machine learning to dynamically optimize traffic signal timings based on live vehicle counts, congestion levels, and emergency vehicle detection.

Unlike conventional fixed-timer signals, this system observes actual traffic conditions every few seconds and allocates green time proportionally — reducing unnecessary waiting, improving intersection throughput, and providing immediate priority corridors for emergency vehicles.



\## Screenshots



\### Live Dashboard

!\[Dashboard](screenshots/dashboard.png)



\### ML Congestion Predictions (5 Minutes Ahead)

!\[Predictions](screenshots/predictions.png)



\### YOLOv8 Simulation Window

!\[Simulation](screenshots/simulation.png)



\### Emergency Vehicle Preemption — North Lane at 60s, Others Preempted to 10s

!\[Emergency](screenshots/emergency.png)



\---

## System Architecture

```
┌─────────────────────┐        HTTP POST        ┌─────────────────────┐
│  traffic\_detection  │ ──────────────────────► │    backend\_api      │
│                     │   /api/traffic/update   │                     │
│  • OpenCV simulator │                         │  • FastAPI server   │
│  • YOLOv8n model    │                         │  • Adaptive logic   │
│  • Lane detection   │                         │  • SQLite database  │
│  • Emergency flag   │                         │  • ML prediction    │
└─────────────────────┘                         └─────────┬───────────┘
                                                          │ HTTP GET
                                                          │ /api/traffic/analytics
                                                ┌─────────▼───────────┐
                                                │   app\_dashboard     │
                                                │                     │
                                                │  • Streamlit UI     │
                                                │  • Live KPI cards   │
                                                │  • Plotly charts    │
                                                │  • Prediction panel │
                                                │  • CSV export       │
                                                └─────────────────────┘
```

\---

## Tech Stack

|Layer|Technology|Purpose|
|-|-|-|
|Computer Vision|OpenCV + YOLOv8n|Vehicle detection and simulation|
|Backend API|FastAPI + Uvicorn|REST API, adaptive signal logic|
|Database|SQLite|Persistent signal event logging|
|Machine Learning|Scikit-learn (Linear Regression)|Congestion forecasting|
|Frontend|Streamlit + Plotly|Interactive real-time dashboard|
|Security|API Key Authentication|Endpoint protection|

\---

## Key Features

* **Real-time vehicle detection** across 4 lanes using YOLOv8n
* **Adaptive signal optimization**: High → 45s, Medium → 30s, Low → 15s green time
* **Emergency vehicle preemption**: Ambulance detected → 60s green, others capped at 10s
* **60-second emergency lock**: Backend holds emergency state regardless of CV engine updates
* **ML congestion prediction**: Linear Regression forecasts density 5 minutes ahead
* **SQLite persistence**: All signal events stored with full timestamp history
* **CSV export**: Full signal log downloadable from dashboard
* **API Key security**: All endpoints require `X-API-Key: traffic2026` header
* **Live dashboard**: Auto-refreshing every 2 seconds with KPI cards, gauges, charts

\---

## Project Structure

```
traffic\_project/
├── traffic\_detection.py   # CV engine — YOLOv8 + OpenCV simulation
├── backend\_api.py         # FastAPI server — adaptive logic + SQLite + ML
├── app\_dashboard.py       # Streamlit dashboard — live UI
├── requirements.txt       # All dependencies
├── README.md              # This file
└── traffic\_logs.db        # SQLite database (auto-created on first run)
```

\---

## Setup \& Installation

### 1\. Create project folder and copy files

```bash
mkdir traffic\_project
cd traffic\_project
# Copy all 4 files here: traffic\_detection.py, backend\_api.py, app\_dashboard.py, requirements.txt
```

### 2\. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3\. Run all three components simultaneously

**Terminal 1 — Start the backend first:**

```bash
cd traffic\_project
python -m uvicorn backend\_api:app --host 127.0.0.1 --port 8000
```

Wait for: `Application startup complete.`

**Terminal 2 — Start the CV engine:**

```bash
cd traffic\_project
python traffic\_detection.py
```

YOLOv8n weights (\~6MB) download automatically on first run.

**Terminal 3 — Start the dashboard:**

```bash
cd traffic\_project
python -m streamlit run app\_dashboard.py
```

Browser opens at `http://localhost:8501`

\---

## API Reference

All endpoints require header: `X-API-Key: traffic2026`

|Method|Endpoint|Description|
|-|-|-|
|POST|`/api/traffic/update`|Receive lane data, apply adaptive logic, save to SQLite|
|POST|`/api/traffic/trigger\_emergency`|Set 60s emergency lock for a lane|
|POST|`/api/traffic/clear\_emergency`|Release emergency lock immediately|
|GET|`/api/traffic/analytics`|Full intersection state + ML predictions + KPIs|
|GET|`/api/traffic/export/csv`|Download all signal events as CSV|
|GET|`/api/traffic/health`|Server health + SQLite record count|

\---

## Adaptive Signal Logic

```
IF emergency vehicle detected in lane:
    That lane  → 60 seconds GREEN (hard lock, 60s duration)
    All others → 10 seconds GREEN (preempted)
ELSE:
    High congestion   (>9 vehicles)  → 45 seconds GREEN
    Medium congestion (5-9 vehicles) → 30 seconds GREEN
    Low congestion    (0-4 vehicles) → 15 seconds GREEN
```

\---

## ML Prediction Model

* **Algorithm:** Linear Regression (scikit-learn)
* **Training data:** Historical vehicle counts and density scores from SQLite
* **Feature:** Current vehicle count per lane
* **Target:** Predicted density score \~5 minutes ahead
* **Retraining:** Automatic every 50 API updates
* **Output:** Predicted congestion level (Low / Medium / High) with confidence info

\---

## Demonstration Steps

1. Start all three terminals as described above
2. Open browser at `http://localhost:8501`
3. Observe live updating KPI cards, gauge charts, and signal log
4. In sidebar: select **North** from Target Lane dropdown
5. Click **🚑 Trigger Emergency Vehicle**
6. Observe: North lane → 60s green, all others → 10s preempted
7. Click **✅ Clear Emergency** — all lanes return to adaptive normal timings
8. Click **⬇️ Download Signal Log CSV** to export the full database history

\---

## Future Scope

* Integration with live CCTV feeds via RTSP streams
* Fine-tuned YOLOv8 model with ambulance/fire-truck classes
* LSTM-based time-series prediction for higher accuracy forecasting
* Cloud deployment on AWS/GCP with PostgreSQL
* Mobile alert system for emergency vehicle operators
* Multi-intersection coordination and green wave optimization

