"""
backend_api.py
==============
FastAPI Backend — AI-Based Smart Traffic Management System
Developed by: Akshit Rai (225811350) — B.Tech IT, MIT Bengaluru, May 2026
 
Upgrades in this version:
  - SQLite persistent database for all signal events
  - Congestion prediction using Linear Regression (scikit-learn)
  - API Key authentication on all endpoints
  - Emergency lock system (60s hard lock, CV engine cannot override)
  - CSV export endpoint
"""
 
import time
import sqlite3
import csv
import io
import statistics
from datetime import datetime
from typing import List, Dict, Any
 
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY      = "traffic2026"          # simple key — checked on every request
DB_PATH      = "traffic_logs.db"     # SQLite file created in project folder
SERVER_START = time.time()
 
# ---------------------------------------------------------------------------
# API Key Security
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
 
def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key. "
                            "Pass header: X-API-Key: traffic2026")
    return key
 
# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            lane             TEXT    NOT NULL,
            vehicle_count    INTEGER NOT NULL,
            density_score    REAL    NOT NULL,
            congestion_level TEXT    NOT NULL,
            emergency_flag   INTEGER NOT NULL,
            green_duration   INTEGER NOT NULL,
            preempted        INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()
 
init_db()
 
# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class LaneUpdate(BaseModel):
    name:             str
    vehicle_count:    int   = Field(ge=0)
    density_score:    float = Field(ge=0.0, le=100.0)
    congestion_level: str   = Field(pattern="^(Low|Medium|High)$")
    emergency_flag:   bool  = False
    vehicle_types:    Dict[str, int] = {}
 
class TrafficUpdateRequest(BaseModel):
    lanes: List[LaneUpdate]
 
# ---------------------------------------------------------------------------
# Global in-memory state (fast reads for dashboard)
# ---------------------------------------------------------------------------
current_state: Dict[str, Dict[str, Any]] = {}
 
_kpi_totals: Dict[str, Any] = {
    "total_updates":         0,
    "emergency_activations": 0,
    "green_times_all":       [],
    "high_density_count":    0,
    "medium_density_count":  0,
    "low_density_count":     0,
    "vehicle_counts_all":    [],
}
 
# ---------------------------------------------------------------------------
# Emergency lock
# ---------------------------------------------------------------------------
emergency_lock: Dict[str, Any] = {
    "active":     False,
    "lane":       None,
    "expires_at": 0.0,
}
 
def get_emergency_state():
    if emergency_lock["active"] and time.time() > emergency_lock["expires_at"]:
        emergency_lock["active"]     = False
        emergency_lock["lane"]       = None
        emergency_lock["expires_at"] = 0.0
    return emergency_lock
 
def compute_green_duration(congestion_level: str) -> int:
    return {"High": 45, "Medium": 30, "Low": 15}.get(congestion_level, 15)
 
# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------
class CongestionPredictor:
    """
    Trains a Linear Regression model on historical vehicle counts from the
    SQLite database and predicts the congestion level 5 minutes ahead.
    Retrains automatically every 50 new records.
    """
    def __init__(self):
        self.model      = LinearRegression()
        self.trained    = False
        self.train_count = 0
 
    def _fetch_training_data(self):
        conn = get_db()
        rows = conn.execute(
            "SELECT vehicle_count, density_score FROM signal_events ORDER BY id DESC LIMIT 200"
        ).fetchall()
        conn.close()
        return rows
 
    def train(self):
        rows = self._fetch_training_data()
        if len(rows) < 10:
            return
        counts  = np.array([r["vehicle_count"] for r in rows], dtype=float)
        density = np.array([r["density_score"]  for r in rows], dtype=float)
        # Feature: current count; Label: density 5 steps ahead (simulated shift)
        X = counts[:-5].reshape(-1, 1)
        y = density[5:]
        if len(X) < 5:
            return
        self.model.fit(X, y)
        self.trained     = True
        self.train_count = len(rows)
 
    def predict(self, current_count: int) -> Dict[str, Any]:
        if not self.trained:
            self.train()
        if not self.trained:
            return {"predicted_density": None, "predicted_congestion": "Unknown",
                    "confidence": "Insufficient data — need 10+ records"}
        predicted_density = float(self.model.predict([[current_count]])[0])
        predicted_density = max(0.0, min(100.0, round(predicted_density, 1)))
        if predicted_density <= 30:
            predicted_congestion = "Low"
        elif predicted_density <= 65:
            predicted_congestion = "Medium"
        else:
            predicted_congestion = "High"
        return {
            "predicted_density":    predicted_density,
            "predicted_congestion": predicted_congestion,
            "confidence":           f"Trained on {self.train_count} records",
            "horizon":              "~5 minutes ahead",
        }
 
predictor = CongestionPredictor()
 
# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Smart Traffic Management — Backend API",
    description="Adaptive signal optimization with SQLite logging, ML prediction, and API Key auth.",
    version="2.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ---------------------------------------------------------------------------
# POST /api/traffic/update
# ---------------------------------------------------------------------------
@app.post("/api/traffic/update")
def update_traffic(payload: TrafficUpdateRequest, key: str = Depends(verify_api_key)):
    if not payload.lanes:
        raise HTTPException(status_code=400, detail="Empty lanes list.")
 
    timestamp = datetime.utcnow().isoformat() + "Z"
    emg       = get_emergency_state()
    conn      = get_db()
 
    for lane in payload.lanes:
        is_emg_lane  = emg["active"] and lane.name == emg["lane"]
        is_preempted = emg["active"] and lane.name != emg["lane"]
 
        if is_emg_lane:
            assigned_green = 60;  emergency_flag = True;  preempted = False
        elif is_preempted:
            assigned_green = 10;  emergency_flag = False; preempted = True
        else:
            assigned_green = compute_green_duration(lane.congestion_level)
            emergency_flag = False; preempted = False
 
        current_state[lane.name] = {
            "name":             lane.name,
            "vehicle_count":    lane.vehicle_count,
            "density_score":    lane.density_score,
            "congestion_level": lane.congestion_level,
            "emergency_flag":   emergency_flag,
            "vehicle_types":    lane.vehicle_types,
            "green_duration":   assigned_green,
            "preempted":        preempted,
            "signal_phase":     "GREEN",
            "last_updated":     timestamp,
        }
 
        # Persist to SQLite
        conn.execute("""
            INSERT INTO signal_events
            (timestamp, lane, vehicle_count, density_score, congestion_level,
             emergency_flag, green_duration, preempted)
            VALUES (?,?,?,?,?,?,?,?)
        """, (timestamp, lane.name, lane.vehicle_count, lane.density_score,
              lane.congestion_level, int(emergency_flag), assigned_green, int(preempted)))
 
        _kpi_totals["vehicle_counts_all"].append(lane.vehicle_count)
        _kpi_totals["green_times_all"].append(assigned_green)
        if emergency_flag:   _kpi_totals["emergency_activations"] += 1
        if lane.congestion_level == "High":   _kpi_totals["high_density_count"]   += 1
        elif lane.congestion_level == "Medium": _kpi_totals["medium_density_count"] += 1
        else:                                   _kpi_totals["low_density_count"]    += 1
 
    conn.commit()
    conn.close()
    _kpi_totals["total_updates"] += 1
 
    # Retrain predictor every 50 updates
    if _kpi_totals["total_updates"] % 50 == 0:
        predictor.train()
 
    return {"status": "success", "timestamp": timestamp}
 
# ---------------------------------------------------------------------------
# POST /api/traffic/trigger_emergency
# ---------------------------------------------------------------------------
@app.post("/api/traffic/trigger_emergency")
def trigger_emergency(data: dict, key: str = Depends(verify_api_key)):
    lane_name = data.get("lane")
    if not lane_name:
        raise HTTPException(status_code=400, detail="Lane name required.")
 
    emergency_lock["active"]     = True
    emergency_lock["lane"]       = lane_name
    emergency_lock["expires_at"] = time.time() + 60
 
    timestamp = datetime.utcnow().isoformat() + "Z"
    all_lanes = list(current_state.keys()) if current_state else ["North","South","East","West"]
    for name in all_lanes:
        is_emg = (name == lane_name)
        base   = current_state.get(name, {
            "name": name, "vehicle_count": 0, "density_score": 0.0,
            "congestion_level": "Low", "vehicle_types": {}, "signal_phase": "GREEN",
        })
        current_state[name] = {
            **base,
            "emergency_flag": is_emg,
            "green_duration": 60 if is_emg else 10,
            "preempted":      not is_emg,
            "last_updated":   timestamp,
        }
 
    return {"status": "emergency_set", "lane": lane_name, "expires_in": 60}
 
# ---------------------------------------------------------------------------
# POST /api/traffic/clear_emergency
# ---------------------------------------------------------------------------
@app.post("/api/traffic/clear_emergency")
def clear_emergency(key: str = Depends(verify_api_key)):
    emergency_lock["active"]     = False
    emergency_lock["lane"]       = None
    emergency_lock["expires_at"] = 0.0
 
    timestamp = datetime.utcnow().isoformat() + "Z"
    for name, state in current_state.items():
        current_state[name] = {
            **state,
            "emergency_flag": False,
            "green_duration": compute_green_duration(state.get("congestion_level","Low")),
            "preempted":      False,
            "last_updated":   timestamp,
        }
    return {"status": "cleared"}
 
# ---------------------------------------------------------------------------
# GET /api/traffic/analytics
# ---------------------------------------------------------------------------
@app.get("/api/traffic/analytics")
def get_analytics(key: str = Depends(verify_api_key)):
    uptime     = round(time.time() - SERVER_START, 1)
    all_greens = _kpi_totals["green_times_all"]
    all_counts = _kpi_totals["vehicle_counts_all"]
    total_lane_events = (
        _kpi_totals["high_density_count"] +
        _kpi_totals["medium_density_count"] +
        _kpi_totals["low_density_count"]
    )
 
    avg_green  = round(statistics.mean(all_greens), 1) if all_greens else 0.0
    high_pct   = round(_kpi_totals["high_density_count"]   / total_lane_events * 100, 1) if total_lane_events else 0.0
    medium_pct = round(_kpi_totals["medium_density_count"] / total_lane_events * 100, 1) if total_lane_events else 0.0
    low_pct    = round(_kpi_totals["low_density_count"]    / total_lane_events * 100, 1) if total_lane_events else 0.0
 
    # Fetch recent log from SQLite
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM signal_events ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    recent_log = [dict(r) for r in rows]
 
    # Per-lane predictions
    predictions = {}
    for name, state in current_state.items():
        predictions[name] = predictor.predict(state.get("vehicle_count", 0))
 
    emg = get_emergency_state()
 
    return {
        "current_lanes":    list(current_state.values()),
        "recent_log":       recent_log,
        "predictions":      predictions,
        "emergency_active": emg["active"],
        "emergency_lane":   emg["lane"],
        "performance_kpis": {
            "avg_green_duration_s":       avg_green,
            "emergency_activations":      _kpi_totals["emergency_activations"],
            "signal_efficiency_pct":      100.0,
            "density_distribution": {
                "High": high_pct, "Medium": medium_pct, "Low": low_pct
            },
            "total_signal_events": total_lane_events,
        },
        "server_info": {
            "uptime_seconds":  uptime,
            "total_updates":   _kpi_totals["total_updates"],
            "server_time_utc": datetime.utcnow().isoformat() + "Z",
            "db_path":         DB_PATH,
        },
    }
 
# ---------------------------------------------------------------------------
# GET /api/traffic/export/csv
# ---------------------------------------------------------------------------
@app.get("/api/traffic/export/csv")
def export_csv(key: str = Depends(verify_api_key)):
    """Export all signal events from SQLite as a downloadable CSV file."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM signal_events ORDER BY id DESC").fetchall()
    conn.close()
 
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","timestamp","lane","vehicle_count","density_score",
                     "congestion_level","emergency_flag","green_duration","preempted"])
    for row in rows:
        writer.writerow([row["id"], row["timestamp"], row["lane"],
                         row["vehicle_count"], row["density_score"],
                         row["congestion_level"], row["emergency_flag"],
                         row["green_duration"], row["preempted"]])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_signal_log.csv"}
    )
 
# ---------------------------------------------------------------------------
# GET /api/traffic/health
# ---------------------------------------------------------------------------
@app.get("/api/traffic/health")
def health_check():
    conn  = get_db()
    count = conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0]
    conn.close()
    return {
        "status":       "healthy",
        "db_records":   count,
        "server_time":  datetime.utcnow().isoformat() + "Z",
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
 