"""
traffic_detection.py
====================
Core Computer Vision Engine — AI-Based Smart Traffic Management System
Developed by: Akshit Rai (225811350) — B.Tech IT, MIT Bengaluru, May 2026

Upgrades in this version:
  - Posts to /api/traffic/update as before
  - Emergency vehicle simulation with red ambulance box
  - Blended YOLO + simulator ground truth counts
"""

import cv2
import numpy as np
import random
import time
import threading
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from ultralytics import YOLO

FRAME_W, FRAME_H = 1280, 720
NUM_LANES        = 4
LANE_NAMES       = ["North", "South", "East", "West"]
FPS              = 10
VEHICLE_CLASSES  = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
EMERGENCY_COLOR  = (0, 0, 220)
EMERGENCY_LABEL  = "ambulance"
DENSITY_LOW_MAX  = 4
DENSITY_MED_MAX  = 9
BACKEND_URL      = "http://127.0.0.1:8000/api/traffic/update"


@dataclass
class LaneState:
    name:             str
    vehicle_count:    int   = 0
    density_score:    float = 0.0
    congestion_level: str   = "Low"
    emergency_flag:   bool  = False
    vehicle_types:    Dict[str, int] = field(default_factory=dict)


class SimVehicle:
    COLORS = {
        "car":        (180, 180, 180),
        "motorcycle": (100, 200, 100),
        "bus":        (200, 140,  50),
        "truck":      (140,  80, 200),
    }

    def __init__(self, lane_index: int, frame_w: int, frame_h: int,
                 is_emergency: bool = False):
        self.lane_index   = lane_index
        self.is_emergency = is_emergency
        self.vertical     = lane_index < 2

        if is_emergency:
            self.vtype = EMERGENCY_LABEL
            self.color = EMERGENCY_COLOR
            self.w, self.h = 44, 22
        else:
            self.vtype = random.choice(list(VEHICLE_CLASSES.values()))
            self.color = self.COLORS.get(self.vtype, (200, 200, 200))
            sizes = {"bus": (42, 20), "truck": (48, 22), "motorcycle": (20, 12), "car": (30, 16)}
            self.w, self.h = sizes.get(self.vtype, (30, 16))

        if self.vertical:
            strip_w = frame_w // 4
            strip_x = lane_index * strip_w
            self.x  = random.randint(strip_x + 10, strip_x + strip_w - self.w - 10)
            self.y  = 0 if lane_index == 0 else frame_h - self.h
            self.dx, self.dy = 0, (3 if lane_index == 0 else -3)
        else:
            strip_h = frame_h // 4
            strip_y = (lane_index - 2) * strip_h + frame_h // 2
            self.x  = 0 if lane_index == 2 else frame_w - self.w
            self.y  = random.randint(strip_y + 5, strip_y + strip_h - self.h - 5)
            self.dx, self.dy = (3 if lane_index == 2 else -3), 0

        self.speed = random.uniform(1.5, 3.5)

    def move(self):
        self.x += int(self.dx * self.speed)
        self.y += int(self.dy * self.speed)

    def is_offscreen(self, frame_w, frame_h):
        return self.x > frame_w + 60 or self.x < -60 or self.y > frame_h + 60 or self.y < -60

    def draw(self, frame):
        x1, y1, x2, y2 = self.x, self.y, self.x + self.w, self.y + self.h
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
        if self.is_emergency:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(frame, (mx, y1 + 3), (mx, y2 - 3), (255, 255, 255), 2)
            cv2.line(frame, (x1 + 3, my), (x2 - 3, my), (255, 255, 255), 2)


class TrafficAnalyzer:
    def __init__(self, video_path=None, model_path="yolov8n.pt",
                 post_to_backend=True, emergency_lane=None):
        self.video_path      = video_path
        self.post_to_backend = post_to_backend
        self.emergency_lane  = emergency_lane

        print("[TrafficAnalyzer] Loading YOLOv8 model ...")
        self.model = YOLO(model_path)
        print("[TrafficAnalyzer] Model loaded.")

        self.lane_states = [LaneState(name=LANE_NAMES[i]) for i in range(NUM_LANES)]
        self._vehicles   = [[] for _ in range(NUM_LANES)]
        self._lock       = threading.Lock()
        self._running    = False
        self._frame_idx  = 0
        self._populate_initial_vehicles()

    def _populate_initial_vehicles(self):
        for i in range(NUM_LANES):
            for _ in range(random.randint(2, 8)):
                self._vehicles[i].append(SimVehicle(i, FRAME_W, FRAME_H))

    def _spawn_vehicle(self, lane_idx, force_emergency=False):
        is_emg = force_emergency or (
            self.emergency_lane == lane_idx and random.random() < 0.08
        )
        self._vehicles[lane_idx].append(SimVehicle(lane_idx, FRAME_W, FRAME_H, is_emergency=is_emg))

    def _build_synthetic_frame(self):
        frame = np.full((FRAME_H, FRAME_W, 3), 45, dtype=np.uint8)
        for i in range(1, 4):
            cv2.line(frame, (i * FRAME_W // 4, 0), (i * FRAME_W // 4, FRAME_H), (80, 80, 80), 1)
            cv2.line(frame, (0, i * FRAME_H // 4), (FRAME_W, i * FRAME_H // 4), (80, 80, 80), 1)
        cv2.rectangle(frame, (FRAME_W//4, FRAME_H//4), (3*FRAME_W//4, 3*FRAME_H//4), (60,60,60), -1)

        for lane_idx in range(NUM_LANES):
            alive = []
            for v in self._vehicles[lane_idx]:
                v.move()
                if not v.is_offscreen(FRAME_W, FRAME_H):
                    v.draw(frame)
                    alive.append(v)
            self._vehicles[lane_idx] = alive
            if len(self._vehicles[lane_idx]) < random.randint(3, 10):
                if random.random() < 0.4:
                    self._spawn_vehicle(lane_idx)

        positions = [(FRAME_W//8, 20), (FRAME_W//8, FRAME_H-10),
                     (10, FRAME_H//2-30), (FRAME_W-70, FRAME_H//2-30)]
        for i, (lx, ly) in enumerate(positions):
            cv2.putText(frame, LANE_NAMES[i], (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        return frame

    def _compute_density_score(self, count, max_v=15):
        return min(round((count / max_v) * 100, 1), 100.0)

    def _classify_congestion(self, count):
        if count <= DENSITY_LOW_MAX:   return "Low"
        elif count <= DENSITY_MED_MAX: return "Medium"
        return "High"

    def _check_emergency(self, lane_idx):
        return any(v.is_emergency for v in self._vehicles[lane_idx])

    def _detect_vehicles_in_lane(self, frame, lane_idx):
        h, w = frame.shape[:2]
        crops = [(0,0,w//4,h),(w//4,0,w//2,h),(w//2,0,3*w//4,h),(3*w//4,0,w,h)]
        x1,y1,x2,y2 = crops[lane_idx]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0, {}

        results    = self.model(crop, verbose=False, conf=0.25, iou=0.45)
        type_counts: Dict[str, int] = {}
        total = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    vname = VEHICLE_CLASSES[cls_id]
                    type_counts[vname] = type_counts.get(vname, 0) + 1
                    total += 1

        sim_count = len(self._vehicles[lane_idx])
        blended   = max(total, int(sim_count * 0.85))
        if blended > total:
            for v in self._vehicles[lane_idx]:
                if v.vtype != EMERGENCY_LABEL:
                    type_counts[v.vtype] = type_counts.get(v.vtype, 0) + 1
        return blended, type_counts

    def _post_to_backend(self):
        payload = {"lanes": [
            {
                "name":             ls.name,
                "vehicle_count":    ls.vehicle_count,
                "density_score":    ls.density_score,
                "congestion_level": ls.congestion_level,
                "emergency_flag":   ls.emergency_flag,
                "vehicle_types":    ls.vehicle_types,
            } for ls in self.lane_states
        ]}
        try:
            requests.post(BACKEND_URL, json=payload, timeout=2, headers={"X-API-Key": "traffic2026"})
        except Exception:
            pass

    def inject_emergency(self, lane_idx):
        with self._lock:
            self.emergency_lane = lane_idx
            self._vehicles[lane_idx].append(SimVehicle(lane_idx, FRAME_W, FRAME_H, is_emergency=True))

    def _annotate_frame(self, frame):
        positions = [(10,30),(FRAME_W//4+10,30),(FRAME_W//2+10,30),(3*FRAME_W//4+10,30)]
        for i, ls in enumerate(self.lane_states):
            ox, oy = positions[i]
            color  = (0,200,0) if ls.congestion_level=="Low" else \
                     (0,165,255) if ls.congestion_level=="Medium" else (0,0,255)
            cv2.putText(frame, f"{ls.name}: {ls.vehicle_count}v [{ls.congestion_level}]",
                        (ox,oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if ls.emergency_flag:
                cv2.putText(frame, "EMERGENCY!", (ox, oy+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame #{self._frame_idx}", (FRAME_W-130, FRAME_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        return frame

    def run(self, display=False, max_frames=None):
        self._running = True
        cap = None
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                cap = None

        print("[TrafficAnalyzer] Starting processing loop ...")
        delay = 1.0 / FPS

        while self._running:
            t0 = time.time()
            if cap:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            else:
                frame = self._build_synthetic_frame()

            with self._lock:
                for lane_idx in range(NUM_LANES):
                    count, types = self._detect_vehicles_in_lane(frame, lane_idx)
                    emg          = self._check_emergency(lane_idx)
                    ls = self.lane_states[lane_idx]
                    ls.vehicle_count    = count
                    ls.vehicle_types    = types
                    ls.density_score    = self._compute_density_score(count)
                    ls.congestion_level = self._classify_congestion(count)
                    ls.emergency_flag   = emg

            annotated = self._annotate_frame(frame.copy())
            if display:
                cv2.imshow("AI Traffic Management — Simulation", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.post_to_backend:
                self._post_to_backend()

            self._frame_idx += 1
            if max_frames and self._frame_idx >= max_frames:
                break
            time.sleep(max(0.0, delay - (time.time() - t0)))

        if cap: cap.release()
        if display: cv2.destroyAllWindows()

    def stop(self):
        self._running = False

    def get_lane_states(self):
        with self._lock:
            return list(self.lane_states)


if __name__ == "__main__":
    analyzer = TrafficAnalyzer(video_path=None, model_path="yolov8n.pt", post_to_backend=True)
    print("Press Ctrl+C to stop.\n")
    try:
        analyzer.run(display=True)
    except KeyboardInterrupt:
        analyzer.stop()
        print("Stopped.")