from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import cv2
import sqlite3
from ultralytics import YOLO
from urllib.parse import parse_qs
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from utils import *
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/violations", StaticFiles(directory="violations"), name="violations")
templates = Jinja2Templates(directory="templates")

VIDEO_BASE_PATH = './data/video'
MODEL_PATH_VEHICLE = './model/vehicle.pt'
MODEL_PATH_HELMET = './model/best_helmet_end.pt'
VIOLATION_FOLDER = './violations'
os.makedirs(VIOLATION_FOLDER, exist_ok=True)
active_websocket = None

# Deep Sort trackers
track_vehicle = DeepSort(max_age=30)
track_helmet = DeepSort(max_age=30)
track_light = DeepSort(max_age=30)

# Load YOLO models
model_vehicle = YOLO(MODEL_PATH_VEHICLE).to('cuda')
model_helmet = YOLO(MODEL_PATH_HELMET).to('cuda')

# Initialize SQLite database
DATABASE_PATH = './violations.db'
conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    image_path TEXT,
                    video_path TEXT,
                    category TEXT
                  )''')
conn.commit()

executor = ThreadPoolExecutor()

def save_violation_to_db(timestamp, image_path, video_path, category):
    cursor.execute('INSERT INTO violations (timestamp, image_path, video_path, category) VALUES (?, ?, ?, ?)',
                   (timestamp, image_path, video_path, category))
    conn.commit()

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/tables", response_class=HTMLResponse)
async def get_tables(request: Request):
    return templates.TemplateResponse("tables.html", {"request": request})

@app.get("/camera", response_class=HTMLResponse)
async def get_camera(request: Request):
    video_files = os.listdir(VIDEO_BASE_PATH)
    return templates.TemplateResponse("camera.html", {"request": request, "video_files": video_files, "enumerate": enumerate})

def get_all_violations():
    cursor.execute("SELECT * FROM violations")
    violations = cursor.fetchall()
    print(violations)  # In ra để kiểm tra xem có dữ liệu hay không
    return violations


@app.get("/api/violations")
async def read_violations():
    violations = get_all_violations()
    return JSONResponse(content={"data": violations})
@app.get("/api/statistics")
async def get_statistics():
    cursor.execute('SELECT COUNT(*) FROM violations WHERE category = "Lane Violation"')
    lane_violations = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM violations WHERE category = "No Helmet"')
    no_helmet_violations = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM violations WHERE category = "Light Violation"')
    light_violations = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM violations')
    total_violations = cursor.fetchone()[0]

    return JSONResponse(content={
        "lane_violations": lane_violations,
        "no_helmet_violations": no_helmet_violations,
        "light_violations": light_violations,
        "total_violations": total_violations
    })
@app.get("/api/violations_daily")
async def get_violations_daily():
    cursor.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM violations
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """)
    violations_daily = cursor.fetchall()
    data = [{"date": row[0], "count": row[1]} for row in violations_daily]
    return JSONResponse(content={"data": data})
@app.websocket("/ws/{video_name}")
async def websocket_endpoint(websocket: WebSocket, video_name: str, roi_x1: int = 100, roi_y1: int = 350, roi_x2: int = 1150, roi_y2: int = 750):
    global active_websocket
    if active_websocket:
        await active_websocket.close()

    active_websocket = websocket
    await websocket.accept()
    video_path = os.path.join(VIDEO_BASE_PATH, video_name)

    # Parse query parameters
    query_params = parse_qs(websocket.url.query)
    detection_type = query_params.get('detectionType', ['none'])[0].lower()
    left_labels = json.loads(query_params.get('leftLabels', ['[]'])[0])
    right_labels = json.loads(query_params.get('rightLabels', ['[]'])[0])

    print(f"Connected to {video_name}")
    print(f"Detection Type: {detection_type}")
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop to the region of interest
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            if detection_type == 'vehicle' or detection_type == 'helmet':
                detect_vehicle = []
                results_vehicle = model_vehicle.predict(source=roi, imgsz=320, conf=0.3, iou=0.4)[0]
                for box in results_vehicle.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    detect_vehicle.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

                current_track_vehicle = track_vehicle.update_tracks(detect_vehicle, frame=roi)

                for i, track in enumerate(current_track_vehicle):
                    if not (track.is_confirmed() and track.det_conf):
                        continue

                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = list(map(int, ltrb))
                    track_id = track.track_id
                    label = track.det_class
                    confidence = track.det_conf

                    if confidence > 0.65:
                        text = f"{model_vehicle.names[int(label)]}, id: {track_id}, conf: {round(confidence, 2)}"
                        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(roi, text, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if detection_type == 'helmet':
                    for i, track in enumerate(current_track_vehicle):
                        if not (track.is_confirmed() and track.det_conf):
                            continue

                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = list(map(int, ltrb))
                        crop_img = roi[y1:y2, x1:x2]
                        if crop_img.size != 0:
                            results_helmet = model_helmet.predict(source=crop_img, imgsz=320, conf=0.45, iou=0.45)[0]
                            for helmet_box in results_helmet.boxes:
                                hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0].tolist())
                                hlabel = helmet_box.cls[0]
                                hconfidence = helmet_box.conf[0]
                                htext = f"{model_helmet.names[int(hlabel)]}: {hconfidence:.2f}"
                                cv2.rectangle(roi, (x1 + hx1, y1 + hy1), (x1 + hx2, y1 + hy2), (255, 0, 0) if model_helmet.names[int(hlabel)] == "Without Helmet" else (0, 255, 0), 1)
                                cv2.putText(roi, htext, (x1 + hx1, y1 + hy1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255) if model_helmet.names[int(hlabel)] == "Without Helmet" else (255, 0, 0), 1, cv2.LINE_AA)

                                if hconfidence > 0.65 and model_helmet.names[int(hlabel)] == "Without Helmet":
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    violation_image_path = os.path.join(VIOLATION_FOLDER, f"violation_{timestamp}.jpg")
                                    cv2.imwrite(violation_image_path, crop_img)

                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    executor.submit(save_violation_to_db, current_time, violation_image_path, video_path, "No Helmet")

            elif detection_type == 'lane':
                left_labels = json.loads(query_params.get('leftLabels', ['[]'])[0])
                right_labels = json.loads(query_params.get('rightLabels', ['[]'])[0])
                # Define midpoint
                midpoint_x = roi_x1 + (roi_x2 - roi_x1) // 2  # Midpoint to divide lanes

                # Draw the middle line
                cv2.line(frame, (midpoint_x, roi_y1), (midpoint_x, roi_y2), (0, 0, 255), 2)

                # Process vehicle detections
                results_vehicle = model_vehicle.predict(source=roi, imgsz=320, conf=0.3, iou=0.4)[0]
                for box in results_vehicle.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    vehicle_center_x = (x1 + x2) // 2
                    vehicle_label = model_vehicle.names[int(box.cls.item())]

                    if vehicle_label in left_labels:
                        lane = "Left"
                    elif vehicle_label in right_labels:
                        lane = "Right"
                    else:
                        continue  # Ignore unknown labels

                    # Determine violation based on lane and position
                    if (lane == "Left" and vehicle_center_x > midpoint_x) or (lane == "Right" and vehicle_center_x < midpoint_x):
                        alert_message = f"Violation!!! Lane of vehicles"
                        violation_detected = True
                        color = (0, 0, 255)  # Red for violation
                    else:
                        continue  # Only show violations

                    # Highlight the vehicle and label it
                    cv2.rectangle(frame, (roi_x1 + x1, roi_y1 + y1), (roi_x1 + x2, roi_y1 + y2), color, 2)
                    cv2.putText(frame, f"{vehicle_label} - {alert_message}", (roi_x1 + x1, roi_y1 + y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)

                    if violation_detected:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        violation_image_path = os.path.join(VIOLATION_FOLDER, f"violation_{timestamp}.jpg")
                        frame_with_bounding_box = frame.copy()
                        cv2.rectangle(frame_with_bounding_box, (roi_x1 + x1, roi_y1 + y1), (roi_x1 + x2, roi_y1 + y2), color, 2)
                        cv2.putText(frame_with_bounding_box, f"{vehicle_label} - {alert_message}", (roi_x1 + x1, roi_y1 + y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
                        cv2.imwrite(violation_image_path, frame_with_bounding_box)

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        executor.submit(save_violation_to_db, current_time, violation_image_path, video_path, "Lane Violation")

            elif detection_type == 'light':
                tich_luy = 0
                state_all = {}
                ok = {}

                light_roi_x1 = int(query_params.get('lightRoiX1', [0])[0])
                light_roi_y1 = int(query_params.get('lightRoiY1', [0])[0])
                light_roi_x2 = int(query_params.get('lightRoiX2', [0])[0])
                light_roi_y2 = int(query_params.get('lightRoiY2', [0])[0])
                y_line = int(query_params.get('yLine', [450])[0])
                y_line_buffer = 30  # Buffer zone around y_line

                light_frame = frame[light_roi_y1:light_roi_y2, light_roi_x1:light_roi_x2, :]
                red, tich_luy, _ = is_red(light_frame, tich_luy_hien_tai=tich_luy)
                text = 'RED' if red else 'GREEN'
                light_color = (0, 0, 255) if red else (0, 255, 0)

                # Draw light ROI bounding box and y_line
                cv2.putText(frame, f"Light: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)

                cv2.rectangle(frame, (light_roi_x1, light_roi_y1), (light_roi_x2, light_roi_y2), (0, 255, 255), 2)
                cv2.line(frame, (0, y_line), (frame.shape[1], y_line), (0, 255, 255), 2)

                result = model_vehicle.predict(roi, conf=0.35, verbose=False)
                if len(result):
                    result = result[0]
                    names = result.names
                    detect = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        detect.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

                    tracks = track_light.update_tracks(detect, frame=roi)
                    for i, track in enumerate(tracks):
                        if track.is_confirmed() and track.det_conf:
                            ltrb = track.to_ltrb()
                            x1, y1, x2, y2 = list(map(int, ltrb))
                            yc = roi_y1 + (y1 + (y2 - y1) // 2)

                            track_id = track.track_id

                            is_ok = ok.get(track_id, None)
                            label = None  # Khởi tạo nhãn là None

                            if is_ok:
                                label = "k vuot" if ok[track_id] == 2 else 'vuot'
                            else:
                                state = state_all.get(track_id)
                                if state is None:
                                    if yc > (y_line + y_line_buffer):
                                        ok[track_id] = 2
                                        label = "k vuot"
                                    elif yc < (y_line - y_line_buffer) and red:
                                        ok[track_id] = 1
                                        label = "vuot"
                                    else:
                                        state_all[track_id] = red
                                else:
                                    if yc > (y_line + y_line_buffer):
                                        ok[track_id] = 1 if state_all[track_id] else 2
                                        label = "k vuot" if ok[track_id] == 2 else 'vuot'
                                    elif yc < (y_line - y_line_buffer) and red:
                                        ok[track_id] = 1
                                        label = "vuot"
                                    else:
                                        state_all[track_id] = red

                            if label:  # Chỉ hiển thị nhãn khi có giá trị "vượt" hoặc "k vượt"
                                color = (0, 0, 255) if label == 'vuot' else (0, 255, 0)
                                cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)
                                cv2.rectangle(roi, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color, -1)
                                cv2.putText(roi, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)

            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        active_websocket = None
        await websocket.close()
