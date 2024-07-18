from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import cv2
import asyncio
from ultralytics import YOLO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

VIDEO_BASE_PATH = './data/video'
MODEL_PATH = './model/vehicle.pt'
active_websocket = None

# Load YOLO model
model = YOLO(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/camera", response_class=HTMLResponse)
async def get_camera(request: Request):
    video_files = os.listdir(VIDEO_BASE_PATH)
    return templates.TemplateResponse("camera.html", {"request": request, "video_files": video_files, "enumerate": enumerate})

@app.websocket("/ws/{video_name}")
async def websocket_endpoint(websocket: WebSocket, video_name: str):
    global active_websocket
    if active_websocket:
        await active_websocket.close()
    
    active_websocket = websocket
    await websocket.accept()
    video_path = os.path.join(VIDEO_BASE_PATH, video_name)
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to reduce data size
            frame = cv2.resize(frame, (720, 480))  # Adjust the resolution as needed

            # Perform YOLO prediction
            results = model.predict(source=frame, imgsz=320, conf=0.45, iou=0.45)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = box.cls[0]
                confidence = box.conf[0]
                text = f"{model.names[int(label)]}: {confidence:.2f}"
                
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green box
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)  # Green text

            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(1 / 60)  # 60 FPS
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        active_websocket = None
        await websocket.close()