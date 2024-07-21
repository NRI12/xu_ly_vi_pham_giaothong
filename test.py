import cv2
import numpy as np

def is_red(frame, threshold=0.008):
    # Chuyển đổi frame sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Xác định ngưỡng màu đỏ trong không gian HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # Tạo mask cho màu đỏ
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    # Tính tỷ lệ pixel màu đỏ
    red_ratio = np.sum(mask) / (mask.size)
    return red_ratio > threshold

# Đường dện đến video
video_path = r'C:\Users\pc\Desktop\xu_ly_vi_pham_giaothong\data\video\La-Khê-Hà_Đông.mp4'
# Vùng ROI cho đèn
x1, y1, x2, y2 = 650, 5, 700, 50

# Tạo đối tượng VideoCapture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Cắt ROI
    roi = frame[y1:y2, x1:x2]
    
    # Kiểm tra màu đỏ trong ROI
    light_status = "Red Light" if is_red(roi) else "Green Light"
    print(f"{light_status} detected!")
    light_color = (0, 0, 255) if "Red" in light_status else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), light_color, 2)
    
    # Vẽ trạng thái đèn lên góc trên bên trái của frame
    cv2.putText(frame, light_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)

    # Hiển thị frame
    cv2.imshow('Frame', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
