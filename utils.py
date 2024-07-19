
import numpy as np
import cv2
def draw_text(image,
              text,
              org = (50,50),
              fontFace = cv2.FONT_HERSHEY_SIMPLEX,
              fontScale = 2,
              color = (0, 0, 255) , # white color
              thickness = 2
              ):
    image = image.copy()
    image = cv2.putText(image, text, org, fontFace, fontScale, color, thickness)
    return image
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() 
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def is_red(frame, mode='bgr', threshold=0.008, tich_luy_hien_tai=None, tich_luy=3, visualize=False):
    h, w = frame.shape[:2]
    if mode == 'bgr':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif mode == 'rgb':
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask0 + mask1
    rate = np.count_nonzero(mask) / (w * h)

    if rate > threshold:
        result = [True, 0]
    else:
        if tich_luy_hien_tai is not None and tich_luy_hien_tai < tich_luy:
            result = [True, tich_luy_hien_tai + 1]
        else:
            result = [False, 99]

    if visualize:
        result.append(cv2.bitwise_and(frame, frame, mask=mask))
    else:
        result.append(None)

    return result
    