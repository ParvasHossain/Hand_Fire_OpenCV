import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os

# 1. Load GIF frames
def load_gif_frames(gif_path):
    if not os.path.exists(gif_path):
        print(f"Error: {gif_path} not found!")
        return []
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGBA')
            opencv_frame = np.array(frame)
            opencv_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_RGBA2BGRA)
            frames.append(opencv_frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

# 2. Rotate Image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

# 3. Transparent Overlay Function
def overlay_transparent(background, overlay, x, y, size=150, angle=0):
    if size <= 0:
        return
    
    overlay = cv2.resize(overlay, (size, size))
    if angle != 0:
        overlay = rotate_image(overlay, angle)

    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, _ = overlay.shape

    x1, y1 = max(x - ov_w // 2, 0), max(y - ov_h // 2, 0)
    x2, y2 = min(x + ov_w // 2, bg_w), min(y + ov_h // 2, bg_h)

    ov_x1 = max(0, (ov_w // 2) - x)
    ov_y1 = max(0, (ov_h // 2) - y)
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1 or ov_x2 <= ov_x1 or ov_y2 <= ov_y1:
        return

    overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    background_crop = background[y1:y2, x1:x2]

    alpha = overlay_crop[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    blended = (1.0 - alpha) * background_crop + alpha * overlay_crop[:, :, :3]
    background[y1:y2, x1:x2] = blended

# 4. Open Palm Check
def is_open_palm(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcps = [5, 9, 13, 17]
    extended_fingers = sum(1 for tip, mcp in zip(finger_tips, finger_mcps) if landmarks[tip].y < landmarks[mcp].y)
    return extended_fingers >= 3

# 5. MediaPipe Tasks Setup
if not os.path.exists('hand_landmarker.task'):
    print("\n[ERROR] 'hand_landmarker.task' file is missing!")
    exit()

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

fire_frames = load_gif_frames('fire.gif')
if not fire_frames:
    print("\n[ERROR] Could not load fire.gif!")
    exit()

total_fire_frames = len(fire_frames)
cap = cv2.VideoCapture(0)

window_name = "Hand Fire Controller"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fire_index = 0
is_fullscreen = False

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            if is_open_palm(hand_landmarks):
                lm0 = hand_landmarks[0]  # Wrist
                lm9 = hand_landmarks[9]  # Middle knuckle
                
                x0, y0 = lm0.x * w, lm0.y * h
                x9, y9 = lm9.x * w, lm9.y * h
                
                radians = math.atan2(y9 - y0, x9 - x0)
                angle = math.degrees(radians) - 90
                
                # --- POSITION OFFSET ADJUSTMENT ---
                dx = x9 - x0
                dy = y9 - y0
                cx = int(x9 + dx * 0.8)  # Moved higher up from wrist/knuckle
                cy = int(y9 + dy * 0.8)
                
                hand_dist = np.sqrt((lm9.x - lm0.x)**2 + (lm9.y - lm0.y)**2)
                fire_size = int(hand_dist * w * 2.2)
                fire_size = max(80, fire_size)

                current_fire = fire_frames[fire_index]
                flipped_fire = cv2.flip(current_fire, 0)
                overlay_transparent(frame, flipped_fire, cx, cy, size=fire_size, angle=-angle)

    fire_index = (fire_index + 1) % total_fire_frames

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()