import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load GIF frames
def load_gif_frames(gif_path):
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

# 2. Transparent Overlay Function
def overlay_transparent(background, overlay, x, y, size=150):
    if size <= 0:
        return
    
    overlay = cv2.resize(overlay, (size, size))
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

# 3. Open Palm Detection (Triggers when hand is open)
def is_open_palm(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcps = [5, 9, 13, 17]
    
    extended_fingers = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        # Tip y-coordinate is lower than MCP joint y-coordinate when hand is raised & open
        if landmarks[tip].y < landmarks[mcp].y:
            extended_fingers += 1
            
    return extended_fingers >= 3

# 4. MediaPipe Tasks API Setup (Python 3.14)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Load Fire GIF
try:
    fire_frames = load_gif_frames('fire.gif')
    total_fire_frames = len(fire_frames)
except Exception as e:
    print("Error: 'fire.gif' file not found.")
    exit()

cap = cv2.VideoCapture(0)
fire_index = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Trigger fire when hand is OPEN
            if is_open_palm(hand_landmarks):
                lm9 = hand_landmarks[9]
                lm0 = hand_landmarks[0]
                
                cx, cy = int(lm9.x * w), int(lm9.y * h)
                
                # --- FIRE SIZE ADJUSTMENT ---
                # Increased multiplier from 1.8 to 3.2 for a much larger fire effect
                hand_dist = np.sqrt((lm9.x - lm0.x)**2 + (lm9.y - lm0.y)**2)
                fire_size = int(hand_dist * w * 3.2)
                fire_size = max(100, fire_size)  # Set a larger minimum size (100px)

                current_fire = fire_frames[fire_index]
                overlay_transparent(frame, current_fire, cx, cy, size=fire_size)

    fire_index = (fire_index + 1) % total_fire_frames

    cv2.imshow("Hand Fire Controller", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()