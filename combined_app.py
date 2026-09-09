import os
# Quiets down MediaPipe/TensorFlow Lite's internal logging (informational
# noise about the model, not actual errors) so it doesn't clutter output.
os.environ.setdefault("GLOG_minloglevel", "2")

import ctypes
import sys
import threading
import time
from collections import deque

# When this script is bundled into a standalone .exe (via PyInstaller),
# running the exe extracts bundled data files to a temporary folder at
# startup and sets sys._MEIPASS to that folder's path -- it's NOT the same
# as the current working directory (which is wherever the user double-
# clicked from). When running as a normal .py script, there is no
# sys._MEIPASS, so we fall back to the folder this file lives in. Either
# way, BASE_DIR ends up pointing at wherever "models/hand_landmarker.task"
# actually is.
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
import onnxruntime as ort

from insightface.utils import ensure_available
from insightface.model_zoo.attribute import Attribute
from insightface.app.common import Face
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

WINDOW_NAME = "Face Cam App"

# ===========================================================================
# Three modes, cycled with 'm':
#   "analyze" -- face box + age/gender/emotion
#   "draw"    -- finger-drawing overlaid on top of the live camera feed
#   "notepad" -- finger-drawing on a blank page; camera feed is hidden, but
#                the camera is still used behind the scenes to track your hand
#
# Only "analyze", or one of the two drawing modes, actually runs its models
# each frame -- never both families at once. See the loading/unloading of
# hand_landmarker in the 'm' key handling below for why.
# ===========================================================================

# --- Face detection ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Age/gender model ---
print("Loading age/gender model...")
_model_dir = ensure_available("models", "buffalo_l")
_genderage_path = os.path.join(_model_dir, "genderage.onnx")
_sess_options = ort.SessionOptions()
_sess_options.intra_op_num_threads = 1
_sess_options.inter_op_num_threads = 1
_genderage_session = ort.InferenceSession(
    _genderage_path, sess_options=_sess_options, providers=["CPUExecutionProvider"]
)
genderage_model = Attribute(model_file=_genderage_path, session=_genderage_session)

# --- Emotion model ---
print("Loading emotion model...")
emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")

# --- Hand landmark model options (created/destroyed on mode switch) ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(BASE_DIR, "models", "hand_landmarker.task")
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)
hand_landmarker = None

# --- Age/emotion background worker (unchanged) ---
job_lock = threading.Lock()
pending_job = None
latest_label = None
age_history = deque(maxlen=10)
emotion_score_history = deque(maxlen=8)


def inference_worker():
    global pending_job, latest_label
    while True:
        with job_lock:
            job = pending_job
            pending_job = None
        if job is None:
            time.sleep(0.01)
            continue

        job_frame, (x1, y1, x2, y2) = job
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        gender, age = genderage_model.get(job_frame, Face(bbox=bbox))
        age_history.append(age)
        smoothed_age = round(sum(age_history) / len(age_history))
        gender_label = "Male" if gender == 1 else "Female"

        emotion_label = "?"
        face_crop = job_frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            _, scores = emotion_model.predict_emotions(face_crop_rgb, logits=False)
            emotion_score_history.append(scores)
            avg_scores = np.mean(emotion_score_history, axis=0)
            emotion_label = emotion_model.idx_to_class[int(np.argmax(avg_scores))]

        with job_lock:
            latest_label = f"{gender_label}, {smoothed_age}y, {emotion_label}"


threading.Thread(target=inference_worker, daemon=True).start()

RUN_MODELS_EVERY = 3

# ---------------------------------------------------------------------------
# Hand-drawing tuning -- two changes here directly target "laggy and not
# accurate":
#
# 1. EMA (exponential moving average) smoothing on the fingertip position.
#    A hand-landmark model's output is never perfectly stable frame to
#    frame -- tiny lighting/angle changes shift the reported point by a few
#    pixels even when your finger hasn't moved, which shows up as a shaky,
#    jittery line. Blending each new reading with the *previous smoothed*
#    point (instead of drawing straight to the raw reading) averages that
#    noise out. SMOOTHING is how much weight a brand-new reading gets: 1.0
#    would mean no smoothing at all (raw/jittery); 0.4 is a reasonable
#    middle ground between "smooth" and "still feels responsive."
#
# 2. Hysteresis on the pinch gesture. A single distance threshold for
#    "pinched vs not" means that whenever your fingers happen to hover near
#    that exact distance, tiny noise flips the state back and forth many
#    times a second -- strokes get chopped into fragments, which reads as
#    "inaccurate." Two thresholds instead (fingers must open PAST
#    PINCH_START to begin drawing, then close PAST the smaller PINCH_STOP
#    to stop) means the state can't flicker right at one boundary value.
# ---------------------------------------------------------------------------
INDEX_TIP = 8
THUMB_TIP = 4
PINCH_START = 45
PINCH_STOP = 28
SMOOTHING = 0.4
DETECT_WIDTH = 256  # smaller than before (320) -- the hand model's own
                     # internal input is small anyway, so this loses very
                     # little accuracy while cutting real per-frame cost.

COLORS = {
    ord('1'): ((0, 0, 255), "Red"),
    ord('2'): ((0, 255, 0), "Green"),
    ord('3'): ((255, 0, 0), "Blue"),
    ord('4'): ((0, 255, 255), "Yellow"),
    ord('5'): ((255, 255, 255), "White"),
}
draw_color = (0, 0, 255)
color_name = "Red"
THICKNESS = 5
ERASER_THICKNESS = 40
erasing = False

camera_canvas = None   # "draw" mode's surface -- sized to the camera frame,
                        # overlaid on the live video
notepad_canvas = None  # "notepad" mode's surface -- a separate blank page,
                        # sized to your actual screen, camera feed hidden
prev_point = None      # last pen-down position (for drawing a connecting line)
smoothed_point = None  # EMA-smoothed fingertip position
pen_down = False       # current pinch state (persists across frames for hysteresis)

MODES = ["analyze", "draw", "notepad"]
mode = "analyze"
frame_count = 0
frames_since_face = 0

# ---------------------------------------------------------------------------
# Screen resolution, via the Windows API (this app is Windows-only already).
# The notepad page is sized to this rather than to the camera resolution --
# unlike "draw" mode, notepad mode never displays the camera image itself,
# so there's no reason to limit it to the camera's resolution. That's also
# what makes a crisp fullscreen notepad possible.
# ---------------------------------------------------------------------------
_user32 = ctypes.windll.user32
SCREEN_W = _user32.GetSystemMetrics(0)
SCREEN_H = _user32.GetSystemMetrics(1)


def make_notepad_page():
    # A plain, slightly warm off-white rather than pure white or black --
    # reads more like a paper notepad, less like a blank video signal.
    return np.full((SCREEN_H, SCREEN_W, 3), (238, 245, 248), dtype=np.uint8)


notepad_canvas = make_notepad_page()

CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
if not cap.isOpened():
    print("Could not open the webcam")
    sys.exit(1)

# WINDOW_NORMAL (resizable) is required for the fullscreen toggle below to
# work at all -- the default AUTOSIZE window ignores it.
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
is_fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if camera_canvas is None:
        camera_canvas = np.zeros_like(frame)

    frame_count += 1

    if mode == "analyze":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            frames_since_face += 1
            if frames_since_face > 30:
                with job_lock:
                    latest_label = None
                age_history.clear()
                emotion_score_history.clear()
        else:
            frames_since_face = 0
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

            if frame_count % RUN_MODELS_EVERY == 0:
                pad_w, pad_h = int(fw * 0.2), int(fh * 0.2)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + fw + pad_w)
                y2 = min(h, y + fh + pad_h)
                with job_lock:
                    pending_job = (frame.copy(), (x1, y1, x2, y2))

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            with job_lock:
                label_to_show = latest_label
            if label_to_show is not None:
                cv2.putText(
                    frame, label_to_show, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        # "draw" mode's on-camera canvas still shows through here too, so a
        # sketch you made doesn't vanish just because you checked your
        # age/emotion readout in between.
        display = cv2.add(frame, camera_canvas)

    else:  # mode == "draw" or "notepad"
        detect_height = int(DETECT_WIDTH * h / w)
        small_frame = cv2.resize(frame, (DETECT_WIDTH, detect_height))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if mode == "draw":
            target_canvas, target_w, target_h = camera_canvas, w, h
        else:
            target_canvas, target_w, target_h = notepad_canvas, SCREEN_W, SCREEN_H

        index_px = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            index_tip = landmarks[INDEX_TIP]
            thumb_tip = landmarks[THUMB_TIP]

            # Pinch distance is measured in the CAMERA FRAME's own scale
            # (w, h), regardless of which mode/canvas we're drawing onto --
            # otherwise the same physical pinch would register as a tiny
            # gap on "draw" mode's 640-wide canvas but a huge one on
            # notepad mode's 1920-wide screen, making the gesture threshold
            # meaningless in one mode or the other.
            index_ref = (index_tip.x * w, index_tip.y * h)
            thumb_ref = (thumb_tip.x * w, thumb_tip.y * h)
            pinch_dist = np.hypot(index_ref[0] - thumb_ref[0], index_ref[1] - thumb_ref[1])

            if pen_down:
                if pinch_dist < PINCH_STOP:
                    pen_down = False
            else:
                if pinch_dist > PINCH_START:
                    pen_down = True

            # The actual drawing/cursor position, mapped to whichever
            # canvas is active right now.
            raw_point = (index_tip.x * target_w, index_tip.y * target_h)
            if smoothed_point is None:
                smoothed_point = raw_point
            else:
                smoothed_point = (
                    SMOOTHING * raw_point[0] + (1 - SMOOTHING) * smoothed_point[0],
                    SMOOTHING * raw_point[1] + (1 - SMOOTHING) * smoothed_point[1],
                )
            index_px = (int(smoothed_point[0]), int(smoothed_point[1]))

            if pen_down:
                stroke_color = (0, 0, 0) if erasing else draw_color
                stroke_thickness = ERASER_THICKNESS if erasing else THICKNESS
                if prev_point is not None:
                    cv2.line(target_canvas, prev_point, index_px, stroke_color, stroke_thickness)
                prev_point = index_px
            else:
                prev_point = None
        else:
            prev_point = None
            smoothed_point = None
            pen_down = False

        if mode == "draw":
            display = cv2.add(frame, camera_canvas)
        else:
            display = notepad_canvas.copy()

        if index_px is not None:
            if not pen_down:
                dot_color = (0, 0, 255)
            elif erasing:
                dot_color = (128, 128, 128)
            else:
                dot_color = draw_color
            cv2.circle(display, index_px, 10, dot_color, -1)

    # HUD text needs to be dark on notepad's light page, but stays light
    # everywhere else (video is generally dark/busy enough for white text).
    hud_color = (40, 40, 40) if mode == "notepad" else (255, 255, 255)

    if mode == "analyze":
        hud = "ANALYZE  |  m: next mode (draw)  |  f: fullscreen  |  c: clear drawing  |  q: quit"
    else:
        tool = "ERASER" if erasing else color_name
        next_mode = "notepad" if mode == "draw" else "analyze"
        hud = f"{mode.upper()} (tool: {tool})  |  m: next mode ({next_mode})  |  f: fullscreen  |  1-5: color  |  e: eraser  |  c: clear  |  q: quit"
    cv2.putText(
        display, hud, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2
    )

    if mode in ("draw", "notepad"):
        cv2.rectangle(display, (10, 35), (40, 65), draw_color, -1)
        cv2.rectangle(display, (10, 35), (40, 65), hud_color, 1)

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        )
    elif key == ord('m'):
        idx = MODES.index(mode)
        next_mode = MODES[(idx + 1) % len(MODES)]
        needs_hand = next_mode in ("draw", "notepad")
        had_hand = mode in ("draw", "notepad")
        if needs_hand and not had_hand:
            print("Loading hand tracking model...")
            hand_landmarker = HandLandmarker.create_from_options(hand_options)
        elif not needs_hand and had_hand:
            hand_landmarker.close()
            hand_landmarker = None
        mode = next_mode
        prev_point = None
        smoothed_point = None
        pen_down = False
    elif key == ord('c'):
        if mode == "notepad":
            notepad_canvas = make_notepad_page()
        else:
            camera_canvas[:] = 0
    elif key == ord('e'):
        erasing = not erasing
    elif key in COLORS:
        draw_color, color_name = COLORS[key]
        erasing = False

if hand_landmarker is not None:
    hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()
