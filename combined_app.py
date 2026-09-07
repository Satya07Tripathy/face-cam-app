import os
# Quiets down MediaPipe/TensorFlow Lite's internal logging (informational
# noise about the model, not actual errors) so it doesn't clutter output.
os.environ.setdefault("GLOG_minloglevel", "2")

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

# ===========================================================================
# This combines the three separate scripts (face_detect.py, face_age_emotion
# .py, hand_draw.py) into one app with two MODES you switch between with the
# 'm' key:
#   "analyze" -- face box + age/gender/emotion (from face_age_emotion.py)
#   "draw"    -- finger-drawing (from hand_draw.py)
# Only one mode's models actually run detection per frame -- running the
# face+age+emotion pipeline AND the hand tracker at the same time would
# double up on exactly the CPU cost that was already causing lag in each one
# separately, for no real benefit (you're either looking at your own face
# stats or drawing, not usually both at once).
#
# The hand-tracking model is also only *loaded* while draw mode is active
# (created when you switch in, closed when you switch out) rather than kept
# around the whole time. Each of these ML libraries (onnxruntime for
# age/emotion, MediaPipe's own TensorFlow Lite runtime for hand tracking)
# spins up its own internal thread pool for parallelism, and MediaPipe's in
# particular has no exposed setting to limit it. Leaving it loaded even
# while unused meant it was still sitting there competing for CPU time with
# the face-detection loop and the age/emotion worker thread -- which is
# almost certainly why analyze mode got laggier once draw mode was added to
# the same app, even though analyze mode's own code didn't change.
# ===========================================================================

# --- Face detection ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Age/gender model (see face_age_emotion.py for the full explanation of
# why this needs a 326MB one-time download) ---
print("Loading age/gender model...")
_model_dir = ensure_available("models", "buffalo_l")
_genderage_path = os.path.join(_model_dir, "genderage.onnx")
# This model is tiny (1.3MB), so it gains nothing from onnxruntime's default
# of spreading work across every CPU core -- but doing so still means it
# competes for those cores with everything else running. Pinning it to a
# single thread costs us nothing here and removes one more source of
# contention. (insightface's get_model() helper doesn't expose this option,
# so we build the session ourselves and hand it to their Attribute class,
# which is the same class get_model() would have returned anyway.)
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

# --- Hand landmark model options (the model itself is created/destroyed on
# demand when switching in and out of draw mode -- see the 'm' key handling
# below) ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(BASE_DIR, "models", "hand_landmarker.task")
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)
hand_landmarker = None

# ---------------------------------------------------------------------------
# Age/emotion background worker -- unchanged from face_age_emotion.py.
# Inference is handed off to a separate thread so the main loop (camera
# capture + display, and in this combined app, hand tracking too) never
# blocks waiting on it.
# ---------------------------------------------------------------------------
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

RUN_MODELS_EVERY = 3  # how often (in frames) to submit a fresh age/emotion job

# --- Hand-drawing state (see hand_draw.py for the full explanation) ---
INDEX_TIP = 8
THUMB_TIP = 4
PINCH_THRESHOLD = 40
DETECT_WIDTH = 320

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
canvas = None       # persistent drawing surface -- survives mode switches,
                     # only 'c' clears it
prev_point = None

mode = "analyze"  # or "draw"
frame_count = 0
frames_since_face = 0

CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)

    frame_count += 1

    if mode == "analyze":
        # Keep this fresh every frame we're not drawing, so that whenever
        # we switch back into "draw" mode, the first stroke doesn't jump
        # in from a stale old fingertip position.
        prev_point = None

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

    else:  # mode == "draw"
        detect_height = int(DETECT_WIDTH * h / w)
        small_frame = cv2.resize(frame, (DETECT_WIDTH, detect_height))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            index_tip = landmarks[INDEX_TIP]
            thumb_tip = landmarks[THUMB_TIP]
            index_px = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            pinch_dist = np.hypot(
                index_px[0] - thumb_px[0], index_px[1] - thumb_px[1]
            )
            pen_down = pinch_dist > PINCH_THRESHOLD

            if not pen_down:
                dot_color = (0, 0, 255)
            elif erasing:
                dot_color = (128, 128, 128)
            else:
                dot_color = draw_color
            cv2.circle(frame, index_px, 10, dot_color, -1)

            if pen_down:
                stroke_color = (0, 0, 0) if erasing else draw_color
                stroke_thickness = ERASER_THICKNESS if erasing else THICKNESS
                if prev_point is not None:
                    cv2.line(canvas, prev_point, index_px, stroke_color, stroke_thickness)
                prev_point = index_px
            else:
                prev_point = None
        else:
            prev_point = None

    # Canvas overlay applies in both modes, so a drawing you made stays
    # visible even while you're back in analyze mode looking at your
    # age/emotion readout.
    combined = cv2.add(frame, canvas)

    if mode == "analyze":
        hud = "Mode: ANALYZE (face/age/emotion) | m: switch to draw | c: clear drawing | q: quit"
    else:
        tool = "ERASER" if erasing else color_name
        hud = f"Mode: DRAW (tool: {tool}) | m: switch to analyze | 1-5: color | e: eraser | c: clear | q: quit"
    cv2.putText(
        combined, hud, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )

    if mode == "draw":
        cv2.rectangle(combined, (10, 35), (40, 65), draw_color, -1)
        cv2.rectangle(combined, (10, 35), (40, 65), (255, 255, 255), 1)

    cv2.imshow("Face + Age + Emotion + Draw", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        if mode == "analyze":
            mode = "draw"
            print("Loading hand tracking model...")
            hand_landmarker = HandLandmarker.create_from_options(hand_options)
        else:
            mode = "analyze"
            hand_landmarker.close()
            hand_landmarker = None
        prev_point = None
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('e'):
        erasing = not erasing
    elif key in COLORS:
        draw_color, color_name = COLORS[key]
        erasing = False

if hand_landmarker is not None:
    hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()
