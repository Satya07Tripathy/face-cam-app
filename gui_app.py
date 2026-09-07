import os
# Quiets down MediaPipe/TensorFlow Lite's internal logging (informational
# noise about the model, not actual errors) so it doesn't clutter output.
os.environ.setdefault("GLOG_minloglevel", "2")

import sys
import threading
import time
import tkinter as tk
from collections import deque

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk

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
# Same app as combined_app.py (face/age/emotion analyze mode + finger-
# drawing mode), but with a real Tkinter GUI -- buttons instead of keyboard
# shortcuts. The model loading, background inference worker, and per-frame
# detection/drawing logic are unchanged from combined_app.py; what's new is
# how the video gets on screen and how mode/color/eraser get toggled.
#
# The key idea that's different from a plain OpenCV `while True` loop: a GUI
# toolkit like Tkinter owns its own event loop (root.mainloop()), which is
# what actually watches for and responds to button clicks and window events.
# If we blocked that loop ourselves with `while True: cap.read(); ...`, like
# the keyboard-driven version does, button clicks would never get noticed.
# Instead, we hand Tkinter a function (update_frame) and ask it to call that
# function again and again via root.after(...) -- each call grabs one camera
# frame, processes it, and reschedules itself, so Tkinter's own loop stays
# free to handle clicks in between calls.
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

# --- Age/emotion background worker (unchanged from combined_app.py) ---
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

# --- Hand-drawing state ---
INDEX_TIP = 8
THUMB_TIP = 4
PINCH_THRESHOLD = 40
DETECT_WIDTH = 320

COLOR_OPTIONS = [
    ((0, 0, 255), "Red"),
    ((0, 255, 0), "Green"),
    ((255, 0, 0), "Blue"),
    ((0, 255, 255), "Yellow"),
    ((255, 255, 255), "White"),
]
draw_color = COLOR_OPTIONS[0][0]
color_name = COLOR_OPTIONS[0][1]
THICKNESS = 5
ERASER_THICKNESS = 40
erasing = False
canvas = None
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
    sys.exit(1)

# ===========================================================================
# GUI layout: video on the left, a column of controls on the right.
# ===========================================================================
root = tk.Tk()
root.title("Face Cam App")

video_label = tk.Label(root)
video_label.pack(side=tk.LEFT)

controls = tk.Frame(root, padx=12, pady=12)
controls.pack(side=tk.LEFT, fill=tk.Y)

status_var = tk.StringVar()
tk.Label(controls, textvariable=status_var, justify=tk.LEFT, font=("Segoe UI", 10, "bold")).pack(pady=(0, 10), anchor="w")


def update_status():
    tool = "Eraser" if erasing else color_name
    status_var.set(f"Mode: {mode.title()}\nTool: {tool}")


def toggle_mode():
    global mode, hand_landmarker, prev_point
    if mode == "analyze":
        mode = "draw"
        print("Loading hand tracking model...")
        hand_landmarker = HandLandmarker.create_from_options(hand_options)
        mode_button.config(text="Switch to Analyze Mode")
    else:
        mode = "analyze"
        hand_landmarker.close()
        hand_landmarker = None
        mode_button.config(text="Switch to Draw Mode")
    prev_point = None
    update_status()


mode_button = tk.Button(controls, text="Switch to Draw Mode", width=22, command=toggle_mode)
mode_button.pack(pady=(0, 15))

tk.Label(controls, text="Color", anchor="w").pack(fill=tk.X)
color_frame = tk.Frame(controls)
color_frame.pack(pady=(0, 10))

color_buttons = {}


def set_color(color, name):
    global draw_color, color_name, erasing
    draw_color = color
    color_name = name
    erasing = False
    for c, btn in color_buttons.items():
        btn.config(relief=tk.SUNKEN if c == color else tk.RAISED)
    eraser_button.config(relief=tk.RAISED)
    update_status()


for color, name in COLOR_OPTIONS:
    # Tkinter wants colors as "#rrggbb", but our colors are stored BGR
    # (OpenCV's order) -- swap them back to RGB for the button itself.
    hex_color = "#%02x%02x%02x" % (color[2], color[1], color[0])
    btn = tk.Button(
        color_frame, bg=hex_color, activebackground=hex_color, width=3,
        relief=tk.SUNKEN if color == draw_color else tk.RAISED,
        command=lambda c=color, n=name: set_color(c, n),
    )
    btn.pack(side=tk.LEFT, padx=2)
    color_buttons[color] = btn


def toggle_eraser():
    global erasing
    erasing = not erasing
    eraser_button.config(relief=tk.SUNKEN if erasing else tk.RAISED)
    if erasing:
        for btn in color_buttons.values():
            btn.config(relief=tk.RAISED)
    update_status()


eraser_button = tk.Button(controls, text="Eraser", width=22, command=toggle_eraser)
eraser_button.pack(pady=(0, 10))


def clear_canvas():
    if canvas is not None:
        canvas[:] = 0


tk.Button(controls, text="Clear Drawing", width=22, command=clear_canvas).pack(pady=(0, 10))


def on_close():
    cap.release()
    if hand_landmarker is not None:
        hand_landmarker.close()
    root.destroy()


tk.Button(controls, text="Quit", width=22, command=on_close).pack(pady=(20, 0))
root.protocol("WM_DELETE_WINDOW", on_close)

update_status()

# ===========================================================================
# The per-frame processing below is identical in substance to
# combined_app.py's main loop -- it's just called once per Tkinter
# root.after() tick instead of once per `while True` iteration.
# ===========================================================================


def update_frame():
    global canvas, frame_count, frames_since_face, pending_job, latest_label, prev_point

    ret, frame = cap.read()
    if not ret:
        root.after(15, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)

    frame_count += 1

    if mode == "analyze":
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

    combined = cv2.add(frame, canvas)

    # Tkinter's PhotoImage wants RGB, and needs a PIL Image in between to
    # convert from a raw numpy array.
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(combined_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    # Tkinter doesn't keep its own reference to the image data -- without
    # stashing it somewhere (here, as an attribute on the label itself),
    # Python would garbage-collect it immediately and the video would go
    # blank or flicker.
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(15, update_frame)


root.after(0, update_frame)
root.mainloop()
