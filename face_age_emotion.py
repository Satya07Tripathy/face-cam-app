import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from insightface.utils import ensure_available
from insightface.model_zoo import get_model as get_insightface_model
from insightface.app.common import Face
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# ---------------------------------------------------------------------------
# Step 1: Face detection -- same Haar cascade as face_detect.py. It just
# finds WHERE the faces are; age/gender/emotion are separate models that run
# on the cropped face region afterwards.
# ---------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------------------------
# Step 2: Age + gender model ("genderage", from the insightface project).
#
# insightface ships its models bundled into "packs". We only need the tiny
# genderage.onnx file, but it lives inside the "buffalo_l" pack alongside
# face-recognition/landmark models we don't use here. The first time this
# script runs, ensure_available() downloads that pack (~326 MB, one time)
# from insightface's official GitHub releases and unzips it into
# ~/.insightface/models/buffalo_l. Every run after that just reads the
# cached files -- no re-download.
#
# The model itself takes in a cropped, aligned face image and outputs 3
# numbers: a score for "female", a score for "male", and a normalized age
# value in [0, 1] (multiply by 100 to get years). insightface's Attribute
# class (which this model loads as) does the crop/align/scale math for us --
# we just hand it the full frame plus a bounding box.
# ---------------------------------------------------------------------------
print("Loading age/gender model (first run downloads ~326MB, please be patient)...")
_model_dir = ensure_available("models", "buffalo_l")
genderage_model = get_insightface_model(
    os.path.join(_model_dir, "genderage.onnx"),
    providers=["CPUExecutionProvider"],
)

# ---------------------------------------------------------------------------
# Step 3: Emotion model (hsemotion-onnx).
#
# This is a small CNN trained to classify a face crop into one of 8 emotions
# (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise).
# Its weights download once (cached under ~/.hsemotion) the first time you
# construct HSEmotionRecognizer. enet_b0_8_best_vgaf is the library's own
# default, trained on real-world video rather than acted movie clips, which
# tends to generalize better to a live webcam than the "afew" variant.
# ---------------------------------------------------------------------------
print("Loading emotion model...")
emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")

# ---------------------------------------------------------------------------
# Background inference worker.
#
# Even running the models only every few frames, each call still takes long
# enough (tens to hundreds of milliseconds on a CPU) that doing it *inside*
# the main loop blocks video capture and display while it thinks -- that's
# what shows up as stutter/lag. The fix is to hand each face crop off to a
# separate thread: it keeps computing in the background, and the main loop
# just displays whatever the latest finished answer is, never waiting on it.
#
# `job_lock` protects the two variables shared between the main thread and
# the worker thread (`pending_job`, `latest_label`) so they're never read
# and written at the same instant from both threads.
# ---------------------------------------------------------------------------
job_lock = threading.Lock()
pending_job = None    # (frame_copy, (x1, y1, x2, y2)) waiting to be processed
latest_label = None   # most recent "Gender, Age, Emotion" string to display

# Both age and emotion are noisy per-frame even for the same, unmoving face
# (slightly different angle/lighting/compression each frame). Averaging the
# last several readings smooths that jitter out. This does NOT fix
# systematic model bias -- age can still be consistently off, and subtle
# expressions can still be misread -- it only stops the *displayed* value
# from swinging wildly between one frame and the next.
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

        # --- Age & gender ---
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        gender, age = genderage_model.get(job_frame, Face(bbox=bbox))
        age_history.append(age)
        smoothed_age = round(sum(age_history) / len(age_history))
        gender_label = "Male" if gender == 1 else "Female"

        # --- Emotion ---
        # hsemotion expects a plain face crop in RGB order. OpenCV reads
        # frames in BGR by default, so swap channel order first. We average
        # the raw probability scores (not just the winning label) across
        # frames, then pick the winner of the averaged scores -- that way a
        # single noisy frame can't flip the displayed emotion by itself.
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

# Only submit a new job every few frames -- the worker thread is running in
# parallel, but a fresh job every single frame would still mean copying and
# queuing more work than it can realistically keep up with.
RUN_MODELS_EVERY = 3

frame_count = 0
frames_since_face = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally so the feed behaves like a mirror (otherwise your
    # right hand shows up on the frame's right side, not the left). This
    # has to happen before detection/analysis so the box coordinates and
    # the face crops line up with what's actually being displayed.
    frame = cv2.flip(frame, 1)

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        frames_since_face += 1
        # No one's been in frame for a while -- drop stale results instead
        # of showing an old label if a different person sits down later.
        if frames_since_face > 30:
            with job_lock:
                latest_label = None
            age_history.clear()
            emotion_score_history.clear()
    else:
        frames_since_face = 0
        # Only analyze the largest detected face, in case Haar picks up
        # more than one candidate box (e.g. a false positive in the
        # background). Keeps things simple and fast for a single-user app.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        if frame_count % RUN_MODELS_EVERY == 0:
            # The age/emotion models were trained on crops with some margin
            # around the face (forehead, chin, a bit of background), not a
            # razor-tight box like Haar gives us. Padding the box before
            # handing it off gives them the context they expect.
            pad_w, pad_h = int(w * 0.2), int(h * 0.2)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)

            # Copy the frame before handing it to the worker thread: we're
            # about to draw a rectangle/text on `frame` below, and without
            # a copy the worker could end up analyzing a face crop with a
            # green box baked into it.
            with job_lock:
                pending_job = (frame.copy(), (x1, y1, x2, y2))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        with job_lock:
            label_to_show = latest_label
        if label_to_show is not None:
            cv2.putText(
                frame, label_to_show, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    cv2.imshow("Face + Age + Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
