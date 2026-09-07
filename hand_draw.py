import os
# Quiets down MediaPipe/TensorFlow Lite's internal logging (informational
# noise about the model, not actual errors) so it doesn't clutter output.
os.environ.setdefault("GLOG_minloglevel", "2")

import time

import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# MediaPipe's hand model looks at one image and outputs 21 (x, y, z) points
# per hand it finds -- one for the wrist, plus several along each finger,
# always in the same fixed order regardless of hand size or pose. We only
# need two of those 21:
#   8 = tip of the index finger  -> this becomes our "pen"
#   4 = tip of the thumb         -> used to detect a pinch ("pen up/down")
# ---------------------------------------------------------------------------
INDEX_TIP = 8
THUMB_TIP = 4

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,  # we feed it one frame at a time,
                                            # in order, and get an answer back
                                            # immediately (as opposed to
                                            # LIVE_STREAM mode, which answers
                                            # via a callback on another
                                            # thread -- more setup than we
                                            # need here).
    num_hands=1,
)

# Pinch distance below this many pixels counts as "pinched" (pen up). This
# is a raw pixel distance, so it depends on how close your hand is to the
# camera -- raise or lower it if pinch detection feels wrong for you.
PINCH_THRESHOLD = 40

# ---------------------------------------------------------------------------
# Performance: the neural net doesn't need to see full camera resolution to
# find a hand -- more pixels just means more math for the same answer. We
# ask the camera for a modest capture size (most of the per-frame cost --
# color conversion, drawing, display -- scales with this), and additionally
# hand the *detector* an even smaller resized copy. Landmark coordinates
# come back normalized to [0, 1] (a fraction of width/height) regardless of
# what resolution we fed in, so we can multiply them by the FULL-res frame's
# dimensions and get correct pixel positions either way -- no rescaling math
# needed, we just point two different image sizes at the same [0, 1] space.
# ---------------------------------------------------------------------------
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
DETECT_WIDTH = 320

# Available draw colors, selected with number keys (BGR order, since that's
# what OpenCV uses).
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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
if not cap.isOpened():
    print("Could not open the webcam")
    exit()

canvas = None       # persistent drawing surface; created once we know the
                     # frame size, and never cleared except by pressing 'c'
prev_point = None    # last pen-down position, so we can draw a connecting
                      # line to the current position instead of just a dot

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Build the small copy the detector actually looks at.
        detect_height = int(DETECT_WIDTH * h / w)
        small_frame = cv2.resize(frame, (DETECT_WIDTH, detect_height))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires a timestamp that strictly increases between
        # calls, so it can reason about motion between frames. Wall-clock
        # milliseconds is the simplest thing that satisfies that.
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]  # only tracking one hand

            # Landmark coordinates are normalized to [0, 1], so multiply by
            # the FULL-res frame's size (not small_frame's) to draw at full
            # quality regardless of what size the detector saw.
            index_tip = landmarks[INDEX_TIP]
            thumb_tip = landmarks[THUMB_TIP]
            index_px = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            pinch_dist = np.hypot(
                index_px[0] - thumb_px[0], index_px[1] - thumb_px[1]
            )
            pen_down = pinch_dist > PINCH_THRESHOLD

            # Visual feedback: a dot on your fingertip, colored to match
            # whatever's about to be drawn (grey while erasing), and red
            # while pinched/paused, so the state is obvious.
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
                # Pen lifted -- forget the last point so that when drawing
                # resumes, it doesn't draw a stray line jumping from the
                # old position to the new one.
                prev_point = None
        else:
            prev_point = None

        # Overlay the drawing on top of the live video. The canvas starts
        # all black (0, 0, 0), and adding black to a pixel leaves it
        # unchanged, so this only visibly affects pixels we've drawn on.
        # Erasing works the same way in reverse: drawing black back onto a
        # previously-colored spot returns it to "unchanged", i.e. erased.
        combined = cv2.add(frame, canvas)

        mode_text = "ERASER" if erasing else color_name
        cv2.putText(
            combined, f"Tool: {mode_text}  |  1-5: color  e: eraser  c: clear all  q: quit",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )
        # Small swatch showing the currently selected color.
        cv2.rectangle(combined, (10, 35), (40, 65), draw_color, -1)
        cv2.rectangle(combined, (10, 35), (40, 65), (255, 255, 255), 1)

        cv2.imshow("Finger Drawing", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('e'):
            erasing = not erasing
        elif key in COLORS:
            draw_color, color_name = COLORS[key]
            erasing = False  # picking a color implies you want to draw again

cap.release()
cv2.destroyAllWindows()
