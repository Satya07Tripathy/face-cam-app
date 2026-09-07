# Face Cam App

A webcam computer vision app built with Python and OpenCV. It detects your face, estimates your age/gender/emotion, and lets you draw on screen by tracking your fingertip through the camera.

**Windows only. Requires a webcam.**

## Download

Grab the latest `FaceCamApp.exe` from the [Releases page](../../releases/latest) — no Python or installation required. Just download and double-click.

> First launch downloads ~326MB of AI model files (one-time, needs internet). Windows may show a "Windows protected your PC" SmartScreen warning since the app isn't code-signed — click **More info → Run anyway**. This is a common false positive for this kind of packaged app, not a sign of anything wrong.

## Features

- **Face detection** — a green box tracks your face in real time.
- **Age, gender & emotion estimation** — runs two small neural networks on the detected face.
- **Finger drawing** — pinch your thumb and index finger together to lift the pen, spread them apart to draw, tracked entirely through your hand in the camera feed (no mouse or touchscreen).

## Controls

| Key | Action |
|---|---|
| `m` | Switch between Analyze mode and Draw mode |
| `1`–`5` | Pick a draw color (red/green/blue/yellow/white) |
| `e` | Toggle eraser |
| `c` | Clear the drawing |
| `q` | Quit |

## Running from source

```
git clone <this repo>
cd face-cam-app
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\pip install mediapipe==1.0.1 --no-deps
.\venv\Scripts\python combined_app.py
```

There's also `gui_app.py`, a button-driven version of the same app (no keyboard shortcuts needed) built with Tkinter.

## How it works

- Face detection: OpenCV Haar cascades.
- Age/gender: [insightface](https://github.com/deepinsight/insightface)'s `genderage` model, via ONNX Runtime.
- Emotion: [hsemotion-onnx](https://github.com/av-savchenko/hsemotion-onnx).
- Hand tracking: [MediaPipe](https://github.com/google-ai-edge/mediapipe) Hand Landmarker.

All models run fully on-device — no cloud APIs, no accounts, no data leaves your machine.

## License

MIT — see [LICENSE](LICENSE).
