import cv2

# OpenCV ships with pre-trained detectors as XML files. This one is trained
# to recognize the general pattern of a human face (eyes, nose bridge, etc).
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally so the feed behaves like a mirror (otherwise your
    # right hand shows up on the frame's right side, not the left).
    frame = cv2.flip(frame, 1)

    # Face detection works on grayscale images (it only cares about light/dark
    # patterns, not color), and it's faster on a single-channel image too.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Returns a list of rectangles, one per detected face: (x, y, width, height)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much the image is shrunk at each scan pass
        minNeighbors=5,    # higher = fewer false positives, may miss real faces
        minSize=(60, 60),  # ignore anything smaller than 60x60 pixels
    )

    for (x, y, w, h) in faces:
        # Draw a green rectangle around each detected face on the color frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "Face", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
