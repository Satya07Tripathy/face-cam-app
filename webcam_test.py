import cv2

# 0 means "the first camera Windows finds" (usually your built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    # .read() grabs one frame (one image) from the camera.
    # ret tells us if it worked, frame is the actual image (as a grid of pixel numbers)
    ret, frame = cap.read()
    if not ret:
        break

    # The raw feed is what the camera literally sees, which is flipped
    # compared to a mirror (your right hand appears on the frame's right
    # side too, not the left like in a mirror). Flipping horizontally
    # (code 1 = flip left-right) makes it feel natural to look at.
    frame = cv2.flip(frame, 1)

    cv2.imshow("Webcam", frame)

    # Without this, the window would freeze instantly.
    # It waits 1ms for a keypress; if you press 'q', we exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
