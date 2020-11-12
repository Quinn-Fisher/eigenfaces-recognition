import numpy as np
import cv2
from imutils.video import VideoStream

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')



cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow('frame', frame2)

    faces = face_cascade.detectMultiScale(gray)
    img = frame
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#
# vs = VideoStream(src=0).start()
# while True:
#     frame = vs.read()
#     cv2.imshow("frame", frame)
#     cv2.waitKey(0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# faces = face_cascade.detectMultiScale(gray, 1.3, 5)


# for (x, y, w, h) in faces:
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





