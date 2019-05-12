import cv2 as cv
import numpy as np
cv.namedWindow("preview")

video = cv.VideoCapture(0)


def get_video():
    if video.isOpened(): ## try to get first frame
        rval, frame = video.read()
    else:
        rval = False

    while rval:
        frame = face_detection(frame)

        cv.imshow("preview", frame)
        rval, frame = video.read()
        key = cv.waitKey(20)

        if key == 27: #exit on esc
            break

    cv.destroyWindow("preview")


def face_detection(image):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def lane_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    lower_yellow = np.array([20, 255, 255], dtype="uint8")




if __name__ == "__main__":

    get_video()