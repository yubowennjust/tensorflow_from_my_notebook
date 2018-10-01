import cv2
import time

start = int(time.time())

url = 'rtsp://admin:njust123@192.168.2.118:554'

cap = cv2.VideoCapture(url)

while(cap.isOpened()):

    ret, frame = cap.read()

    cv2.imshow('frame',cv2.resize(frame, (960,540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()

