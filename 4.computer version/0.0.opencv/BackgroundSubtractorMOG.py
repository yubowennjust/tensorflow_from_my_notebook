import cv2

url = 'rtsp://admin:njust123@169.254.62.157:554'
capture = cv2.VideoCapture(url)

# capture = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = capture.read()
    fmask = fgbg.apply(frame)

    cv2.imshow("Orignal Frame", frame)
    cv2.imshow("F Mask", fmask)

    if cv2.waitKey(30) & 0xff == 27:
        break

capture.release()
cv2.destroyAllWindows()
