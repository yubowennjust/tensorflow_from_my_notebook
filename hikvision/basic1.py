import multiprocessing as mp
import face_recognition
import cv2
import os


'''2018-05-21 Yonv1943'''
'''2018-07-02 setattr(), run_multi_camera()'''


def queue_img_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 1 else None


def queue_img_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run():  # single camera
    user_name, user_pwd, camera_ip = "admin", "njust123", "192.168.2.118"
    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=queue_img_get, args=(queue, camera_ip))]
    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]

if __name__ == '__main__':
    run()

