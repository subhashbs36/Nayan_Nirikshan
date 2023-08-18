import cv2
import datetime
import threading
import telepot

def record_video(frame):
    # cap = cv2.VideoCapture(0) # capture video from default camera
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # video codec
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + '.avi'
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480)) # video writer

    start_time = datetime.datetime.now()
    while (datetime.datetime.now() - start_time).total_seconds() < 10: # record for 10 seconds
        # ret, frame = cap.read()
        if frame:
            out.write(frame)
        else:
            break

    out.release()

    bot = telepot.Bot("5895458028:AAFhA94qkkuWnTu2OmH4mbp0dlT38mEOCfk")
    bot.sendVideo(941558875, video=open(filename, 'rb'))

if __name__ == '__main__':
    t = threading.Thread(target=record_video, )
    t.start()
