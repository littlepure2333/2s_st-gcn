# Thaks to this blog: https://blog.csdn.net/weixin_43409627/article/details/89454885
# Create a new frame buffer zone to save latest frames.
# Solve the problem that computing is slower than receiving live stream

import cv2
import time

class Stack:
 
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size
 
    def size(self):
        return len(self.items)

    def pop(self):
        if self.size() > 0:
            latest_index = self.size() - 1
            return self.items[latest_index]
        else:
            print("pop none")
            return None
 
    def push(self, item):
        if self.size() >= self.stack_size:
            for i in range(self.size() - self.stack_size + 1):
                self.items.pop(0)
        self.items.append(item)

def capture_thread(video_path, frame_buffer, lock):
#def capture_thread(video_path, frame_buffer):
    print("capture_thread start......")
    vid = cv2.VideoCapture(video_path)
    if not try_to_open(vid):
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        now = time.time()
        #print("capture: {}".format(now))
        if return_value is not True:
            break
        lock.acquire()
        frame_buffer.push(frame)
        #print("stack size: {}".format(frame_buffer.size()))
        lock.release()
        cv2.waitKey(1)

def try_to_open(vid):
    for i in range(1000):
        if vid.isOpened():
            return True
    return False