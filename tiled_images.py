import cv2
from enum import Enum
import numpy as np
import time
import string
import random

def start_stop():
    global RunState,StartCountDown
    global BackgroundCaptureState
    global out

    print("start stop")
    BackgroundCaptureState = BackgroundCaptureState.RELEASED
    if (RunState != RunState.STOPPED):
        RunState = RunState.STOPPED
        if (out is not None):
            cv2.setWindowTitle( "Window", "not recording" )
            out.release()
            out = None
    else:
        RunState = RunState.STARTING            
        outname = str(int(time.time()))+ ".avi"
        cv2.setWindowTitle( "Window", outname)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(outname,fourcc,10.0,(int(frame_width * frame_scale),int(frame_height * frame_scale)))
        StartCountDown = time.time() + 10


def GetGrey(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #b, g, r = cv2.split(frame)
    return grey
    
    
def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop

def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


# Open the camera uncomment whichever line applies
#
# single camera default raw mode
#cam = cv2.VideoCapture(0)

# dual camera default raw mode
#cam = cv2.VideoCapture(2)

# dual camera higher res mjpeg mode
#cam = cv2.VideoCapture("v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 !jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink")

#from video  
cam = cv2.VideoCapture('../schlieren/1733870027.avi')

# Get the frame width and height, set scale to resize to 800px width
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_scale = 800.0 / frame_width;
print("frame width=={0} frame height=={1} frame scale=={2}".format(frame_width,frame_height,frame_scale))

cv2.namedWindow("Window")
loop_time = 0
loop_ctr = 0
frame_ctr = 0
while True:

    # display loop timer, start loop timer
    if (loop_ctr == 100):
        print("average loop time == {0}ms".format(int(loop_time*10.0)))
        loop_time = 0
        loop_ctr = 0
    loop_start = time.perf_counter()
    
    # get frame from camera
    ret, frame = cam.read()

    # produce crop images (first 100 frames)
    if (frame_ctr < 100):
        image_ctr = 0
        h = 0
        w = 0
        while (h < frame_height):
            if (h + 224 > frame_height):
                h = frame_height - 224
            while (w < frame_width):
                if (w + 224 > frame_width):
                    w = frame_width - 224
                #cv2.rectangle(frame,(w,h),(w+224,h+224),(0,255,0),3)
                crop_image = frame[h:h+224, w:w+224]
                crop_filename = f"image-{frame_ctr:05}-{image_ctr:02}.png"
                image_ctr += 1
                cv2.imwrite(crop_filename,crop_image)
                w += 224
            w = 0        
            h += 224
        frame_ctr += 1         
  
    display = frame
    # resize image to 800px wide to fit screen.
    small = cv2.resize(display, (0,0), fx=frame_scale, fy=frame_scale) 

    
    # show image  
    cv2.imshow('Window', small)
    
    # Press 'q' to exit the loop
    key = cv2.waitKey(80)
    if (key == 113):
        print("quitting...")
        break
    elif (key != -1):
        print(key)             

    # end loop timer and add to total
    loop_end = time.perf_counter()
    loop_time = loop_time + (loop_end - loop_start)
    loop_ctr = loop_ctr + 1

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()

