import cv2
from enum import Enum
import numpy as np
import time

# global variables
BackgroundImage = None 
StartCountDown = None
frame = None
out = None   

class BackgroundCaptureState(Enum):
    RELEASED = 1
    CAPTURING = 2
    CAPTURED = 3
BackgroundCaptureState = BackgroundCaptureState.RELEASED

class RunState(Enum):
    STOPPED = 1
    STARTING = 2
    STARTED = 3
    PAUSED = 4
RunState = RunState.STOPPED

class FlowState(Enum):
    TURBULENT = 1           # turbulent flow background subtraction of set image
    SQUAREDx20 = 2          # squared formula frame difference of set image
    LAMINAR = 3             # laminar flow with background detected and perspective transfomed
    ROLLING = 4             # subtraction of last frame
    NONE = 5                # no flow processing (use to record)
    def next(self):
        members = list(self.__class__)
        index = members.index(self) + 1
        if (index >= len(members)):
            index = 0
        return members[index]    
FlowState = FlowState.TURBULENT        


# id values for markers
# 0 top left 
# 1 top right
# 2 bottom right 
# 3 bottom left
#
def extractMarkerArea(corners,ids,grey):
    global frame_width, frame_height
    
    for i in range(4):
      if (ids[i] == 0):
          topLeft = corners[i][0][3]	    # top left point of top left marker
      elif (ids[i] == 1):
          topRight = corners[i][0][0]  	    # top right point of top right marker
      elif (ids[i] == 2):
          bottomRight = corners[i][0][1] 	# bottom right point of bottom right marker
      elif (ids[i] == 3):
          bottomLeft = corners[i][0][2]	    # bottom left point of bottom left marker

    input_pts = np.float32([topLeft,topRight,bottomRight,bottomLeft])
    output_pts = np.float32([[0, 0],[frame_width - 1, 0],[frame_width- 1, frame_height - 1],[0, frame_height - 1]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    grey = cv2.warpPerspective(grey,M,(frame_width, frame_height),flags=cv2.INTER_LINEAR)
    return grey

def set_pause():
    global RunState
    if (RunState == RunState.STARTED):
        RunState = RunState.PAUSED
        print("pause recording")
    elif (RunState == RunState.PAUSED):
        RunState = RunState.STARTED
        print("resume recording")

def set_background():
    global BackgroundCaptureState
    print("reset background")
    BackgroundCaptureState = BackgroundCaptureState.RELEASED

def set_flow():
    global FlowState
    global BackgroundCaptureState

    FlowState = FlowState.next()
    print("set flow type to " + FlowState.name)
    BackgroundCaptureState = BackgroundCaptureState.RELEASED


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
    

def AnalyzeLaminarFlow():
    global frame
    global BackgroundCaptureState
    global BackgroundImage

    grey = GetGrey(frame)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict,parameters=arucoParams)
    if (len(corners) == 4):
        grey = extractMarkerArea(corners,ids,grey)
        if (BackgroundCaptureState == BackgroundCaptureState.RELEASED):
            BackgroundCaptureState = BackgroundCaptureState.CAPTURED
            BackgroundImage = grey
        else:
            grey = cv2.absdiff(grey,BackgroundImage)
            grey = cv2.multiply(grey,6.0)
    return grey


def AnalyzeTurbulentFlow():
    global frame
    global BackgroundCaptureState
    global BackgroundImage

    grey = GetGrey(frame)
    if (BackgroundCaptureState == BackgroundCaptureState.RELEASED):
        BackgroundCaptureState = BackgroundCaptureState.CAPTURED
        BackgroundImage = grey
        display = frame
    else:
        display = cv2.absdiff(grey,BackgroundImage)
        display = cv2.multiply(display,6.0)
    return display


def AnalyzeRollingFlow():
    global frame
    global BackgroundCaptureState
    global BackgroundImage

    grey = GetGrey(frame)
    if (BackgroundCaptureState == BackgroundCaptureState.RELEASED):
        BackgroundCaptureState = BackgroundCaptureState.CAPTURED
        BackgroundImage = grey
        display = frame
    else:
        display = cv2.absdiff(grey,BackgroundImage)
        display = cv2.multiply(display,6.0)
        BackgroundImage = grey
    return display


def AnalyzeSquaredFlow():
    global frame
    global BackgroundCaptureState
    global BackgroundImage

    grey = GetGrey(frame)
    if (BackgroundCaptureState == BackgroundCaptureState.RELEASED):
        BackgroundCaptureState = BackgroundCaptureState.CAPTURED
        BackgroundImage = grey
        display = frame
    else:
        grey_float = np.float32(grey)
        background_float = np.float32(BackgroundImage)
        grey_squared = np.square(grey_float - background_float)
        grey_sum = grey_float + background_float
        grey_sum = grey_sum / 2.0
        grey_sum = grey_sum + 1.0
        grey_sum = grey_squared / grey_sum
        grey_sum = grey_sum * 20.0
        display = np.uint8(np.clip(grey_sum, 0, 255))
    return display


# Open the camera uncomment whichever line applies
#
# single camera default raw mode
cam = cv2.VideoCapture(0)
# dual camera default raw mode
#cam = cv2.VideoCapture(2)
# dual camera higher res mjpeg mode
#cam = cv2.VideoCapture("v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 !jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink")


# Get the frame width and height, set scale to resize to 800px width
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_scale = 800.0 / frame_width;
print("frame width=={0} frame height=={1} frame scale=={2}".format(frame_width,frame_height,frame_scale))

cv2.namedWindow("Window")
cv2.setWindowTitle( "Window", "not recording" )

loop_time = 0
loop_ctr = 0

while True:

    # display loop timer, start loop timer
    if (loop_ctr == 100):
        print("average loop time == {0}ms".format(int(loop_time*10.0)))
        loop_time = 0
        loop_ctr = 0
    loop_start = time.perf_counter()
    
    # get frame from camera
    ret, frame = cam.read()

    if (RunState == RunState.STOPPED):
        display = frame

    if (RunState == RunState.STARTING):
        if (time.time() > StartCountDown):
            RunState = RunState.STARTED
        else:
            display = frame    

    if (RunState == RunState.STARTED or RunState == RunState.PAUSED):
        if (FlowState == FlowState.LAMINAR):
            display = AnalyzeLaminarFlow()
        elif (FlowState == FlowState.TURBULENT):
            display = AnalyzeTurbulentFlow()
        elif (FlowState == FlowState.SQUAREDx20):
            display = AnalyzeSquaredFlow()
        elif (FlowState == FlowState.ROLLING):
            
            display = AnalyzeRollingFlow()
        else:
            display = frame

    # resize image to 800px wide to fit screen.
    small = cv2.resize(display, (0,0), fx=frame_scale, fy=frame_scale) 

    # add status info
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    lineType = 2
    fontColor = (255,255,255)

    bottomLeftCornerOfText = (10,int(frame_height * frame_scale) - 20)
    cv2.putText(small,RunState.name,bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)
    bottomLeftCornerOfText = (int(frame_width * frame_scale) - 200,int(frame_height * frame_scale) - 20)
    cv2.putText(small,FlowState.name,bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)
        
    # show image  
    cv2.imshow('Window', small)
    
    if (RunState == RunState.STARTED or RunState == RunState.STARTING):
        # if no flow state remove titles from image written to video
        if (FlowState == FlowState.NONE):
            small = cv2.resize(display, (0,0), fx=frame_scale, fy=frame_scale) 
        # convert to color if greyscale, can not mix the two in a video
        if (len(small.shape) == 2):
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        out.write(small)

    # Press 'q' to exit the loop
    key = cv2.waitKey(80)
    if (key == 113):
        print("quitting...")
        break
    elif (key == 114):
        set_background()
    elif (key == 115):
        start_stop()
    elif (key == 102):
        set_flow()
    elif (key == 112):
        set_pause()    
    elif (key != -1):
        print(key)             

    # end loop timer and add to total
    loop_end = time.perf_counter()
    loop_time = loop_time + (loop_end - loop_start)
    loop_ctr = loop_ctr + 1

# Release the capture and writer objects
cam.release()
if (out is not None):
    out.release()
cv2.destroyAllWindows()

