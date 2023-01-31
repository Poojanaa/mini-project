# mini-project
Tensorflow Object Detection API Tutorial by sentdex<br>
Raccoon Detector Dataset by Datitran<br>
Tzuta Lin's LabelImg for dataset annotation<br>
This implementation also took a lot of inspiration from the Victor D github repository : https://github.com/victordibia/handtracking<br>
 google colleb<br>
<br>
<br>
<br>
from _collections import deque
import cv2 as cv
import numpy as np


#######################################################################################
                                ## FUNCTIONS ##
#######################################################################################

def empty(a):
    pass

## Defining the HSV values for the blue, orange and green colour
blue_colour = [84, 116, 0, 149, 255, 255]
orange_colour = [0, 120, 118, 62, 255, 255]
green_colour = [41, 89, 94, 71, 200, 255]

## Defining the point for drawing the rectangles for palette
clear_rect = [(625 - 40, 185), (665 - 40, 225)]
blue_rect = [(625 - 40, 5), (665 - 40, 45)]
green_rect = [(625 - 40, 245), (665 - 40, 285)]
yellow_rect = [(625 - 40, 65), (665 - 40, 105)]
red_rect = [(625 - 40, 125), (665 - 40, 165)]
brown_rect = [(625 - 40, 305), (665 - 40, 345)]
orange_rect = [(625 - 40, 365), (665 - 40, 405)]
eraser_rect = [(625 - 40, 425), (665 - 40, 465)]


colors = [(255, 0, 0), ##blue
          (0, 255, 0),  ##green
          (0, 0, 255),  ##red
          (0, 255, 255), ##yellow
          (255, 0 ,255),  ## purple
          (255, 255, 0), ##cyan
          (200, 150 , 0)]


##Adding it to an array of colours for future expansion plans.
myColors = [blue_colour, orange_colour, green_colour]

## Find colour function which detects the colours mentioned in the array
def findColor (img, myColors, paintcolor, clear):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    newPoints = []

    ## Looping through the colours defined to get a mask for each.
    for color in myColors:
        ## Making an array of all the minimum values for h, s and v.
        lower = np.array(color[0:3])
        ## Making an array of all the maximum values for h, s and v.
        upper = np.array(color[3:6])

        ## Create a mask for the frame from the values obtained from the trackbar positions.
        mask = cv.inRange(imgHSV, lower, upper)
        ## Making a new variable to store the bitwise AND of the image with itself applying the mask.
        imgNew = cv.bitwise_and(img, img, mask=mask)
        kernel = np.ones((5, 5), np.uint8)

        ##eroding and dilating the mask
        erosion = cv.erode(mask, kernel, iterations = 1)
        dilation = cv.erode(mask, kernel, iterations = 1)
        ##applying opening and closing on the mask in order to cancel out false negatives.
        opening = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        x,y,detected= getContours(opening)
        if(detected):
            cv.circle(frame, (x,y), 10, paintcolor, cv.FILLED)
        else:
            cv.circle(frame, (x,y), 10, paintcolor, 2)
        ## If not point is detected, i.e x,y = 0,0
        if x!=0 and y!=0:
            ## If the stylus is detected over the drawn rectangles, change the paintcolour.
            if red_rect[0][0]<x<red_rect[1][0] and red_rect[0][1]<y<red_rect[1][1] :
                # print("Detecting red at ", x+40, y)
                paintcolor = colors[2]
            elif blue_rect[0][0]<x<blue_rect[1][0] and blue_rect[0][1]<y<blue_rect[1][1] :
                # print("Detecting blue at ", x+40, y)
                paintcolor = colors[0]
            elif yellow_rect[0][0]<x<yellow_rect[1][0] and yellow_rect[0][1]<y<yellow_rect[1][1] :
                # print("Detecting yellow at ", x+40, y)
                paintcolor = colors[3]
            elif green_rect[0][0]<x<green_rect[1][0] and green_rect[0][1]<y<green_rect[1][1] :
                # print("Detecting green at ", x+40, y)
                paintcolor = colors[1]
            elif brown_rect[0][0]<x<brown_rect[1][0] and brown_rect[0][1]<y<brown_rect[1][1] :
                # print("Detecting green at ", x+40, y)
                paintcolor = colors[4]
            elif orange_rect[0][0]<x<orange_rect[1][0] and orange_rect[0][1]<y<orange_rect[1][1] :
                # print("Detecting green at ", x+40, y)
                paintcolor = colors[5]
            elif eraser_rect[0][0]<x<eraser_rect[1][0] and eraser_rect[0][1]<y<eraser_rect[1][1] :
                # print("Detecting green at ", x+40, y)
                paintcolor = colors[6]
            elif clear_rect[0][0] < x < clear_rect[1][0] and clear_rect[0][1] < y < clear_rect[1][1]:
                # print("Cleared screen")
                clear = True
                paintcolor = (0, 0, 0)
                # print(clear)
            ##append the points to newPoints
            if(detected):
                newPoints.append([x, y, paintcolor])
        ## Show masked image.
        # cv.imshow(str(color[0]), mask)
        # cv.imshow(str(color[1]), erosion)
        # cv.imshow(str(color[2]), dilation)
        # cv.imshow("Opening", opening)
        #cv.imshow("Close", closing)
    if(clear):
        new_points.clear()
    return newPoints, paintcolor, clear


myPoints = []  #[x , y, colorId]



## Function to get the contour of the detected masked image and draw bounding box around it
def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print("Contours : ", contours)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        # print( area )
        cv.drawContours(boundingFrame, cnt, -1, (255, 255, 255), 3)
        peri = cv.arcLength(cnt, True)  # The 'true' here is to ensure we're detecting closed stuff
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)  # To get the corner points?
        x, y, w, h = cv.boundingRect(approx)
        if area > 500:
            return x+w//2,y,True ##return the tip of the detected stylus
    return x+w//2,y,False ##If not detected, return 0,0

## Used to get the desired colour.
def colour_picker():
    ## Creating a window called Trackbars in order to get the desired values to mask ##
    ## Values obtained in the trackbars are then hardcoded to detect the desired colour. ##
    trk = cv.namedWindow("TrackBars")
    cv.resizeWindow("TrackBars", 640, 240)
    cv.createTrackbar("Hue Min", "TrackBars", 41, 179, empty)
    cv.createTrackbar("Hue Max", "TrackBars", 100, 255, empty)
    cv.createTrackbar("Sat Min", "TrackBars", 54, 255, empty)
    cv.createTrackbar("Sat Max", "TrackBars", 177, 255, empty)
    cv.createTrackbar("Val Min", "TrackBars", 67, 255, empty)
    cv.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
    cv.putText(trk, "Adjust!", (208, 33), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

    ## Opening the camera using device '0' i.e in-built camera.
    cap = cv.VideoCapture(0)

    ##  Looping while True where we read framewise, flip the frame to get desired directional motion.
    while True:
        # img=cv.imread("Resources/Screenshot 2020-09-21 112519.png")
        ret, img = cap.read()
        img = cv.flip(img, 1)
        boundingFrame = img.copy()
        ## Resize the image
        img2 = cv.resize(img, (500, 400))
        ##Convert the image to HSV  (hue, saturation, value, also known as HSB or hue, saturation, brightness).
        imgHSV = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

        ## Getting the values from the trackbars that were adjusted.
        h_min = cv.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv.getTrackbarPos("Val Max", "TrackBars")

        ## Making an array of all the minimum values for h, s and v.
        lower = np.array([h_min, s_min, v_min])
        ## Making an array of all the maximum values for h, s and v.
        upper = np.array([h_max, s_max, v_max])

        ## Print in each iteration to debug.
        ##print(h_min,h_max,s_min,s_max,v_min,v_max)

        ## Create a mask for the frame from the values obtained from the trackbar positions.
        mask = cv.inRange(imgHSV, lower, upper)

        ## Making a new variable to store the bitwise AND of the image with itself applying the mask.
        imgNew = cv.bitwise_and(img2, img2, mask=mask)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(mask, kernel, iterations=1)
        dilation = cv.erode(mask, kernel, iterations=1)
        ##applying opening and closing on the mask in order to cancel out false negatives.
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        ## Show both the image and masked image.
        cv.imshow("mask", erosion)
        cv.imshow("Out", img2)
        ##Delay of 1 ns
        cv.waitKey(1)

        ##If 'q' is pressed, loop breaks and operation stops.
        if (cv.waitKey(1) & 0xFF == ord('q')):
            break

##Looping through the points and drawing the points as circles.
def drawOnCanvas( myPoints, paintWindow):
    for point in myPoints:
        cv.circle(paintWindow,(point[0], point[1]), 10, point[2], cv.FILLED)



#########################################################################################

colorIndex = 0
print(" ############################   AIR CANVAS   ############################\n"
      "   This is a basic project based on computer vision made in OpenCV Python which enables \n"
      "        the user to draw on their system screen by drawing in air with a target \n"
      "* The target colour currently being recognized is : blue, green or orange. \n"
      " ** Make sure your background is clear and no other object is hindering the detecting mechanism ***\n \n"
      " Make a choice: \n"
      "1. Start drawing! \n"
      "2. Choose another stylus colour for detecting \n"
      "3. Exit the application\n "
      )
choice = int(input("Enter your choice: "))
if choice == 1:
    ## Create a window which will act as the canvas.
    paintWindow = np.zeros((471, 636, 3)) + 255
    clearedwindow = paintWindow
    cv.putText(paintWindow, "DRAW HERE BRO!", (208, 33), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

    cv.namedWindow('Paint', cv.WINDOW_AUTOSIZE)
    # cv.imshow("Canvas", paintWindow)
    clear = False
    # Load the video
    camera = cv.VideoCapture(0)
    paintcolor = (0, 0, 0)
    # Keep looping throught the frames.
    while True:
        clear = False

        # Grab the current frame and flip it to get desired direction of stylus
        (grabbed, frame) = camera.read()
        frame = cv.flip(frame, 1)
        boundingFrame = frame.copy()
        # Check to see if we have reached the end of the video (useful when input is a video file not a live video stream)
        if not grabbed:
            break

        ## Draw the colour palette on the right hand side.
        ## Open to editing and reshaping as well as alignment.
        frame = cv.rectangle(frame, clear_rect[0], clear_rect[1], (0, 0, 0), 1)
        frame = cv.rectangle(frame, blue_rect[0], blue_rect[1], colors[0], 2)
        frame = cv.rectangle(frame, green_rect[0], green_rect[1], colors[1], 2)
        frame = cv.rectangle(frame, yellow_rect[0],yellow_rect[1], colors[3], 2)
        frame = cv.rectangle(frame, red_rect[0], red_rect[1], colors[2], 2)
        frame = cv.rectangle(frame, brown_rect[0], brown_rect[1], colors[4], 2)
        frame = cv.rectangle(frame, orange_rect[0], orange_rect[1], colors[5], 2)
        frame = cv.rectangle(frame, eraser_rect[0], eraser_rect[1], colors[6], 2)
        # colour[5] is eraser
        ## Colours can be named as well if desired.
        cv.putText(frame, "CLEAR ALL", (625-45, 185+50), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv.LINE_AA)
        # cv.putText(frame, "BLUE", (185, 33), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(frame, "GREEN", (298, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(frame, "RED", (420, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(frame, "YELLOW", (520, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv.LINE_AA)
        ## Show each frame until 'q' is clicked.
        new_points, paintcolor, clear = findColor(frame, myColors, paintcolor, clear)
        # print(paintcolor)


        if len(new_points)!=0:
            for newpoint in new_points:
                myPoints.append(newpoint)
        if len(myPoints)!=0:
            if(clear):
                paintWindow = np.zeros((471, 636, 3)) + 255
                myPoints.clear()
            else:
                drawOnCanvas(myPoints, paintWindow)
        cv.imshow('frame', frame)

        cv.imshow('frameD', paintWindow)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

elif choice == 2:
    colour_picker()
else :
    exit(0)<br>
    <br>
    <br>
https://github.com/AirCanvas/MyCanvas/blob/main/Air%20Canvas.py<br>
<br>
<br>
<br>
https://risx3.github.io/emojinator/<br>
<br>
import numpy as np
import cv2
from collections import deque


def setValues(x):
   print("")


cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 135, 180,setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 205, 255,setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180,setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 85, 255,setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 111, 255,setValues)


bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]


blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0


kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0


paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)



cap = cv2.VideoCapture(0)


while True:
   
    ret, frame = cap.read()
  
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    Upper_hsv = np.array([u_hue,u_saturation,u_value])
    Lower_hsv = np.array([l_hue,l_saturation,l_value])


   
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
    frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
    frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)


   
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    
    cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None

    
    if len(cnts) > 0:
    	
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]        
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        if center[1] <= 65:
            if 40 <= center[0] <= 140: 
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 
        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1


    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

   
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("mask",Mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()<br>
<br>
<br>

https://data-flair.training/blogs/create-emoji-with-deep-learning/<br>
https://github.com/infoaryan/Air-Canvas-with-ML<br>

import numpy as np
import cv2
from tensorflow.keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
<br>
<br>




