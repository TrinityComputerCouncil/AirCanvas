import numpy as np
import cv2
from collections import deque

from magicinair.config import CENTER_CIRCLE, RADIUS, VIDEO_HEIGHT, VIDEO_WIDTH
from magicinair.utils.checkRadius import check_radius

def setValues(x):
    print("")

# Class for actual computer vision
class MagicAir:
    def __init__(self) -> None:
        self.colors =  [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_index = 0
        self.blue_index = 0
        self.green_index = 0
        self.red_index = 0
        self.yellow_index = 0
        self.kernel = np.ones((5,5),np.uint8)
        self.b_points = [deque(maxlen=1024)]
        self.g_points = [deque(maxlen=1024)]
        self.r_points = [deque(maxlen=1024)]
        self.y_points = [deque(maxlen=1024)]
        self.run = True

    def update(self) -> None:
        self.setup_marker()
        self.setup_canvas()
        self.load_cam()
        self.loop()

    def setup_marker(self) -> None:
        cv2.namedWindow("Color detectors")
        cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180,setValues)
        cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255,setValues)
        cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,setValues)
        cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180,setValues)
        cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255,setValues)
        cv2.createTrackbar("Lower Value", "Color detectors", 49, 255,setValues)

    def setup_canvas(self) -> None:
        self.paintWindow = np.zeros((471,636,3)) + 255
        self.paintWindow = cv2.rectangle(self.paintWindow, (40,1), (140,65), (0,0,0), 2)
        self.paintWindow = cv2.rectangle(self.paintWindow, (160,1), (255,65), self.colors[0], -1)
        self.paintWindow = cv2.rectangle(self.paintWindow, (275,1), (370,65), self.colors[1], -1)
        self.paintWindow = cv2.rectangle(self.paintWindow, (390,1), (485,65), self.colors[2], -1)
        self.paintWindow = cv2.rectangle(self.paintWindow, (505,1), (600,65), self.colors[3], -1)

        cv2.putText(self.paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
        cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    def load_cam(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    def handle_trackbar(self) -> None:
            u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
            u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
            u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
            l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
            l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
            l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
            self.Upper_hsv = np.array([u_hue,u_saturation,u_value])
            self.Lower_hsv = np.array([l_hue,l_saturation,l_value])

    def create_mask(self) -> None:
            # Identifying the pointer by making its mask
            self.Mask = cv2.inRange(self.hsv, self.Lower_hsv, self.Upper_hsv)
            self.Mask = cv2.erode(self.Mask, self.kernel, iterations=1)
            self.Mask = cv2.morphologyEx(self.Mask, cv2.MORPH_OPEN, self.kernel)
            self.Mask = cv2.dilate(self.Mask, self.kernel, iterations=1)

    def handle_btns(self) -> None:

        self.frame = cv2.rectangle(self.frame, (40,1), (140,65), (122,122,122), -1)
        self.frame = cv2.circle(self.frame, CENTER_CIRCLE, RADIUS, (122, 122, 122), -1)
        self.frame = cv2.rectangle(self.frame, (160,1), (255,65), self.colors[0], -1)
        self.frame = cv2.rectangle(self.frame, (275,1), (370,65), self.colors[1], -1)
        self.frame = cv2.rectangle(self.frame, (390,1), (485,65), self.colors[2], -1)
        self.frame = cv2.rectangle(self.frame, (505,1), (600,65), self.colors[3], -1)     
        cv2.putText(self.frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

    def find_countours(self) -> None:
        # Find contours for the pointer after idetifying it
        self.cnts,_ = cv2.findContours(self.Mask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        center = None

    def handle_countours(self) -> None:
        # Ifthe contours are formed
        if len(self.cnts) > 0:
        	# sorting the contours to find biggest 
            cnt = sorted(self.cnts, key = cv2.contourArea, reverse = True)[0]
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            # Draw the circle around the contour
            cv2.circle(self.frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Calculating the center of the detected contour
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))       
            # Now checking if the user wants to click on any button above the screen 
            print(check_radius(CENTER_CIRCLE, center, RADIUS))
            if center[1] <= 65:
                if 40 <= center[0] <= 140: # Clear Button
                    self.b_points = [deque(maxlen=512)]
                    self.g_points = [deque(maxlen=512)]
                    self.r_points = [deque(maxlen=512)]
                    self.y_points = [deque(maxlen=512)]     
                    self.blue_index = 0
                    self.green_index = 0
                    self.red_index = 0
                    self.yellow_index = 0       
                    self.paintWindow[67:,:,:] = 255
                elif 160 <= center[0] <= 255:
                        self.color_index = 0 # Blue
                elif 275 <= center[0] <= 370:
                        self.color_index = 1 # Green
                elif 390 <= center[0] <= 485:
                        self.color_index = 2 # Red
                elif 505 <= center[0] <= 600:
                        self.color_index = 3 # Yellow
            else :
                if self.color_index == 0:
                    self.b_points[self.blue_index].appendleft(center)
                elif self.color_index == 1:
                    self.g_points[self.green_index].appendleft(center)
                elif self.color_index == 2:
                    self.r_points[self.red_index].appendleft(center)
                elif self.color_index == 3:
                    self.y_points[self.yellow_index].appendleft(center)
        # Append the next deques when nothing is detected to avois messing up
        else:
            self.b_points.append(deque(maxlen=512))
            self.blue_index += 1
            self.g_points.append(deque(maxlen=512))
            self.green_index += 1
            self.r_points.append(deque(maxlen=512))
            self.red_index += 1
            self.y_points.append(deque(maxlen=512))
            self.yellow_index += 1      
        # Draw lines of all the colors on the canvas and frame 
        points = [self.b_points, self.g_points, self.r_points, self.y_points]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(self.frame, points[i][j][k - 1], points[i][j][k], self.colors[i], 2)
                    cv2.line(self.paintWindow, points[i][j][k - 1], points[i][j][k], self.colors[i], 2)     

    def update_windows(self) -> None:
        # Show all the windows
        cv2.imshow("Tracking", self.frame)
        cv2.imshow("Paint", self.paintWindow)
        cv2.imshow("mask",self.Mask)

    def handle_close(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
    
    def handle_click(self) -> None:
        # If the 'q' key is pressed then stop the application 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.run = False

    def loop(self) -> None:
        while self.run:
            ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame, 1)
            self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            self.handle_trackbar()
            self.create_mask()
            self.handle_btns()
            self.find_countours()
            self.handle_countours()
            self.update_windows()
            self.handle_click()
        self.handle_close()

air = MagicAir()
air.update()
