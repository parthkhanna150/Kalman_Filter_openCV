import numpy as np
import cv2
import sys
import plotly
import datetime
import random
import matplotlib.pyplot as plt

bgr_color = 30, 15, 140
color_threshold = 70

hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0]
HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])


def detect_ball(frame, x_list):
    x, y, radius = -1, -1, -1
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1, -1)
    # print(len(contours))
    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # make up some data
        
        # x_list.append(x)

        # check that the radius is larger than some threshold
        if radius > 10:
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius


if __name__ == "__main__":
    filepath = sys.argv[1]
    t=0
    t_list=[]
    x_list=[]
    cap = cv2.VideoCapture(filepath)
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # t+=1
        # t_list.append(t)
        
        # print(x_list,t_list)
        
        detect_ball(frame, x_list)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # plt.plot(x_list,t_list)
    # plt.gcf().autofmt_xdate()
    # plt.show()

    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()


