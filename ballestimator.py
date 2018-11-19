import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy import stats
import math

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

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # KalmanFilter.
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # check that the radius is larger than some threshold
        x_list.append(x)
        if radius > 10:
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius

if __name__ == "__main__":
    filepath = sys.argv[1]
    cap = cv2.VideoCapture(filepath)
    x_list = []

    for i in range(0,350):

        # Capture frame-by-frame
        ret, frame = cap.read()
        detect_ball(frame, x_list)

        # Display the resulting frame

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    coords = np.array(x_list)
    coords = np.reshape(coords, (len(coords), 1))

    if filepath == 'ball1.mp4':
        np.save('./true.npy', coords)

    kf = KalmanFilter(transition_matrices=[1],
                  observation_matrices=[1],
                  initial_state_mean=coords[0],
                  initial_state_covariance=1,
                  observation_covariance=1,
                  transition_covariance=(coords[-1]-coords[0])/(coords.shape[0]))
    means, covars = kf.filter(coords)
   
    # index = np.arange(0, len(coords))

    # index =  (index/len(index))
    # plt.plot(index, coords, '-b', linewidth=2, label = 'Measurements')
    # plt.savefig(filepath + '_x_coords'+'.jpg')
    
    # plt.plot(index, means, '-g', label = 'Kalman', linewidth=2)
    # plt.title('Kalman vs Time')
    # plt.legend()
    # plt.savefig(filepath + '_means'+'.jpg')

#     plt.show(block = False)
#     plt.close('all')
#     plt.plot(covars[:,0,0], '-b')
#     plt.title('Variances vs Time')
#     plt.savefig(filepath + '_var'+'.jpg')

#     true = np.load('./true.npy')
#     true_var = np.sum(math.pow((means-true),2))/math.sqrt(len(true_var))
#     true_var = np.sqrt(true_var)


#     kf = KalmanFilter(transition_matrices=[1],
#                   observation_matrices=[1],
#                   initial_state_mean=coords[0],
#                   initial_state_covariance=1,
#                   observation_covariance=true_var,
#                   transition_covariance=(coords[-1]-coords[0])/(coords.shape[0]))

#     means, covars = kf.filter(coords)

#     index = np.arange(0, len(coords))
#     index =  (index/len(index))

#     plt.close('all')
#     plt.plot(index, coords, '-b', linewidth=2, label = 'Obtained coordinates')
#     plt.plot(index, means, '-g', label = 'Kalman', linewidth=2)
#     plt.title('Kalman vs Time: With true variance')
#     plt.legend()
#     plt.savefig(filepath + '_true_var'+'.jpg')
#     plt.show(block = False)

# #plot
    cap.release()
    cv2.destroyAllWindows()