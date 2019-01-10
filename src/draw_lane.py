#!/usr/bin/env python
import sys
import rospy
import cv2
import math
import numpy as numpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_utils import *

bridge = CvBridge()

# Define Lane Coefficients Buffer
global coeff_buffer
coeff_buffer = []

'''
------------------------------------------------------------------------
BASIC SETTINGS
------------------------------------------------------------------------
'''
# Maximum offset pixels from previous lane polynomial
LANE_ROI_OFFSET = 100

# Map size (2018 Competition : 600x600 -> 2019 Competition : 200x200 1px:3cm)
MAP_SIZE = 600

def callback(data):
    img_input = bridge.imgmsg_to_cv2(data, 'bgr8')
    
    '''
    ------------------------------------------------------------------------
    IMAGE PROCESSING
    ------------------------------------------------------------------------
    '''
    # Blurring, Converts BGR -> HSV color space
    img = cv2.GaussianBlur(img_input, (5,5),0)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masking color
    yellow_mask = findColor(hsv_img, lower_yellow, upper_yellow)

    # Eliminating small unnecessary dots (morphologyEx)
    kernel = np.ones((5,5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    mask = yellow_mask
    points_mask = np.where(mask>0)

    x_vals = points_mask[1]
    y_vals = points_mask[0]
    
    '''
    ------------------------------------------------------------------------
    GETTING LANE DATA (LINE FITTING - 2ND ORDER POLYNOMIAL COEFFICIENTS)
    ------------------------------------------------------------------------
    '''

    if(len(coeff_buffer)<3):
        # Previous coefficient data is not sufficient (less than 3)
        coeff = np.polyfit(x_vals, y_vals,2)
        coeff_buffer.append(coeff)
        new_coeff = coeff
    
    else:
        # Previous coefficient data is sufficient (more than 3)
        
        # Calculate coefficients using ROI ###START
        last_coeff = coeff_buffer[2]
        last_f = np.poly1d(last_coeff)

        # Target points inner ROI (Comparing with previous lane data)
        y_vals_roi = y_vals[abs(y_vals - last_f(x_vals))<LANE_ROI_OFFSET]
        x_vals_roi = x_vals[abs(y_vals - last_f(x_vals))<LANE_ROI_OFFSET]
        
        coeff = np.polyfit(x_vals_roi, y_vals_roi,2)

        x_vals = x_vals_roi
        y_vals = y_vals_roi
        
        # Using buffers for filtering
        # 1. Calculate rsquared for last 3 coefficients each
        # 2. Calculate weights of each coefficients using rsquared & softmax function
        prev_f1 = np.poly1d(coeff_buffer[1])
        prev_f2 = np.poly1d(coeff_buffer[2])
        current_f = np.poly1d(coeff)

        rsquared_prev1 = calculate_rsquared(x_vals, y_vals, prev_f1)
        rsquared_prev2 = calculate_rsquared(x_vals, y_vals, prev_f2)
        rsquared_current = calculate_rsquared(x_vals, y_vals, current_f)

        exp_sum = math.exp(rsquared_prev1) + math.exp(rsquared_prev2) + math.exp(rsquared_current)
        weight_prev1 = math.exp(rsquared_prev1) / exp_sum
        weight_prev2 = math.exp(rsquared_prev2) / exp_sum
        weight_current = math.exp(rsquared_current) / exp_sum

        new_coeff = weight_prev1 * coeff_buffer[1] + weight_prev2 * coeff_buffer[2] + weight_current * coeff

        # Updating buffer
        coeff_buffer[0:-1] = coeff_buffer[1:3]
        coeff_buffer[2] = new_coeff

    t = np.arange(0,MAP_SIZE,1)
    f = np.poly1d(new_coeff)

    polypoints = np.zeros((MAP_SIZE,2))
    polypoints[:,0] = t
    polypoints[:,1] = f(t)

    cv2.polylines(img, np.int32([polypoints]), False, (255,0,0),2)

    cv2.imshow('image',yellow_mask)
    cv2.imshow('color_image',img)
    cv2.waitKey(50)
    
    #cv2.imshow("window", img_cv2)
    #cv2.waitKey(30)
       
def listener():

    rospy.init_node('draw_lane', anonymous=True)

    rospy.Subscriber("raw_img", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()