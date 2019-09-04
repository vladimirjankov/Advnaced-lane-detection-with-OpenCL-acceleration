# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pyopencl as cl
from scipy.misc import  imsave
import matplotlib.pyplot as plt
from collections import deque



def calc_avg_fit(fit_past):
    fit_0, fit_1, fit_2 = 0, 0, 0
    count = len(fit_past)
    for i in range(count):
        fit_0 += fit_past[i][0]
        fit_1 += fit_past[i][1]
        fit_2 += fit_past[i][2]
    return fit_0/count, fit_1/count, fit_2/count

def fit_poly_past(binary_warped, l_points, r_points,img,orig_warped):
    '''
    Fit the polynomials through the lane line points, and return the 
    zone between the lane lines
    '''
    deq_size = 10
    leftx = l_points.nonzero()[1]
    lefty = l_points.nonzero()[0]
    rightx = r_points.nonzero()[1]
    righty = r_points.nonzero()[0]
    left_fit_past  = deque([])
    right_fit_past = deque([])
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    print("This iteration left fit:", left_fit)
    print("This iteration right fit:", right_fit)
    
    left_fit_past.append(left_fit)
    right_fit_past.append(right_fit)
    if len(left_fit_past) > deq_size:
        _ = left_fit_past.popleft()
    if len(right_fit_past) > deq_size:
        _ = right_fit_past.popleft()
    left_fit_0, left_fit_1, left_fit_2 = calc_avg_fit(left_fit_past)
    right_fit_0, right_fit_1, right_fit_2 = calc_avg_fit(right_fit_past)
    
    print("This iteration used left fit:", left_fit_0, left_fit_1, left_fit_2, len(left_fit_past))
    print("This iteration used right fit:", right_fit_0, right_fit_1, right_fit_2, len(right_fit_past))
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit_0*ploty**2 + left_fit_1*ploty + left_fit_2
    right_fitx = right_fit_0*ploty**2 + right_fit_1*ploty + right_fit_2
    yvals = np.linspace(0, img.shape[0], num=img.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Create the image
    zone = np.zeros_like(orig_warped)
    cv2.fillPoly(zone, np.int_([pts]), (0,255, 0))
    cv2.polylines(zone, np.array([pts_left], dtype=np.int32), False,(255,0,0),thickness = 15)
    cv2.polylines(zone, np.array([pts_right], dtype=np.int32), False,(0,0,255),thickness = 15)

    return zone, (left_fit_0, left_fit_1, left_fit_2), (right_fit_0, right_fit_1, right_fit_2)

def fit_poly(binary_warped, l_points, r_points,img,orig_warped):
    '''
    Fit the polynomials through the lane line points, and return the 
    zone between the lane lines
    '''
    leftx = l_points.nonzero()[1]
    lefty = l_points.nonzero()[0]
    rightx = r_points.nonzero()[1]
    righty = r_points.nonzero()[0]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    yvals = np.linspace(0, img.shape[0], num=img.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Create the image
    zone = np.zeros_like(orig_warped)
    cv2.fillPoly(zone, np.int_([pts]), (0,255, 0))
    cv2.polylines(zone, np.array([pts_left], dtype=np.int32), False,(255,0,0),thickness = 15)
    cv2.polylines(zone, np.array([pts_right], dtype=np.int32), False,(0,0,255),thickness = 15)

    return zone, left_fit, right_fit

def get_poly_fit(l_points, r_points):
    '''
    Fit the polynomials through the lane line points, and return the 
    zone between the lane lines
    '''
    leftx = l_points.nonzero()[1]
    lefty = l_points.nonzero()[0]
    rightx = r_points.nonzero()[1]
    righty = r_points.nonzero()[0]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit



# Calculate the vehicle position and the radius
def add_radius_and_position_to_img(img, left_curv, right_curv, vehicle_pos):
    '''
    Add the text to the image, describing the vehicle position and the road curvature
    '''
    position_string = "Vehicle position from center: {:.3f} m".format(vehicle_pos) 
    curv_string = "Road curvature: {:.1f} m".format((left_curv + right_curv) / 2) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    colort = (0,255,255)
    
    cv2.putText(img, position_string, (20,40), font, 1, colort, 2)
    cv2.putText(img, curv_string, (20,80), font, 1, colort, 2)
    
    return

# Here, we want to calculate the radius of the curature
# We will use the params of the polynomial which is drawn on the image above
def get_curvature_and_vehicle_position(img_shape, left_fit, right_fit, do_plot=False):
    '''
    Calculate curvature of the lanes, at the position closest to the vehicle
    '''
        
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Generate y data
    ploty = np.linspace(0, img_shape[0], num=img_shape[0]) 
    
    leftx = np.array([left_fit[2] + (y**2)*left_fit[0] + y * left_fit[1] + np.random.randint(-50, high=51) for y in ploty])
    rightx = np.array([right_fit[2] + (y**2)*right_fit[0] + y * right_fit[1] + np.random.randint(-50, high=51) for y in ploty])

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if do_plot == True:

        # Plot up the fake data
        mark_size = 3
        plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        
    # Define y-value where we want radius of curvature
    # Use bottom of the image, where the vehicle is
    y_eval = np.max(ploty)
    
    # Calculate the camera position relative to the center of the image
    left_cam_pos_pix = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_cam_pos_pix = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    mid_point_pix = int(img_shape[1]/2)
    cam_dist_from_center_pix = (left_cam_pos_pix + right_cam_pos_pix) / 2 - mid_point_pix 
    cam_dist_from_center_m = cam_dist_from_center_pix * xm_per_pix
    
    # print(left_cam_pos_pix, right_cam_pos_pix, cam_dist_from_center_pix, cam_dist_from_center_m)
    
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad, cam_dist_from_center_m