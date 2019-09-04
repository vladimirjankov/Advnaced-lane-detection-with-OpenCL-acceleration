from lane_detection_functions import *
from poly_fit_functions import *
import numpy as np
import cv2
import pyopencl as cl
from scipy.misc import  imsave
from support_functions import *


def pipeline_final(img,mtx,dist,M,M_inv): 
    '''
    The complete pipeline for processing the video frames
    Takes the input image and draws the green zone on top of it
    '''
    # Use the previously defined pipeline method for getting a binary image
    result = pipeline(img)
    
    # Create warped images: binary and original 
    _, binary_warped = undistort_and_warp(result, mtx, dist, M)
    orig_undist, orig_warped = undistort_and_warp(img, mtx, dist, M)

    # Get the points for fitting the polynomial
    _, l_points, r_points = get_lane_points_and_windows(binary_warped)
   
    # Get the zone image
    # For single frames, use fit_poly, not fit_poly_past
    # zone_img = fit_poly(binary_warped, l_points.astype(int), r_points.astype(int))
    zone_img, left_fit, right_fit = fit_poly_past(binary_warped, l_points.astype(int), r_points.astype(int),img , orig_warped)
     
    # Get curvature and vehicle position
    left_curv, right_curv, vehicle_pos = get_curvature_and_vehicle_position(binary_warped.shape, left_fit, right_fit)
    
    # Warp the zone back to the priginal perspective using M_inv
    zone_warped = cv2.warpPerspective(zone_img, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)
    
    # Add the zone to the original undistorted image
    result = cv2.addWeighted(orig_undist, 1, zone_warped, 0.3, 0)
           
    # Add the text for curvature and vehicle position
    add_radius_and_position_to_img(result, left_curv, right_curv, vehicle_pos)
    
    return result# -*- coding: utf-8 -*-

