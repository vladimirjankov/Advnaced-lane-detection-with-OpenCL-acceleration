# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pyopencl as cl
from scipy.misc import  imsave
from support_functions import *


## nema potrebe nista preradjivati
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    a = int(img_ref.shape[0]-(level+1)*height)
    b = int(img_ref.shape[0]-level*height)
    c = max(0,int(center-width/2))
    d = min(int(center+width/2),img_ref.shape[1])
    output[a:b,c:d] = 1
    return output


## nema potrebe nista preradjivati

def pt_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    a = int(img_ref.shape[0]-(level+1)*height)
    b = int(img_ref.shape[0]-level*height)
    c = max(0,int(center-width/2))
    d = min(int(center+width/2),img_ref.shape[1])
    output[a:b,c:d] = img_ref[a:b,c:d]
    return output.sum()

def my(image, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    a = int(3*image.shape[0]/6)
    b = int(image.shape[1]/2)
    
    ii = image[a:,:b]
    l_sum = np.sum(image[a:,:b], axis=0)
    
    convl = np.convolve(window,l_sum) #paralelizuj
    l_center = np.argmax(convl)-window_width/2
    c = int(3*image.shape[0]/6)
    d = int(image.shape[1]/2)
    r_sum = np.sum(image[c:,d:], axis=0)
    convr = np.convolve(window,r_sum) # paralelizuj
    r_center = np.argmax(convr)-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    # window_centroids.append((l_center,r_center))
    # print(window_centroids)
    # return window_centroids
    # Go through each layer looking for max pixel locations
    for level in range((int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        a = int(image.shape[0]-(level+1)*window_height)
        b = int(image.shape[0]-level*window_height)
        m = int(image.shape[1] / 2)
        n = m
        # print(a, b, m, n)
        image_layer_l = np.sum(image[a:b,:m], axis=0)
        image_layer_r = np.sum(image[a:b,n:], axis=0)
        conv_signal_l = np.convolve(window, image_layer_l)
        conv_signal_r = np.convolve(window, image_layer_r)
        # print(image_layer)
        '''
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        '''
        l_center = np.argmax(conv_signal_l)
        '''
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        '''
        r_center = m + np.argmax(conv_signal_r)
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        
    return window_centroids

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    a = int(3*image.shape[0]/4)
    b = int(image.shape[1]/2)
    
    ii = image[a:,:b]
    l_sum = np.sum(image[a:,:b], axis=0)
    
    convl = np.convolve(window,l_sum)
    l_center = np.argmax(convl)-window_width/2
    c = int(3*image.shape[0]/4)
    d = int(image.shape[1]/2)
    r_sum = np.sum(image[c:,d:], axis=0)
    convr = np.convolve(window,r_sum)
    r_center = np.argmax(convr)-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    # print(window_centroids)
    # return window_centroids
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        a = int(image.shape[0]-(level+1)*window_height)
        b = int(image.shape[0]-level*window_height)
        image_layer = np.sum(image[a:b,:], axis=0)
        print (a, b, image_layer)
        conv_signal = np.convolve(window, image_layer)
        print(conv_signal)
        # print(image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def get_lane_points_and_windows(warped):

    # window settings
    window_width = 30 
    window_height = 40 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    mid = int(warped.shape[1] / 2)
    window_centroids = my(warped, window_width, window_height, margin)
    # window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
        
    # If we found any window centers
    if len(window_centroids) > 0:

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            lsum = pt_mask(window_width,window_height,warped,window_centroids[level][0],level)
            rsum = pt_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            
            if window_centroids[level][0] < 0.67 * mid and lsum > 50:
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            if window_centroids[level][1] > 1.33  * mid and rsum > 50:
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
            
            '''
            if lsum > 20:
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            if rsum > 20:
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
            '''
            # l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            # r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
            # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.uint8(np.dstack((warped, warped, warped))*255) # making the original road pixels 3 color channels
        output_img = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output_img = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    
    return output_img, l_points, r_points