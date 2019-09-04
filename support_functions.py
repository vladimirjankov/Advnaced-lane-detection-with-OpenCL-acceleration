# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pyopencl as cl
from scipy.misc import  imsave
from get_binary_support_functions import *

local_work_group = [16,32]




def abs_sobel_thresh(img, orient='x', sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Calculate the Sobel gradient on the image, either in x or y direction
    """
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    mf = cl.mem_flags

    
    ctx = cl.Context([device])
    cpq = cl.command_queue_properties
    queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)
    
    
    
    prg = cl.Program(ctx, open('kernel_for_image.cl').read()).build()
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray)
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[1]))
    height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[0]))
    result_g1 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    
    
    #__kernel void sobelXFilter(__global float *img, __global float *result, __global int *width, __global int *height){
    # Calculate Sobel
    sob = None
    if orient == 'x':
        prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        
        #   sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    elif orient == 'y':
       # sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
    # Construct the scaled Sobel image and the binary
    
    sob = np.empty_like(gray)

    queue.finish()
    cl.enqueue_copy(queue,sob,result_g1)

    abs_sob = np.absolute(sob)
    scaled_sobel = np.uint8(255*abs_sob/np.max(abs_sob))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    queue.finish()
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Calculate the magnitude of the Sobel function on the image, 
    taking into account both x and y gradients
    """
    
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    mf = cl.mem_flags

    ctx = cl.Context([device])
    cpq = cl.command_queue_properties
    queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray)
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[1]))
    height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[0]))
    result_g1 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    result_g2 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    result_g3 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)

    
    prg = cl.Program(ctx, open('kernel_for_image.cl').read()).build()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Calculate x and y gradients, and get the magnitude by
    # taking both x an dy directions into account
    if sobel_kernel == 9:
        
        prg.sobelXFilter9x9(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        prg.sobelYFilter9x9(queue,gray.shape,local_work_group,img_g,result_g2,width_g,height_g)
    elif sobel_kernel == 3 :
        prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g2,width_g,height_g)
    else:
        prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g2,width_g,height_g)
    
    
    abs_sob_xy = np.empty_like(gray)
    sobx = np.empty_like(gray)
    soby = np.empty_like(gray)
    queue.finish()
    cl.enqueue_copy(queue,sobx,result_g1) 
    cl.enqueue_copy(queue,soby,result_g2)
    
#    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
#    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
  
    prg.magnitude(queue,gray.shape,local_work_group,result_g1,result_g2,result_g3,width_g)
    queue.finish()
    cl.enqueue_copy(queue,abs_sob_xy,result_g3)
 #  abs_sob_xy = np.sqrt(np.square(sobx) + np.square(soby))
    
    # Construct the scaled Sobel image and the binary
    scaled_sobel = np.uint8(255*abs_sob_xy/np.max(abs_sob_xy))
    sxbinary = np.zeros_like(scaled_sobel)
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return sxbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    Calculate the direction of the gradient using Sobel derivatives
    The function calculates arctan(sobely/sobelx)
    '''
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    mf = cl.mem_flags

    ctx = cl.Context([device])
    cpq = cl.command_queue_properties
    queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)
    
    # Convert to grayscale and calculate x and y gradients
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prg = cl.Program(ctx, open('kernel_for_image.cl').read()).build()
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray)
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[1]))
    height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[0]))
    result_g1 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    result_g2 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    
    if sobel_kernel == 9:      
        prg.sobelXFilter9x9(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        prg.sobelYFilter9x9(queue,gray.shape,local_work_group,img_g,result_g2,width_g,height_g)
    elif sobel_kernel == 3 :
        prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g2,width_g,height_g)
    else:
        prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g2,width_g,height_g)
    
   # sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
 #   soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    sobx = np.empty_like(gray)
    soby = np.empty_like(gray)
    queue.finish()
    cl.enqueue_copy(queue,sobx,result_g1)
    cl.enqueue_copy(queue,soby,result_g2)
    # Calculate the gradient direction
    grad_dir = np.arctan2(np.absolute(soby),np.absolute(sobx))
    
    # Construct the scaled Sobel image and the binary
    binary = np.zeros_like(grad_dir)
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary[(grad_dir >= thresh_min) & (grad_dir <= thresh_max)] = 1
    
    return binary

def mag_dir_threshold(img, ksize=3, magn_thresh=(20, 255), dir_thresh=(0.7, 1/3)):
    '''
    Calculate both magnitude and direction of the gradient, and return the binary image
    where both magnitude and direction are taken into account
    '''
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, mag_thresh=magn_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, mag_thresh=magn_thresh)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=magn_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)
    
    # Get the combined result
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1  
    
    # Another possible option:
    # combined[(gradx == 1) & (dir_binary == 1)] = 1
    
    return combined

def sobel_calc(img, sobel_kernel = 3, direction='x'):
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    mf = cl.mem_flags

    ctx = cl.Context([device])
    cpq = cl.command_queue_properties
    queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)
    
    # Convert to grayscale and calculate x and y gradients
    gray = img.astype(np.float32)
    prg = cl.Program(ctx, open('kernel_for_image.cl').read()).build()
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray)
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[1]))
    height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[0]))
    result_g1 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    
    if sobel_kernel == 9: 
        if direction == 'x':
            prg.sobelXFilter9x9(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        else:
            prg.sobelYFilter9x9(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
    elif sobel_kernel == 3 :
        if direction =='x' :
            prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        else:
            prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
    else:
        if direction == 'x': 
            prg.sobelXFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
        else:
            prg.sobelYFilter(queue,gray.shape,local_work_group,img_g,result_g1,width_g,height_g)
    
   # sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
 #   soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    sob = np.empty_like(gray)
  #  queue.finish()
    cl.enqueue_copy(queue,sob,result_g1)
    # Calculate the gradient direction
    
    
    return sob
    


def pipeline(img, s_thresh=(180, 255), sd_thresh=(0.7, 1.3), sx_thresh=(10, 130)):
    '''
    Pipeline to get the lane pixels by using L and S channel
    '''
    
    # img = np.copy(img)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x on the S channel
    sobel_s_x = sobel_calc(s_channel,9,'x') # Take the derivative in x
    abs_sobel_s_x = np.absolute(sobel_s_x) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_s_x = np.uint8(255*abs_sobel_s_x/np.max(abs_sobel_s_x))
    sxbinary = get_binary(scaled_sobel_s_x,sx_thresh)
    imsave('sxbinary.png',sxbinary )
    # Sobel x on the L channel
    sobel_l_x = sobel_calc(l_channel,9,'x')   # Take the derivative in x
    abs_sobel_l_x = np.absolute(sobel_l_x) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_l_x = np.uint8(255*abs_sobel_l_x/np.max(abs_sobel_l_x))
    lxbinary = get_binary(scaled_sobel_l_x,sx_thresh)
    imsave('lxbinary.png',lxbinary )
    # Calculate the gradient direction on the L channel
    
    sobx = sobel_calc(s_channel,3,'x')
    soby = sobel_calc(s_channel,3,'x')
    
    grad_dir = np.arctan2(np.absolute(soby),np.absolute(sobx))
    # abs_sob_xy = np.sqrt(np.square(sobx) + np.square(soby))
    # scaled_sobel = np.uint8(255*abs_sob_xy/np.max(abs_sob_xy))
    sdbinary = get_binary(grad_dir,sd_thresh)
    imsave('sdbinary.png',sdbinary )
    # Combine x gradient on S channel and direction gradient on L channel
    combined = np.zeros_like(grad_dir)
    combined[((sxbinary == 1) | (lxbinary == 1)) & (sdbinary == 1)] = 1
    
    # Threshold color channel
    s_binary = get_binary(s_channel,s_thresh)
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) #  * 255
    color_binary = np.dstack(( np.zeros_like(combined), combined, s_binary)) #  * 255
    
    # Create the final result
    final = np.zeros_like(s_channel)
    final[(combined == 1) | (s_binary == 1)] = 1
    
    return combined

def undistort_and_warp(img, mtx, dist, M):
    '''
    Return both the undistorted and undistorted+warped images
    '''
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist_warped = cv2.warpPerspective(undist, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return undist, undist_warped