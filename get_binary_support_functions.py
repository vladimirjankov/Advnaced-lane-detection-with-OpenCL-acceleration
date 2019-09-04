# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pyopencl as cl
from scipy.misc import  imsave

local_work_group = [16,32]


def get_rgb_binary(image, mag_thresh_r=(30, 130), mag_thresh_g=(30, 130), mag_thresh_b=(30, 130)):
    '''
    Returns thresholded R, G, B components of the image
    '''
    R = image[:,:,2].astype(np.float32)    
    G = image[:,:,1].astype(np.float32)     
    B = image[:,:,0].astype(np.float32) 
    
    R_b= get_binary(R,mag_thresh_r)
    G_b= get_binary(G,mag_thresh_g)
    B_b= get_binary(B,mag_thresh_b)

    imsave('R.png',R_b)
    imsave('G.png',G_b)
    imsave('B.png',B_b)


    return (R_b, G_b, B_b)

def get_rgb(image):
    '''
    Returns R, G, B components of the image
    '''
    R = image[:,:,2] 
    G = image[:,:,1]    
    B = image[:,:,0]
    
    zero_channel = np.zeros_like(R) # create a zero color channel
    R = np.array(cv2.merge((R,zero_channel,zero_channel)),np.uint8)
    G = np.array(cv2.merge((zero_channel,G,zero_channel)),np.uint8)
    B = np.array(cv2.merge((zero_channel,zero_channel,B)),np.uint8)


    return (R, G, B)

def get_hls(image):
    '''
    Returns thresholded H, L, S components of the image
    '''
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    zero_channel = np.zeros_like(H) # create a zero color channel
    H = np.array(cv2.merge((H,zero_channel,zero_channel)),np.uint8)
    L = np.array(cv2.merge((zero_channel,L,zero_channel)),np.uint8)
    S = np.array(cv2.merge((zero_channel,zero_channel,S)),np.uint8)
    
    return (H, L, S)

def get_hls_binary(image, mag_thresh_h=(30, 130), mag_thresh_l=(30, 130), mag_thresh_s=(30, 130)):
    '''
    Returns H, L, S components of the image
    '''
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    H_b= get_binary(H,mag_thresh_h)
    L_b= get_binary(L,mag_thresh_l)
    S_b= get_binary(S,mag_thresh_s)
    
    
    return (H_b, L_b, S_b)

def get_gray_binary(image, thresh=(100, 200)):
    '''
    Returns thresholded grayscale image
    '''
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    print(device)
    mf = cl.mem_flags
    
    ctx = cl.Context([device])
    cpq = cl.command_queue_properties
    queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)
    
    prg = cl.Program(ctx, open('kernel_for_binarization.cl').read()).build()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray)
    t1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(thresh[0]))
    t2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(thresh[1]))
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[1]))
    result_g1 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    prg.get_gray_binary(queue,gray.shape,local_work_group,img_g,result_g1,t1,t2,width_g)
    
    
    binary = np.empty_like(gray)
    queue.finish()
    cl.enqueue_copy(queue,binary,result_g1)
    
    
    return gray, binary


def get_binary(image, thresh=(100, 200)):
    '''
    Returns thresholded grayscale image
    '''
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    mf = cl.mem_flags
    
    ctx = cl.Context([device])
    cpq = cl.command_queue_properties
    queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)
    
    prg = cl.Program(ctx, open('kernel_for_binarization.cl').read()).build()
    gray = image.astype(np.float32)
    
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray)
    t1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(thresh[0]))
    t2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(thresh[1]))
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(gray.shape[1]))
    result_g1 = cl.Buffer(ctx, mf.READ_WRITE, gray.nbytes)
    prg.get_gray_binary(queue,gray.shape,local_work_group,img_g,result_g1,t1,t2,width_g)
    
    
    binary = np.empty_like(gray)
 #   queue.finish()
    cl.enqueue_copy(queue,binary,result_g1)
    
    
    return binary
