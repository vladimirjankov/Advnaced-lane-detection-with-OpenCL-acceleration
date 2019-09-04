# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from collections import deque
import shutil
from support_functions import *
from get_binary_support_functions import *
from lane_detection_functions import *
from poly_fit_functions import *
from final_pipline import *
from Video_manipulation import *




all_img = glob.glob('camera_cal/calibration*.jpg')
print("Camera calibration images:", all_img)
img = mpimg.imread(all_img[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
img_size = gray.shape[::-1]
print("Grayscale image size:", img_size)
plt.imshow(img)




# Initialize object points and image points arrays
objpoints = []
imgpoints = []
processed_images = []
# Some images have 9x5 corners to detect, some 9x6
# We have to assume both scenarios for each image
objp_9x6 = np.zeros([6*9,3], np.float32)
objp_9x6[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp_9x5 = np.zeros([5*9,3], np.float32)
objp_9x5[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)


count_images_taken = 0
for img_name in all_img:
    img = mpimg.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # First try to find 9x6 corners, and if that doesn't work, then
    # try to find 9x5. 
    # If none of those work, which is the case for 2 images, skip them
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp_9x6)
        processed_images.append(img)
    else:
        ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp_9x5)
            processed_images.append(img)
    # Print image name and the detection flag, so that 
    # we know if the image was taken into account or not 
    if ret == True:
        count_images_taken += 1
    print("Detected corners in the image " + img_name + "? " + str(ret))
    
print("Added points from " + str(count_images_taken) + " images") 

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Show an example of an undistorted image
img = mpimg.imread("camera_cal/calibration5.jpg")
undist = cv2.undistort(img, mtx, dist, None, mtx)
out_img_path = os.path.join("output_images", "calibration_5_calibrated.jpg")
mpimg.imsave(out_img_path, undist)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistorted image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

test_img = 'test_images/test2.jpg'
img = mpimg.imread(test_img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
img_size = gray.shape[::-1]
plt.imshow(img)


gradx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, mag_thresh=(10, 255))
grady_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=3, mag_thresh=(10, 255))
# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(gradx_binary, cmap='gray')
ax2.set_title('Thresholded x gradient', fontsize=30)
ax3.imshow(grady_binary, cmap='gray')
ax3.set_title('Thresholded y gradient', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
gradx_img_path = os.path.join("output_images", "test_gradx.jpg")
mpimg.imsave(gradx_img_path, gradx_binary, cmap='gray')
grady_img_path = os.path.join("output_images", "test_grady.jpg")
mpimg.imsave(grady_img_path, grady_binary, cmap='gray')

# Calculate the magnitude of the gradient, both x and y are taken into account
mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=(20, 255))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded gradient magnitude', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
grad_mag_img_path = os.path.join("output_images", "test_grad_mag.jpg")
mpimg.imsave(grad_mag_img_path, mag_binary, cmap='gray')

# Calculate the direction of the gradient
dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded gradient direction', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
grad_dir_img_path = os.path.join("output_images", "test_grad_dir.jpg")
mpimg.imsave(grad_dir_img_path, dir_binary, cmap='gray')

## everything working fineee !!!! 
# change kernel in dir_threshold to 15x15

# Get the combined gradient magnitude and direction
combined = mag_dir_threshold(img)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Thresholded direction and magnitude of the gradient', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
grad_magdir_img_path = os.path.join("output_images", "test_grad_mag_dir.jpg")
mpimg.imsave(grad_magdir_img_path, combined, cmap='gray')

# Calculate the gray binary image
gray, gray_binary = get_gray_binary(img, thresh=(100,255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(gray, cmap='gray')
ax1.set_title('Gray', fontsize=30)
ax2.imshow(gray_binary, cmap='gray')
ax2.set_title('Gray binary', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
gray_binary_img_path = os.path.join("output_images", "test_gray_binary.jpg")
mpimg.imsave(gray_binary_img_path, gray_binary, cmap='gray')

# Display R, G, B components
R, G, B = get_rgb(img)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(R)
ax1.set_title('R', fontsize=30)
ax2.imshow(G)
ax2.set_title('G', fontsize=30)
ax3.imshow(B)
ax3.set_title('B', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
r_img_path = os.path.join("output_images", "test_r.jpg")
mpimg.imsave(r_img_path, R)
g_img_path = os.path.join("output_images", "test_g.jpg")
mpimg.imsave(g_img_path, G)
b_img_path = os.path.join("output_images", "test_b.jpg")
mpimg.imsave(b_img_path, B)


# Calculate binaries of R, G, B components
R_b, G_b, B_b = get_rgb_binary(img, mag_thresh_r=(120,256), mag_thresh_g=(100,255), mag_thresh_b=(150,255))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(R_b, cmap='gray')
ax1.set_title('R binary', fontsize=30)
ax2.imshow(G_b, cmap='gray')
ax2.set_title('G binary', fontsize=30)
ax3.imshow(B_b, cmap='gray')
ax3.set_title('B binary', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
r_binary_img_path = os.path.join("output_images", "test_r_binary.jpg")
mpimg.imsave(r_binary_img_path, R_b, cmap='gray')
g_binary_img_path = os.path.join("output_images", "test_g_binary.jpg")
mpimg.imsave(g_binary_img_path, G_b, cmap='gray')
b_binary_img_path = os.path.join("output_images", "test_b_binary.jpg")
mpimg.imsave(b_binary_img_path, B_b, cmap='gray')



# Display binary H, L, S components
H_b, L_b, S_b = get_hls_binary(img, mag_thresh_h=(40,150), mag_thresh_l=(120, 255), mag_thresh_s=(50,255))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(H_b, cmap='gray')
ax1.set_title('H binary', fontsize=30)
ax2.imshow(L_b, cmap='gray')
ax2.set_title('L binary', fontsize=30)
ax3.imshow(S_b, cmap='gray')
ax3.set_title('S binary', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
h_binary_img_path = os.path.join("output_images", "test_h_binary.jpg")
mpimg.imsave(h_binary_img_path, H_b, cmap='gray')
l_binary_img_path = os.path.join("output_images", "test_l_binary.jpg")
mpimg.imsave(l_binary_img_path, L_b, cmap='gray')
s_binary_img_path = os.path.join("output_images", "test_s_binary.jpg")
mpimg.imsave(s_binary_img_path, S_b, cmap='gray')

# Test on image test4.jpg
test_img = 'test_images/test4.jpg'
img = mpimg.imread(test_img)
result = pipeline(img)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(result, cmap='gray')
ax2.set_title('Processed image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
processed_img_path = os.path.join("output_images", "test_4_processed.jpg")
mpimg.imsave(processed_img_path, result, cmap='gray')

# Test on image test2.jpg
test_img = 'test_images/test2.jpg'
img = mpimg.imread(test_img)
result = pipeline(img)
'''
kernel = np.ones((3,3), np.uint8)
img_erosion = cv2.erode(result, kernel, iterations=1)
kernel = np.ones((5,5), np.uint8)
result2= cv2.dilate(img_erosion, kernel, iterations=1)
'''
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(result, cmap='gray')
ax2.set_title('Processed image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
processed_img_path = os.path.join("output_images", "test_2_processed.jpg")
mpimg.imsave(processed_img_path, result, cmap='gray')
print(img.shape)


# Undistort and warp the image
test_img = 'test_images/test2.jpg'
img = mpimg.imread(test_img)

# Specify the points to calculate the transform matrices
img_size = (img.shape[1], img.shape[0])
offset = 0
# src = np.float32([[500,470], [780,470], [1080,650], [200,650]])
# src = np.float32([[600,450], [750,450], [1200,700], [220,700]])
src = np.float32([[500,470], [780,470], [1080,650], [200,650]])
dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])

# Get both transform and inverse-transform matrices
M = cv2.getPerspectiveTransform(src,dst)
M_inv = cv2.getPerspectiveTransform(dst,src)

# Warp the images
result = pipeline(img)
undist = cv2.undistort(result, mtx, dist, None, mtx)
# binary_warped = cv2.warpPerspective(undist, M, (1280, 720), flags=cv2.INTER_LINEAR)
# orig_warped = cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_LINEAR)
_, binary_warped = undistort_and_warp(result, mtx, dist, M)
orig_undist, orig_warped = undistort_and_warp(img, mtx, dist, M)

# Plot results
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(orig_warped)
ax1.set_title('Original image warped', fontsize=30)
ax2.imshow(binary_warped, cmap='gray')
ax2.set_title('Binary image warped', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ow_img_path = os.path.join("output_images", "test_orig_warped.jpg")
mpimg.imsave(ow_img_path, orig_warped)
bw_img_path = os.path.join("output_images", "test_binary_warped.jpg")
mpimg.imsave(bw_img_path, binary_warped, cmap='gray')


# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
plt.plot(histogram)
hist_img_path = os.path.join("output_images", "test_hist.jpg")
plt.title('Number of pixels detected at the center line')
plt.savefig(hist_img_path)

# Test the function to generate windows over lane lines
output_img, l_points, r_points = get_lane_points_and_windows(binary_warped)

# Plot outputs
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(binary_warped, cmap='gray')
ax1.set_title('Binary image warped', fontsize=30)
ax2.imshow(output_img)
ax2.set_title('Binary image warped, with windows', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
bww_img_path = os.path.join("output_images", "test_binary_warped_win.jpg")
mpimg.imsave(bww_img_path, output_img)

left_fit_past  = deque([])
right_fit_past = deque([])
deq_size = 10

# Test the function that generates the zone
zone, left_fit, right_fit = fit_poly(binary_warped, l_points.astype(int), r_points.astype(int),img,orig_warped)
combined = cv2.addWeighted(orig_warped, 1, zone, 0.3, 0)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(zone)
ax1.set_title('Zone between lane lines', fontsize=30)
ax2.imshow(combined)
ax2.set_title('Zone and test image (warped) combined', fontsize=30)
zone_img_path = os.path.join("output_images", "test_zone.jpg")
mpimg.imsave(zone_img_path, zone)
zone_combined_img_path = os.path.join("output_images", "test_zone_combined.jpg")
mpimg.imsave(zone_combined_img_path, combined)

left_curv, right_curv, vehicle_pos = get_curvature_and_vehicle_position(img.shape, left_fit, right_fit, True)
print("Curvature in meters:", left_curv, 'm', right_curv, 'm')
print("Vehicle position from center:", vehicle_pos, 'm')

# Transform the image back to the original perspective by using the inverse tranform matrix
zone_warped = cv2.warpPerspective(zone, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)
result = cv2.addWeighted(orig_undist, 1, zone_warped, 0.3, 0)
add_radius_and_position_to_img(result, left_curv, right_curv, vehicle_pos)
    
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original image', fontsize=30)
ax2.imshow(result)
ax2.set_title('Original image with lanes detected', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
orig_detected_img_path = os.path.join("output_images", "test_orig_detected.jpg")
mpimg.imsave(orig_detected_img_path, result)

left_fit_past  = deque([])
right_fit_past = deque([])

# Test the pipeline
test_img = mpimg.imread('test_images/test2.jpg')
# test_img = mpimg.imread('frames/421.jpg')
result = pipeline_final(test_img,mtx,dist,M,M_inv) # , mtx, dist, M, M_inv)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(result)
ax2.set_title('Original image with lanes detected', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)




video_to_frames("project_video.mp4", "project_video")
print("Done storing images")



# Uncomment for batch analysis of different history sizes
# sizes = [1, 2, 4, 6, 8, 10]
sizes = [9]

#input_dir = "frames_challenge2"
#dir_prefix = "Done storing images2"  
input_dir = "project_video"
dir_prefix = "project_video2"  
#input_dir = "frames_harder_challenge"
#dir_prefix = "frames_harder_challen2ge123"    

for sz in sizes:
    out_dir = dir_prefix + str(sz)
    create_clean_dir(out_dir) 
    
    print("Input directory:", input_dir)
    print("Output directory:", out_dir)

    left_fit_past  = deque([])
    right_fit_past = deque([])
    deq_size = sz

    count_images = len(os.listdir(input_dir))

    # Uncomment to create a shorter video
    # count_images = 20
    isorted = []
    step_perc = 5
    next_perc = step_perc
    for i in range(1, count_images):
        curr_perc = 100 * i / count_images
        if curr_perc >= next_perc:
            print("At " + str(curr_perc) + "%")
            next_perc += step_perc
        file_name = str(i) + ".jpg"
        in_file = os.path.join(input_dir, file_name)
        out_file = os.path.join(out_dir, file_name)
        print(out_file)
        iresult = np.zeros(img.shape)
        iimg = mpimg.imread(in_file)
        try:
            iresult = pipeline_final(iimg,mtx,dist,M,M_inv)
        except:
            print("Nije proslo za frame : " +str(i))
        mpimg.imsave(out_file, iresult)
        isorted.append(out_file)
print("Done storing images")

import os
from moviepy.editor import ImageSequenceClip
img_dir = "project_video29"
images = os.listdir(img_dir)
print(len(images))
image_list = []
count = len(images)
for i in range(1,count):
    file_name = str(i) + ".jpg"
    out_file = os.path.join(img_dir, file_name)
    image_list.append(out_file)
video_file = img_dir + "_video.mp4"
fps = 20
print("Creating video {}, FPS={}".format(video_file, fps))
clip = ImageSequenceClip(image_list, fps=fps)
clip.write_videofile(video_file)
print("Done creating video")























