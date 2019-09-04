# -*- coding: utf-8 -*-

import os
import shutil
import cv2

def create_clean_dir(out_name):
    if os.path.exists(out_name):
        print("Removing the existing directory:", out_name)
        shutil.rmtree(out_name)
    os.makedirs(out_name)
    
# Specify the number of frames, all -1 for all frames
count_images = -1
def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    create_clean_dir(path_output_dir)
    vidcap = cv2.VideoCapture(video)
    count = 1
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.jpg') % count, image)
        else:
            break
        if count_images > 0 and count == count_images:
            break
        count += 1
    cv2.destroyAllWindows()
    vidcap.release()