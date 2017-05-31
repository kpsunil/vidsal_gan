import numpy as np
import os
import h5py
import cv2
import multiprocessing
from utils import ProgressBar
video_dir ="../../../../scratch/skp442/videos/ftp.ivc.polytech.univ-nantes.fr/IRCCyN_IVC_Eyetracking_For_Stereoscopic_Videos/Videos/"

def get_video_frames(filename):
    global progress
    print ("Opening video!",video_dir+filename)
    capture = cv2.VideoCapture(video_dir+filename)
    print ("Video opened\nChoosing frames")
    frames = []
    i=0
    
    while(capture.isOpened()):

        capture.set(1,i)
        ret, frame = capture.read()
        if frame is None :
            break
        w, h, t = frame.shape
        frame = frame[0:w/2, 0:h]
        frame = cv2.resize(frame, (256, 256)).astype(np.float32)
    
    
        frames.append(np.asarray(frame))
        if ret == False:
            break
        i+=1

    frames = np.array(frames)#convert to (num_frames, width, height, depth)
    
    print ("Frames chosen")
    print ("Length of video %d" % frames.shape[0])
    with h5py.File(video_dir+'maps_data.h5', 'a') as hf:
        hf.create_dataset(filename,  data=frames)
    progress.current+=1
    progress()
    return video_dir+filename


video_list = [x for x in os.listdir(video_dir) if x.endswith('.avi') ]
progress = ProgressBar(len(video_list))

p = multiprocessing.Pool(1)
p.map(get_video_frames,video_list)
progress.done() 