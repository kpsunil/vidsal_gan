import numpy as np
import os
import h5py
import cv2
import multiprocessing
from utils import ProgressBar
video_dir ="../../../../scratch/kvg245/vidsal_gan/vidsal_gan/data/savam/gaussian_vizualizations/"

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
        frame = cv2.resize(frame, (224, 224)).astype(np.float32)
	print i
        frames.append(np.asarray(frame))
        if ret == False:
            break
        i+=1

    frames = np.array(frames)#convert to (num_frames, width, height, depth)
    
    print ("Frames chosen")
    print ("Length of video %d" % frames.shape[0])
    with h5py.File(video_dir+'video_data.h5', 'a') as hf:
        hf.create_dataset(filename,  data=frames)
    progress.current+=1
    progress()
    return video_dir+filename


c = [6, 11, 13, 22, 23, 26, 30, 35, 41]
h = h5py.File(video_dir+'video_data.h5')
video_list = [x for x in os.listdir(video_dir) if x.endswith('.avi') and x not in h.keys()]
print [x[:3]for x in video_list]
h.close()
progress = ProgressBar(len(video_list))

p = multiprocessing.Pool(1)
p.map(get_video_frames,video_list)
progress.done()                                            
