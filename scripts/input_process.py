import numpy as np
import os
import h5py
import cv2
import multiprocessing

video_dir ="../../../../scratch/kvg245/vidsal_gan/vidsal_gan/data/savam/video_data/"

def get_video_frames(filename):
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

        frames.append(np.asarray(frame))
        if ret == False:
            break
        i+=1

    frames = np.array(frames)#convert to (num_frames, width, height, depth)

    print ("Frames chosen")
    print ("Length of video %d" % frames.shape[0])
    with h5py.File(video_dir++'video_data.h5', 'a') as hf:
        hf.create_dataset(filename,  data=frames)

    return video_dir+filename



video_list = [x for x in os.listdir(video_dir) if x.endswith('.avi') ]
print video_list
p = multiprocessing.Pool(5)
print p.map(get_video_frames,video_list)                                            
