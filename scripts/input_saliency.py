import numpy as np
import os
import glob
import re
import h5py
import cv2
import multiprocessing
from utils import ProgressBar

video_dir = "/home/skp442/ftp.ivc.polytech.univ-nantes.fr/IRCCyN_IVC_Eyetracking_For_Stereoscopic_Videos/Fixation_Density_Maps/"

video_dir_map ="../../../../scratch/skp442/videos/ftp.ivc.polytech.univ-nantes.fr/IRCCyN_IVC_Eyetracking_For_Stereoscopic_Videos/Videos/"

def get_saliency_file(filename):
    print ("Opening video!",video_dir+filename)
    
    frames = []
    i=0

    os.chdir(video_dir+filename+'/')

    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    for infile in sorted(glob.glob('*.png'), key=numericalSort):
        frame = cv2.imread(infile,0)
        frame = cv2.resize(frame, (224, 224)).astype(np.float32)
        frames.append(np.asarray(frame))
       

    os.chdir('../../')

    frames = np.array(frames)#convert to (num_frames, width, height, depth)

    print ("Frames chosen")
    print ("Length of video %d" % frames.shape[0])
    with h5py.File(video_dir_map+'maps_data_saliency.h5', 'a') as hf:
        hf.create_dataset(filename,  data=frames)
    return video_dir+filename
    progress.current+=1
    progress()


video_list = [x for x in os.listdir(video_dir) if (x.endswith('325') or (x.endswith('400')))]


p = multiprocessing.Pool(1)
p.map(get_saliency_file,video_list)
progress.done() 