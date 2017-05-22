import os
import numpy
import h5py
import pickle
from utils import ProgressBar

lag = 3
num_frames = 4

file_dir = '../../../../scratch/kvg245/vidsal_gan/vidsal_gan/data/savam/video_data/'

files_list= [x for x in os.listdir(file_dir) if x.endswith('avi')]
print sorted(files_list)
print len(files_list)
data_list = []
print "reading file"
progress = ProgressBar(len(files_list),fmt=ProgressBar.FULL)
index_list = []
v = 0
with h5py.File(file_dir+'video_data.h5','r') as hf:
    for file in sorted(files_list):
    	progress.current+=1
	progress()
	data = hf[file][:]
	 

	
	index_list.append([])
        for f in range(data.shape[0]):
	   
            
            if f<num_frames*lag-lag:
                pass
	    else:
                index_list[v].append([])
                for n in range (num_frames):
                    index_list[v][f-num_frames*lag+lag].append(f-n*lag)
        v+=1
progress.done()
print "saving pickle"
with open(file_dir+'indices','w') as ifile:
    pickle.dump(index_list,ifile)
