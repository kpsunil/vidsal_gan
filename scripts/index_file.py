import os
import numpy
import h5py
import pickle

lag = 3
num_frames = 4

file_dir = '../data/ftp.ivc.polytech.univ-nantes.fr/IRCCyN_IVC_Eyetracker_SD_2009_12/H264_Streams/'

files_list= [x for x in os.listdir(file) if x.endswith('.h5')]
data_list = []
for file in files_list:
    with h5py.File(file_dir+file,'r') as hf:
        data_list.append(hf['frames'][:])
index_list = []
h = h5py.File('inputs.h5', 'w')
for v in range(len(data_list)):
    index_list.append([])
    h.create_dataset(str(v),data = data_list[v])
    for f in range(data_list[v].shape[0]):
        if f<num_frames*lag-1:
            pass
        for n in range (num_frames):
            index_list[v].append([f-n*lag])
with open('indices','w') as ifile:
    pickle.dump(index_list,ifile)
