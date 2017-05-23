import os
import random
import collections
import pickle

import numpy  as np
import tensorflow as tf
import h5py

from utils import *
from model import *

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

data_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/savam/'
output_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/'
input_file = 'gaussian_vizualizations/video_data.h5'
target_file = 'video_data/video_data.h5'
index_file = 'video_data/indices'

seed = 4

vid_dict = {0: 'v01_Hugo_2172_left.avi', 1: 'v02_Dolphin_131474_left.avi', 2: 'v03_StepUp_67443_left.avi', 3: 'v04_LIVE1_0_left.avi', 4: 'v05_LIVE2_0_left.avi', 5: 'v06_LIVE3_0_left.avi', 6: 'v07_Avatar_142222_left.avi', 7: 'v08_Dolphin_127156_left.avi', 8: 'v09_StepUpRevolution_119518_left.avi', 9: 'v10_VQEG01_0_left.avi', 10: 'v11_VQEG02_0_left.avi', 11: 'v12_VQEG03_0_left.avi', 12: 'v13_IntoTheDeep_36475_left.avi', 13: 'v14_Pirates_47241_left.avi', 14: 'v15_Sanctum_147749_left.avi', 15: 'v16_StepUp_17153_left.avi', 16: 'v17_SpiderMan_76686_left.avi', 17: 'v18_StepUp_76411_left.avi', 18: 'v19_Avatar_206134_left.avi', 19: 'v20_DriveAngry_2820_left.avi', 20: 'v21_Pirates_25246_left.avi', 21: 'v22_VQEG04_0_left.avi', 22: 'v23_VQEG05_0_left.avi', 23: 'v24_VQEG06_0_left.avi', 24: 'v25_Dolphin_17437_left.avi', 25: 'v26_Galapagos_14830_left.avi', 26: 'v27_UnderworldAwakening_69044_left.avi', 27: 'v28_VQEG10_0_left.avi', 28: 'v29_Avatar_46279_left.avi', 29: 'v30_Dolphin_79095_left.avi', 30: 'v31_DriveAngry_83142_left.avi', 31: 'v32_Dolphin_81162_left.avi', 32: 'v33_Hugo_87461_left.avi', 33: 'v34_VQEG07_0_left.avi', 34: 'v35_VQEG08_0_left.avi', 35: 'v36_VQEG09_0_left.avi', 36: 'v37_UnderworldAwakening_91276_left.avi', 37: 'v38_StepUp_82572_left.avi', 38: 'v39_Avatar_98125_left.avi', 39: 'v42_MSOffice_242_left.avi', 40: 'v43_Panasonic_373_left.avi', 41: 'video_data.h5', 42: 'video_data.h5'}


class batch_generator:
    """Provides batches of input and target files with training indices
       Also opens h5 datasets"""
    
    def __init__( self,batch_size = 8):
    	self.batch_size = batch_size
	self.index_data,self.target_data,self.input_data = self.open_files()
	self.current_epoch = None
        self.batch_index = None

    def open_files(self):
        """Opens h5 dataset files and loads index pickle"""
        with open(data_dir+index_file,'rb') as f:
	    index_data = random.shuffle(pickle.load(f))
        target_data = (h5py.File(data_dir+target_file,'r') 
        input_data = h5py.File(data_dir+input_file,'r')
        return index_data,target_data, input_data

    def create_batch(self, data_list):
	"""Creates and returns a batch of input and output data"""
	for data in data_list:
		data[0] 

    def get_batch_vec(self):
	"""Provides batch of data to process and keeps 
	track of current index and epoch"""
	
        if self.batch_index is None:
	    self.batch_index = 0
	    self.current_epoch = 0
	batch_list = self.create_batch(self.index_data[self.batch_index:self.batch_index + self.batch_size])
        if self.batch_index < self.batch_len+self.batch_size-1:
            self.batch_index += self.batch_size
        else:
            self.num_epoch += 1
            self.batch_index = 0

        return batch_list
    
def main():
    if not os.path.exists(output_dir):
	os.makedirs(output_dir)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    

	
