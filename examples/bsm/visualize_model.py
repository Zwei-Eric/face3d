''' Simple example of pipeline
3D obj(process) --> 2d image
'''
import os, sys
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io,draw,img_as_ubyte
import matplotlib.pyplot as plt
sys.path.append('..')
sys.path.append('../..')
import face3d
from face3d import mesh
from face3d import mesh_numpy
from face3d import objloader

from face3d.morphable_model import MorphabelModel
import dlib
import cv2

bsm = MorphabelModel('../Data/BSM/config', model_type = 'BSM')

print('init bsm model success')


save_folder = '../results/visualize'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

shapePC = bsm.model['expPC']
shapeMU = bsm.model['shapeMU'][:,0]
print('shapePC shape', shapePC.shape)
print('shapeMU shape', shapeMU.shape)

obj = objloader.obj.objloader('pose_0.obj')



for i in range(shapePC.shape[1]):
    pc = shapePC[:,i] 
    print(pc.shape)
    obj.vertices = shapeMU + pc
    obj.save('{}/exp_base_{}.obj'.format(save_folder, i))
    print("expression %d saved" %i )
