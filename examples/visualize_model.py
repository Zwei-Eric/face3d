''' Simple example of pipeline
3D obj(process) --> 2d image
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh

from face3d.morphable_model import MorphabelModel

bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')

h = w = 256

save_folder = 'results/visualize'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)



# -- here use colors to represent the texture of face surf
colors = np.tile([0.8,0.8,0.8],(int(bfm.nver),1))
# set lights
light_positions = np.array([[0, 0, 300]])
light_intensities = np.array([[1, 1, 1]])

shapePC = bfm.model['shapePC']
shapeMU = bfm.model['shapeMU']
print('shapePC shape', shapePC.shape)
print('shapeMU shape', shapeMU.shape)
for i in range(shapePC.shape[1]):
    vertices = shapeMU
    pc = shapePC[:,i].reshape(-1,1)
    #print('PC shape', pc.shape)
    #print(shapeMU[0], pc[0])
    vertices =  shapeMU + pc * 1e06 
    vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T
    print(vertices[0][0])
    # scale. target size=180 for example
    s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
    # rotate 30 degree for example
    R = mesh.transform.angle2matrix([0, 0, 0]) 
    # no translation. center of obj:[0,0]
    t = [0, 0, 0]

    transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)
    lit_colors = mesh.light.add_light(transformed_vertices, bfm.triangles, colors, light_positions, light_intensities)
    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
    fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, lit_colors, h, w)
    
    #fitted_image = np.around(fitted_image, decimals = 4)
    fitted_image = np.minimum(np.maximum(fitted_image, -1), 1)
    io.imsave('{}/shape_base_{}.jpg'.format(save_folder, i), fitted_image)

