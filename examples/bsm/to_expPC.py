import sys
sys.path.append('..')
sys.path.append('../..')
from face3d import objloader
import face3d
import numpy as np

obj = objloader.obj.objloader('pose_0.obj')
fname = "exp_"
expPC = []
for i in range(47):
    path = fname + str(i) + ".obj"
    obj = objloader.obj.objloader(path)
    vert =  obj.vertices
    expPC.append(vert)
expPC = np.asarray(expPC)
print(expPC.shape)
np.savetxt("exp_blendshapes.txt", expPC)
