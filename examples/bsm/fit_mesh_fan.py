''' 3d morphable model example
3dmm parameters --> mesh 
fitting: 2d image + 3dmm -> 3d face
use dlib for feature landmark detection
'''
import os, sys
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io,draw,img_as_ubyte
from time import time
import matplotlib.pyplot as plt
sys.path.append('..')
sys.path.append('../..')
import face3d
from face3d import mesh
from face3d import mesh_numpy
from face3d import objloader

from face3d.morphable_model import MorphabelModel
import dlib
import timeit
import cv2
import face_alignment

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bsm = MorphabelModel('../Data/BSM/config', model_type = 'BSM')
print('init bsm model success')
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('../Data/dlib/shape_predictor_68_face_landmarks.dat')
#print(bsm.model['shapePC'].shape)
#print("triangles shape", bsm.triangles.shape)


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,  device = 'cuda')

# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit

X_ind = bsm.kpt_ind # index of keypoints in 3DMM. fixed.


# ------ light setup
print("nver", bsm.nver)
colors = np.tile([0.8,0.8,0.8],(int(bsm.nver),1))
light_intens = np.array([[1,1,1]])
light_pos = np.array([[0, 0, 300]])


#for img in images:
# for cnt, img in enumerate(images)


img = io.imread("../Data/zwtest3.jpg")
img_target = img.copy()[:,:,:3]
img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w= img_gray.shape[:2]
print("img shape", img_target.shape)
x = np.zeros([68,2])
faces = fa.get_landmarks(img_target)
if faces is None:
    print("image no face detected")
    exit()
elif len(faces) != 0:
    for i in range(len(faces)):
        landmarks = faces[i]
        for idx, point in enumerate(landmarks):
            pos = (int(point[0]), int(point[1]))
            x[idx] = pos
            rr, cc = draw.circle_perimeter(pos[1], pos[0], 2)
            draw.set_color(img_target, [rr,cc], [0 ,0, 233])
x[:,0] = x[:, 0] - w / 2.0
x[:,1] = h / 2.0 - x[:, 1] - 1

#fit mesh

fitted_ep, fitted_s, fitted_angles, fitted_t = bsm.fit(x, X_ind,  max_iter = 10, model_type = 'BSM')
print("fitted_s",fitted_s)
print("fitted_angles", fitted_angles)
print("fitted_t", fitted_t[:2])
fitted_vertices = np.float32(bsm.generate_vertices_bsm(fitted_ep))
np.savetxt("f_ep", fitted_ep)
#fitted_vertices = np.reshape(bsm.model['expPC'][:,0], [int(3), int(len(bsm.model['expMU'])/3)], 'F').T
#fitted_vertices += np.reshape(bsm.model['expMU'], [int(3), int(len(bsm.model['expMU'])/3)], 'F').T
#np.savetxt('verices', fitted_vertices)



obj = objloader.obj.objloader('pose_0.obj')

vert = fitted_vertices
obj.vertices = vert.flatten()
obj.save('fitted.obj')




transformed_vertices = bsm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
lit_colors = mesh.light.add_light(transformed_vertices, bsm.triangles, colors, light_pos, light_intens)
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)





#verify fitted parameters

fitted_image = mesh.render.render_colors(image_vertices, bsm.triangles, lit_colors, h, w)
fitted_image = np.minimum(np.maximum(fitted_image, -1), 1)
fitted_image = (fitted_image * 255.0).astype('u1')        
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
out_img = fitted_image

cv2.imwrite("kpt_face1.jpg", img_target)
cv2.imwrite("fit_mesh.jpg", out_img)
# io.imsave('fit.jpg', fitted_image)

