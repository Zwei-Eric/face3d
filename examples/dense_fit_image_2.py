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
import face3d
from face3d import mesh
from face3d import mesh_numpy
from face3d.morphable_model import MorphabelModel
import dlib
import timeit
# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Data/dlib/shape_predictor_68_face_landmarks.dat')


# --- 2. load frontal image 

img = io.imread("./Data/test.png")
img_gray = img_as_ubyte(io.imread("./Data/test.png", as_grey = True))
img_target = img.copy()
print("dtype", img_target.dtype.name)
print("dtype", img_gray.dtype.name)
h, w= img_target.shape[:2]
print("image size", h ,w)

faces = detector(img_gray,0)

x = np.zeros([68,2])

if len(faces) != 0:
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_target, faces[i]).parts()])
        for idx, point in enumerate(landmarks):
            # position of 68 key points
            pos = (point[0,0], point[0,1])          # notice the size of w:[0,0] h;[0,1]
            x[idx] = pos
            #cv2.circle(img_target, pos, 2, color=(139,0,0)
            
            rr,cc = draw.circle_perimeter(pos[1], pos[0], 1)
            draw.set_color(img_target, [rr,cc],[139,0,0])


# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit
X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.


x[:,0] = x[:, 0] - w / 2.0
x[:,1] = h / 2.0 - x[:, 1] - 1

# --- 3.fit processes

## fit mesh
start = timeit.default_timer()
fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)
end = timeit.default_timer()
print("mesh fitting time usage:", str(end-start))

fitted_vertices = np.float32(bfm.generate_vertices(fitted_sp, fitted_ep))
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
## get dense corresponding of face's pixels 
#print("vertices pos dtype", fitted_vertices.dtype)
#print("triangles shape", bfm.triangles.shape)
#print("vertices shape", fitted_vertices.shape)

start = timeit.default_timer()
_ , triangles_buffer, _ = mesh_numpy.render.rasterize_triangles(image_vertices, bfm.triangles, h, w)
end = timeit.default_timer()
print("rasterization time usage:", str(end - start))

x_colors = []
dense_X_ind = []
for u in range(h):
    for v in range(w):
        #u = h / 2 - u - 1
        #v = v - w / 2
        idx = triangles_buffer[u,v] 
        if idx != -1:
            pixel = img[u,v,:]
            x_colors.append(pixel)
            dense_X_ind.append(bfm.triangles[idx,0])
x_colors = np.array(x_colors)
x_colors = x_colors/ 255.0

dense_X_ind = np.array(dense_X_ind)

## fit color 

start = timeit.default_timer()
fitted_tp  = bfm.fit_color_v2(x_colors, dense_X_ind)  # use first point of triangle
end = timeit.default_timer()
print("texture fitting time usage:", str(end-start))

print("fitted tp", fitted_tp[:6])
fitted_colors = bfm.generate_colors(fitted_tp)
fitted_colors = np.minimum(np.maximum(fitted_colors, 0), 1)



# verify fitted parameters

fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, fitted_colors, h, w)
fitted_image = np.minimum(np.maximum(fitted_image, -1), 1)

#np.savetxt("fitted_vertices", fitted_vertices)

# ------------- print & show 
print('pose, fitted: \n', fitted_s, fitted_angles[0], fitted_angles[1], fitted_angles[2], fitted_t[0], fitted_t[1])

save_folder = 'results/dense_fit_image'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
#cv2.imwrite('{}/keypoint.png',img_target)
io.imsave('{}/keypoint.png'.format(save_folder),img_target)
#io.imsave('{}/generated.jpg'.format(save_folder), image)
io.imsave('{}/fitted.jpg'.format(save_folder), fitted_image)
np.savetxt("{}/dense_X_ind".format(save_folder), dense_X_ind, fmt = '%d')
np.savetxt("{}/triangle_buffer".format(save_folder), triangles_buffer, fmt = '%d')
