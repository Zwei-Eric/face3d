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

from face3d.morphable_model import MorphabelModel
import dlib
import timeit
import cv2


# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('../Data/BFM/Out/BFM.mat')
print('init bfm model success')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../Data/dlib/shape_predictor_68_face_landmarks.dat')


# --- 2. video setup
cap = cv2.VideoCapture('../Data/vtest.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
w , h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 



fourcc = cv2. VideoWriter_fourcc('M','J', 'P', 'G')
images = []



while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    #io.imsave('src1.jpg', frame)
    images.append(np.array(frame))

# img = io.imread("./Data/test.png")
#img_gray = img_as_ubyte(io.imread("./Data/test.png", as_grey = True))
# img_target = img.copy()
#print("dtype", img_target.dtype.name)
# :print("dtype", img_gray.dtype.name)
h, w= images[0].shape[:2]
print("image size", h ,w)




writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (w,h))




# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit

X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.


# ------ light setup
print("nver", bfm.nver)
colors = np.tile([0,0.3,0.8],(int(bfm.nver),1))
light_intens = np.array([[1,1,1]])
light_pos = np.array([[0, 0, 300]])


print(len(images))
#for img in images:
for cnt, img in enumerate(images):
    print(cnt)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    io.imsave('src.jpg', img)
    faces = detector(img_gray,0)

    x = np.zeros([68,2])

    if len(faces) != 0:
        for i in range(len(faces)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
            for idx, point in enumerate(landmarks):
                # position of 68 key points
                pos = (point[0,0], point[0,1])          # notice the size of w:[0,0] h;[0,1]
                x[idx] = pos
                
    x[:,0] = x[:, 0] - w / 2.0
    x[:,1] = h / 2.0 - x[:, 1] - 1

## fit mesh
    fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)
    fitted_vertices = np.float32(bfm.generate_vertices(fitted_sp, fitted_ep))
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
    lit_colors = mesh.light.add_light(transformed_vertices, bfm.triangles, colors, light_pos, light_intens)
    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
    

## get dense corresponding of face's pixels 
#print("vertices pos dtype", fitted_vertices.dtype)
#print("triangles shape", bfm.triangles.shape)
#print("vertices shape", fitted_vertices.shape)

#    _ , triangles_buffer, _ = mesh_numpy.render.rasterize_triangles(image_vertices, bfm.triangles, h, w)
#
#    x_colors = []
#    dense_X_ind = []
#    for u in range(h):
#        for v in range(w):
#            idx = triangles_buffer[u,v] 
#            if idx != -1:
#                pixel = img[u,v,:]
#                x_colors.append(pixel)
#                dense_X_ind.append(bfm.triangles[idx,0])
#            
#    x_colors = np.array(x_colors)
#    x_colors = x_colors/ 255.0
#    dense_X_ind = np.array(dense_X_ind)
#
### fit color 
#
#    fitted_tp  = bfm.fit_color_v2(x_colors, dense_X_ind)  # use first point of triangle
#
#    fitted_colors = bfm.generate_colors(fitted_tp)
#    fitted_colors = np.minimum(np.maximum(fitted_colors, 0), 1)



# verify fitted parameters

    fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, lit_colors, h, w)
    fitted_image = np.minimum(np.maximum(fitted_image, -1), 1)
    fitted_image = (fitted_image * 255.0).astype('u1')        
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
    out_img = cv2.addWeighted(fitted_image, 0.98, img, 1, 0)
    writer.write(out_img)
    # io.imsave('fit.jpg', fitted_image)



cap.release()
writer.release()
