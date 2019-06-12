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
bsm = MorphabelModel('../Data/BSM/config', model_type = 'BSM')
print('init bsm model success')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../Data/dlib/shape_predictor_68_face_landmarks.dat')


# --- 2. video setup
cap = cv2.VideoCapture('../Data/vtest.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
w , h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 



fourcc = cv2.VideoWriter_fourcc('M','J', 'P', 'G')
images = []



#while(cap.isOpened()):
    #io.imsave('src1.jpg', frame)
#    images.append(np.array(frame))

# img = io.imread("./Data/test.png")
#img_gray = img_as_ubyte(io.imread("./Data/test.png", as_grey = True))
# img_target = img.copy()
#print("dtype", img_target.dtype.name)
# :print("dtype", img_gray.dtype.name)
# h, w= images[0].shape[:2]
print("video size", h ,w)




writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (w,h))




# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit

X_ind = bsm.kpt_ind # index of keypoints in 3DMM. fixed.


# ------ light setup
print("nver", bsm.nver)
colors = np.tile([0.1,0.3,0.8],(int(bsm.nver),1))
light_intens = np.array([[1.2,1.2,1.2]])
light_pos = np.array([[0, 0, 300]])


print(len(images))
#for img in images:
while(1):
# for cnt, img in enumerate(images):
    ret, frame = cap.read()
    if not ret:
        break
    img = frame
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
    else:
        continue
    x[:,0] = x[:, 0] - w / 2.0
    x[:,1] = h / 2.0 - x[:, 1] - 1

## fit mesh
    fitted_ep, fitted_s, fitted_angles, fitted_t = bsm.fit(x, X_ind, max_iter = 3, model_type = 'BSM')
    fitted_vertices = np.float32(bsm.generate_vertices_bsm(fitted_ep))
    transformed_vertices = bsm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
    lit_colors = mesh.light.add_light(transformed_vertices, bsm.triangles, colors, light_pos, light_intens)
    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
    




# verify fitted parameters

    fitted_image = mesh.render.render_colors(image_vertices, bsm.triangles, lit_colors, h, w)
    fitted_image = np.minimum(np.maximum(fitted_image, -1), 1)
    fitted_image = (fitted_image * 255.0).astype('u1')        
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
    out_img = cv2.addWeighted(fitted_image, 1, img, 1, 0)
    #out_img = fitted_image
    writer.write(out_img)
    cv2.imshow("fit mesh", out_img)
    # io.imsave('fit.jpg', fitted_image)



cap.release()
writer.release()
