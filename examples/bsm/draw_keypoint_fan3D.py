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
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)



print(bsm.model['shapePC'].shape)
print("triangles shape", bsm.triangles.shape)
# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit

X_ind = bsm.kpt_ind # index of keypoints in 3DMM. fixed.


# ------ light setup
print("nver", bsm.nver)
colors = np.tile([0.8,0.8,0.8],(int(bsm.nver),1))
light_intens = np.array([[1,1,1]])
light_pos = np.array([[0, 0, 300]])





# --- 2. video setup
#cap = cv2.VideoCapture("../Data/videoplayback.mp4")
#fps = cap.get(cv2.CAP_PROP_FPS)
#w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

#for img in images:
# for cnt, img in enumerate(images)

imgs = []
xl = []

for i in range(20):
    path = "../../../facewarehouse_data/Tester_39/TrainingPose/pose_"
    path = path + str(i) + ".png"
    img = io.imread(path)
    imgs.append(img)

#img = io.imread("../Data/qtest1.jpg")
#imgs.append(img)
#img = io.imread("../Data/qtest2.jpg")
#imgs.append(img)

for idx, img in enumerate(imgs):
    img_target = img.copy()[:,:,:3]
    img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w= img_gray.shape[:2]
    #faces = detector(img_gray,0)
    #print("img shape", img_target.shape)
    x = np.zeros([68,2])
    faces = fa.get_landmarks(img_target) 
    if faces is None:
        print("image {} no face detected".format(idx))
        continue
    elif len(faces) != 0:
        for i in range(len(faces)):
            landmarks = faces[i]
            #print("landmarks shape", faces[i].shape)
          
            for idkp, point in enumerate(landmarks):
                # position of 68 key points
                pos = (int(point[0]), int(point[1]))       # notice the size of w:[0,0] h;[0,1]
                x[idkp] = pos
    
                rr, cc = draw.circle_perimeter(pos[1], pos[0], 2)
                draw.set_color(img_target, [rr,cc], [0 ,0, 233])
    else:
        print("image {} no face detected".format(idx))
        continue
    x[:,0] = x[:, 0] - w / 2.0
    x[:,1] = h / 2.0 - x[:, 1] - 1
    xl.append(x)
    print("idx", idx)
    #img_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)
    cv2.imwrite("pose39/kpt_face_{}.jpg".format(idx), img_target)
#fit mesh
print("{} faces detected".format(len(xl)))
#expPC = bsm.fit_specific_blendshapes(xl, X_ind, max_iter = 3)
#print("fitted_info",ret)
#fitted_vertices = np.float32(bsm.generate_vertices(fitted_sp, fitted_ep))
#np.savetxt("f_ep", fitted_ep)
#fitted_vertices = np.reshape(bsm.model['expPC'][:,0], [int(3), int(len(bsm.model['expMU'])/3)], 'F').T
#fitted_vertices += np.reshape(bsm.model['expMU'], [int(3), int(len(bsm.model['expMU'])/3)], 'F').T
#np.savetxt('qz/wid_with_update.out', wid)


#obj = objloader.obj.objloader('pose_0.obj')

#for i in range(47):
#    vert = expPC[:,i]
#    obj.vertices = vert
#    obj.save('zzw/exp_{}.obj'.format(i))

#
#
#
#transformed_vertices = bsm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
#lit_colors = mesh.light.add_light(transformed_vertices, bsm.triangles, colors, light_pos, light_intens)
#image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
#
#
#
#
#
##verify fitted parameters
#
#fitted_image = mesh.render.render_colors(image_vertices, bsm.triangles, lit_colors, h, w)
#fitted_image = np.minimum(np.maximum(fitted_image, -1), 1)
#fitted_image = (fitted_image * 255.0).astype('u1')        
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
#out_img = fitted_image
#
#cv2.imwrite("kpt_face.jpg", img_target)
#cv2.imwrite("fit_mesh.jpg", out_img)
## io.imsave('fit.jpg', fitted_image)
#
