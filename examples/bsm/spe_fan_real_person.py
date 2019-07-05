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
import copy
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



folder = 'xd_3D/'

# ------ light setup
print("nver", bsm.nver)
colors = np.tile([0.8,0.2,0.2],(int(bsm.nver),1))
light_intens = np.array([[1.0 ,1.0 ,1.0]])
light_pos = np.array([[0, 0, 1000]])
light_pos2 = np.array([[-700, 0, 1000]])
light_pos3 = np.array([[700,0 , 1000]])
light_pos4 = np.array([[0, -700 , 1000]])





# --- 2. video setup
#cap = cv2.VideoCapture("../Data/videoplayback.mp4")
#fps = cap.get(cv2.CAP_PROP_FPS)
#w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

#for img in images:
# for cnt, img in enumerate(images)

imgs = []
xl = []

for i in range(60):
    #path = "../../../facewarehouse_data/Tester_39/TrainingPose/pose_"
    path = "../Data/exp_input/exp_input"
    path = path + str(i) + ".jpg"
    img = io.imread(path)
    imgs.append(img[:,:,:3])

#img = io.imread("../Data/qtest1.jpg")
#imgs.append(img)
#img = io.imread("../Data/qtest2.jpg")
#imgs.append(img)
imgs_c = copy.deepcopy(imgs)
for idx, img in enumerate(imgs):
    h, w= img.shape[:2]
    #faces = detector(img_gray,0)
    #print("img shape", img_target.shape)
    x = np.zeros([68,2])
    faces = fa.get_landmarks(img) 
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
                draw.set_color(img, [rr,cc], [233 ,0, 233])
    else:
        print("image {} no face detected".format(idx))
        continue
    x[:,0] = x[:, 0] - w / 2.0
    x[:,1] = h / 2.0 - x[:, 1] - 1
    xl.append(x)
    #cv2.imwrite("pose39_3D/face_kpt{}.jpg".format(idx), img_target)
#fit mesh
print("{} faces detected".format(len(xl)))
expPC ,wid, wexpl, sl, Rl, tl= bsm.fit_specific_blendshapes(xl, X_ind, max_iter = 3)
#print("fitted_info",ret)
#fitted_vertices = np.float32(bsm.generate_vertices(fitted_sp, fitted_ep))
#np.savetxt("f_ep", fitted_ep)
#fitted_vertices = np.reshape(bsm.model['expPC'][:,0], [int(3), int(len(bsm.model['expMU'])/3)], 'F').T
#fitted_vertices += np.reshape(bsm.model['expMU'], [int(3), int(len(bsm.model['expMU'])/3)], 'F').T
#np.savetxt('qz/wid_with_update.out', wid)
X = bsm.show_fitting_result(X_ind,   sl, Rl, tl,  wid, wexpl)

X = np.array(X)
X[:,:,0] = X[:,:, 0] + w / 2.0
X[:,:,1] = h / 2.0 - X[:,:, 1] - 1
X = X.astype(np.int32)
obj = objloader.obj.objloader('pose_0.obj')
for i in range(47):
    vert = expPC[:,i]
    obj.vertices = vert
    obj.save('xd_3D/exp_{}.obj'.format(i))


for idx, img in enumerate(imgs):
    for pos in X[idx]:
        rr, cc = draw.circle_perimeter(pos[1], pos[0], 2)
        draw.set_color(img, [rr,cc], [0 ,233, 0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path = folder + "fitted_keypoint" + str(idx) + ".jpg"
    cv2.imwrite(path, img)
    path = folder + "fitted_mesh" + str(idx) + ".obj"
    vert = bsm.generate_fitting_mesh(wid, wexpl[idx])
    
    # normalize
    #pmax = vert.max()
    #pmin = vert.min()
    #vert = (vert - pmin) / (pmax - pmin)
    
    # save mesh to .obj file
    obj.vertices = vert
    obj.save(path)
    
    vert = np.reshape(vert, [int(3), int(len(vert)/ 3)], 'F').T 
    trans_vert = bsm.similarity_transform(vert, sl[idx], Rl[idx], tl[idx])
    lit_colors1 = mesh.light.add_light(trans_vert, bsm.triangles, colors, light_pos, light_intens)
    lit_colors2 = mesh.light.add_light(trans_vert, bsm.triangles, colors, light_pos2, light_intens)
    lit_colors3 = mesh.light.add_light(trans_vert, bsm.triangles, colors, light_pos3, light_intens)
    lit_colors4 = mesh.light.add_light(trans_vert, bsm.triangles, colors, light_pos4, light_intens)
    lit_colors = (lit_colors1 + lit_colors2 + lit_colors3 + lit_colors4) / 4
    
    image_vertices = mesh.transform.to_image(trans_vert, h, w)
    image_vertices = mesh.transform.to_image(trans_vert, h, w)

    fitted_image = mesh.render.render_colors(image_vertices, bsm.triangles, lit_colors, h, w)
    fitted_image = np.minimum(np.maximum(fitted_image, -1) , 1)    
    fitted_image = (fitted_image * 255.0).astype('u1')
    
    
    #  add mesh on top of the face 
    bg_img = cv2.cvtColor(imgs_c[idx], cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(fitted_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(bg_img, bg_img, mask = mask)
    img2_fg = cv2.bitwise_and(fitted_image, fitted_image, mask = mask_inv)

    mesh_img = cv2.addWeighted(bg_img, 1, img2_fg, 3, 0)
    path = folder + "mix_fitted_image" + str(idx) + ".jpg"
    cv2.imwrite(path, mesh_img)
    
    
    mesh_img = cv2.add(img1_bg, img2_fg)    
    path = folder + "fitted_image" + str(idx) + ".jpg"
    cv2.imwrite(path, mesh_img)






np.savetxt("wid.out", wid)



