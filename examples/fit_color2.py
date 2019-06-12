''' 3d morphable model example
3dmm parameters --> mesh 
fitting: 2d image + 3dmm -> 3d face
'''
import os, sys
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io, draw
from time import time
import matplotlib.pyplot as plt
sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')

# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
sp = bfm.get_shape_para('random')
ep = bfm.get_exp_para('random')
vertices = bfm.generate_vertices(sp, ep)

tp = bfm.get_tex_para('random')

colors = bfm.generate_colors(tp)
colors = np.minimum(np.maximum(colors, 0), 1)

# --- 3. transform vertices to proper position
s = 8e-04
angles = [0, 0, 0]
t = [0, 0, 0]
transformed_vertices = bfm.transform(vertices, s, angles, t)
projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

# --- 4. render(3d obj --> 2d image)
# set prop of rendering
h = w = 256; c = 3
image_vertices = mesh.transform.to_image(projected_vertices, h, w)
image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit
x = projected_vertices[bfm.kpt_ind, :2] # 2d keypoint, which can be detected from image
X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.

np.savetxt("x_targ", x , fmt = "%d")

x_colors = []
#print("shapes of image", image.shape)
#print(image)
for u,v in x:
    u = int(round(u)) + w / 2
    v = h / 2 - int(round(v))
#    print(u,v)
    pixel = image[v,u,:]
#    print(pixel)
    x_colors.append(pixel)

x_colors = np.array(x_colors)

## key point on input  image




# fit
fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)

# fit_tex


np.savetxt("x_colors", x_colors, fmt = "%.3f")
np.savetxt("colors", colors[X_ind,:], fmt = "%.3f")
#x_colors = colors[X_ind,:]
fitted_tp  = bfm.fit_color(x_colors, X_ind)
print("fitted_tp", fitted_tp[:6])
print("tp", tp[:6])
fitted_colors = bfm.generate_colors(fitted_tp)
fitted_colors = np.minimum(np.maximum(fitted_colors, 0), 1)

np.savetxt("fitted_color", fitted_colors[X_ind,:], fmt = "%.3f")

# verify fitted parameters
fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, fitted_colors, h, w)



fitted_image = np.around(fitted_image , decimals = 4)

### ----------------- visualize fitting process

# ------------- print & show 
print('pose, groudtruth: \n', s, angles[0], angles[1], angles[2], t[0], t[1])
print('pose, fitted: \n', fitted_s, fitted_angles[0], fitted_angles[1], fitted_angles[2], fitted_t[0], fitted_t[1])

save_folder = 'results/fit_color'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

io.imsave('{}/generated.jpg'.format(save_folder), image)



img_t =io.imread('{}/generated.jpg'.format(save_folder))

for u,v in x:
    u = int(round(u)) + w / 2
    v = h / 2 - int(round(v))
    rr,cc = draw.circle_perimeter(v,u,1)
    draw.set_color(img_t, [rr,cc], [139,0,0])

io.imsave('{}/keypoint.jpg'.format(save_folder), img_t)

np.savetxt("fit_img", fitted_image[:,:,0])
io.imsave('{}/fitted.jpg'.format(save_folder), fitted_image)

