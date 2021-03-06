from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
from .. import mesh
from . import fit
from . import load
from . import blendshapes
from .. import objloader
class  MorphabelModel(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1]. *
            'shapePC': [3*nver, n_shape_para]. *
            'shapeEV': [n_shape_para, 1]. ~
            'expMU': [3*nver, 1]. ~ 
            'expPC': [3*nver, n_exp_para]. ~
            'expEV': [n_exp_para, 1]. ~
            'texMU': [3*nver, 1]. ~
            'texPC': [3*nver, n_tex_para]. ~
            'texEV': [n_tex_para, 1]. ~
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """
    def __init__(self, model_path, model_type = 'BFM'):
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.model = load.load_BFM(model_path)
            self.nver = self.model['shapePC'].shape[0]/3
            self.n_shape_para = self.model['shapePC'].shape[1]
            self.n_exp_para = self.model['expPC'].shape[1]
            self.n_tex_para = self.model['texPC'].shape[1]
            self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))
        elif model_type == 'BSM':
            self.model = load.load_BSM(model_path)
            self.nver = self.model['core'].shape[0]/3
            self.n_shape_para = self.model['core'].shape[1]
            self.n_exp_para = self.model['core'].shape[2]
            #self.nver = self.model['core'].shape[0]
            #self.n_shape_para = self.model['core'].shape[1]
            #self.n_exp_para = self.model['core'].shape[2]
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        # fixed attributes
        self.model_type = model_type
        self.triangles = self.model['tri']
        self.kpt_ind = self.model['kpt_ind']
        self.ntri = self.model['tri'].shape[0]


    # ------------------------------------- shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, type = 'random'):
        if type == 'zero':
            sp = np.random.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.rand(self.n_shape_para, 1)*1e04
        return sp

    def get_exp_para(self, type = 'random'):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = -1.5 + 3 * np.random.random([self.n_exp_para, 1])
            ep[6:, 0] = 0

        return ep 

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T

        return vertices

    def generate_vertices_bsm(self, exp_para):
        vertices = self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T
        return vertices 

    # -------------------------------------- texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type = 'random'):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tp = np.random.rand(self.n_tex_para, 1)
        return tp

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_para*self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T/255.  
        
        return colors


    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
            x: pitch. positive for looking down 
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        Returns:
            vertices: rotated vertices
        '''
        return mesh.transform.rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    def similarity_transform(self, vertices, s, R, t3d):
        return mesh.transform.similarity_transform(vertices, s, R, t3d)
    
    def transform_3ddfa(self, vertices, s, angles, t3d): # only used for processing 300W_LP data
        R = mesh.transform.angle2matrix_3ddfa(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)
    
    def similarity_transform(self, vertices, s, R, t3d):
        return mesh.transform.similarity_transform(vertices, s, R, t3d)
    
    def fit_specific_blendshapes(self, xl, X_ind, max_iter, kp_type = '3D'):
        #wid = blendshapes.fit_specific_id_param(xl, X_ind, self.model, max_iter = max_iter)

        if kp_type == '3D':
            wid , wexpl, sl, Rl, tl= blendshapes.fit_id_param_bfgs(xl, X_ind, self.model, max_iter = max_iter)
            expPC = blendshapes.generate_blendshapes(self.model, wid, self.nver)
            return expPC, wid, wexpl, sl, Rl, tl
        elif kp_type == '2D':
            wid,  wexpl, sl, Rl, tl, new_X_ind  = blendshapes.fit_id_param_bfgs(xl, X_ind, self.model, max_iter = max_iter, kp_type = '2D')
            expPC = blendshapes.generate_blendshapes(self.model, wid, self.nver)
            return expPC, wid, wexpl, sl, Rl, tl, new_X_ind
        else:
            print('unknown keypoint type')

    def show_fitting_result(self, X_ind, s, R, t3d, wid, wexp):
        '''
        get positions of keypoints
        '''
        valid_ind = self.get_valid_ind(X_ind)
        core = self.model['core'][valid_ind, :, :]
        n = X_ind.shape[0]
        img = blendshapes.show_fitting_result(core, s, R, t3d ,wid, wexp, n)
        return img

    def generate_expression_mesh(self,  wexp, mask = []):
        
        mask = np.asarray(mask)
        meanface = self.model['shapeMU']
        expPC = self.model['expPC']
        print("meanface expPC shape", meanface.shape, expPC.shape)
        if mask.shape[0] == 0:
            mesh = meanface + np.dot(expPC, wexp[:, np.newaxis])
        else: 
            mesh = meanface[mask] + np.dot(expPC[mask,:], wexp[:, np.newaxis])
        return mesh 


    def generate_bilinear_mesh(self, wid, wexp):
        core = self.model['core']
        return blendshapes.generate_bilinear_mesh(core, wid, wexp)

    
    def get_valid_ind(self, X_ind):
        X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
        X_ind_all[1, :] += 1
        X_ind_all[2, :] += 2
        valid_ind = X_ind_all.flatten('F')
        return valid_ind
        

    # --------------------------------------------------- fitting
    def fit(self, x, X_ind, max_iter = 4, isShow = False, model_type = 'BFM', method = 'default'):
        ''' fit 3dmm & pose parameters
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            max_iter: iteration
            isShow: whether to reserve middle results for show
        Returns:
            fitted_sp: (n_sp, 1). shape parameters
            fitted_ep: (n_ep, 1). exp parameters
            s, angles, t
        '''
        if model_type == 'BFM':
            if isShow:
                fitted_sp, fitted_ep, s, R, t = fit.fit_points_for_show(x, X_ind, self.model, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
                angles = np.zeros((R.shape[0], 3))
                for i in range(R.shape[0]):
                    angles[i] = mesh.transform.matrix2angle(R[i])
            else:
                fitted_sp, fitted_ep, s, R, t = fit.fit_points(x, X_ind, self.model, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
                angles = mesh.transform.matrix2angle(R)
            return fitted_sp, fitted_ep, s, angles, t
        elif model_type == 'BSM':
            if method == 'bfgs':
                fitted_ep, s, R, t = fit.fit_exp_bfgs(x, X_ind, self.model, n_ep = self.n_exp_para, max_iter = max_iter)
            else:
                fitted_ep, s, R, t = fit.fit_points_BSM(x, X_ind, self.model, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = mesh.transform.matrix2angle(R)
            return fitted_ep, s, angles, t

    def fit_expression(self, x, X_ind, ini_wexp,  max_iter = 4):
        valid_ind = self.get_valid_ind(X_ind)
        #core = self.model['core'][valid_ind, :, :]
        meanface = self.model['shapeMU'][valid_ind]
        expPC = self.model['expPC'][valid_ind,:]
        wexp , s, R, t3d, fe = fit.fit_exp(x, meanface, expPC,ini_wexp, max_iter = max_iter)
        n = X_ind.shape[0]
        X = meanface + np.dot(expPC, wexp[:, np.newaxis])
        X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
        t2d = np.array(t3d[:2])
        P = np.array([[1,0,0] , [0, 1, 0]], dtype = np.float32)
        A = s* P.dot(R)

        X = A.dot(X)
        X = X + np.tile(t2d[:, np.newaxis], [1, n])
        return X.T, wexp, s, R, t3d, fe

    def fit_expression_2D(self, x, X_ind, ini_wexp,  max_iter = 4):
        

        meanface = self.model['shapeMU']
        expPC = self.model['expPC']
        wexp , s, R, t3d, fe, new_ind = fit.fit_exp_2D(x, X_ind, meanface, expPC,ini_wexp, self.model['face_ind'], max_iter = max_iter)
        return  wexp, s, R, t3d, fe, new_ind



    def fit_color(self, x_colors, X_ind):
        '''
        args: 
        x_color: (n,3) image key points colors
    
        ''' 
        x_colors  = x_colors.copy()
        n_tp = self.n_tex_para
        #print("n_tex_para", n_tp)
        #-- init
        tp = np.zeros((n_tp, 1), dtype = np.float32)
    
        
        X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
        X_ind_all[1, :] += 1
        X_ind_all[2, :] += 2
        valid_ind = X_ind_all.flatten('F')
    
        texMU = self.model['texMU'][valid_ind, :] 
        texPC = self.model['texPC'][valid_ind, :n_tp]
        texEV = self.model['texEV'][:n_tp,:]
   
        # bug fixed: should set RGB value to range 0-255 at first
        x_colors = x_colors * 255.0
        x_colors = np.reshape(x_colors, [-1, 1]) - texMU 
        print(x_colors[0:5])
        print("x_colors_shape:", x_colors.shape)
        #------- solve least sqaure equation
        print("texEV.shape",texEV.shape)
        print("texPC.shape",texPC.shape)
        #T = np.array(texPC.dot(np.diag(texEV[:,0])))
        PtP = np.dot(texPC.T,texPC)
        PtP_inv = np.linalg.inv(PtP)
        #print(x_colors-texMU)
        tex_param = np.dot(PtP_inv,np.dot(texPC.T,x_colors))
        ev_inv = np.linalg.inv(np.diag(texEV[:,0]))
        return np.dot(ev_inv, tex_param)


    def fit_color_v2(self, x_colors, X_ind, lamb = 20):
        '''
        args: 
        x_color: (n,3) image key points colors to be fitted
    
        ''' 
        x_colors  = x_colors.copy()
        n_tp = self.n_tex_para
        #print("n_tex_para", n_tp)
        #-- init
        tp = np.zeros((n_tp, 1), dtype = np.float32)
    
        
        X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
        X_ind_all[1, :] += 1
        X_ind_all[2, :] += 2
        valid_ind = X_ind_all.flatten('F')
    
        texMU = self.model['texMU'][valid_ind, :] 
        texPC = self.model['texPC'][valid_ind, :n_tp]
        texEV = self.model['texEV'][:n_tp,:]
            
        # print("texEV shape", texEV.shape)
        sigma = texEV
        
        # bug fixed: should set RGB value to range 0-255 at first
        x_colors = x_colors * 255.0
        x_colors = np.reshape(x_colors, [-1, 1]) - texMU 
        #------- solve least sqaure equation
        equation_left = np.dot(np.dot(texPC.T, texPC), np.diag(texEV[:,0])) + lamb *  np.diagflat(1/sigma**2)
        #print(x_colors-texMU)
        equation_right = np.dot(texPC.T, x_colors)
        tex_param = np.dot(np.linalg.inv(equation_left), equation_right)
        #ev_inv = np.linalg.inv(np.diag(texEV[:,0]))
        #return np.dot(ev_inv, tex_param)
        return tex_param
#    def dense_fit_color(self, x_colors, X_ind, lamb = 20):

