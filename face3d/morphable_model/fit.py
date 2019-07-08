'''
Estimating parameters about vertices: shape para, exp para, pose para(s, R, t)
'''
import numpy as np
import cv2
from scipy.optimize import minimize
from . import blendshapes
from .. import tensor
from .. import mesh

''' TODO: a clear document. 
Given: image_points, 3D Model, Camera Matrix(s, R, t2d)
Estimate: shape parameters, expression parameters

Inference: 

    projected_vertices = s*P*R(mu + shape + exp) + t2d  --> image_points
    s*P*R*shape + s*P*R(mu + exp) + t2d --> image_poitns

    # Define:
    X = vertices
    x_hat = projected_vertices
    x = image_points
    A = s*P*R
    b = s*P*R(mu + exp) + t2d
    ==>
    x_hat = A*shape + b  (2 x n)

    A*shape (2 x n)
    shape = reshape(shapePC * sp) (3 x n)
    shapePC*sp : (3n x 1)

    * flatten:
    x_hat_flatten = A*shape + b_flatten  (2n x 1)
    A*shape (2n x 1)
    --> A*shapePC (2n x 199)  sp: 199 x 1
    
    # Define:
    pc_2d = A* reshape(shapePC)
    pc_2d_flatten = flatten(pc_2d) (2n x 199)

    =====>
    x_hat_flatten = pc_2d_flatten * sp + b_flatten ---> x_flatten (2n x 1)

    Goals:
    (ignore flatten, pc_2d-->pc)
    min E = || x_hat - x || + lambda*sum(sp/sigma)^2
          = || pc * sp + b - x || + lambda*sum(sp/sigma)^2

    Solve:
    d(E)/d(sp) = 0
    2 * pc' * (pc * sp + b - x) + 2 * lambda * sp / (sigma' * sigma) = 0

    Get:
    (pc' * pc + lambda / (sigma'* sigma)) * sp  = pc' * (x - b)

'''





def estimate_shape(x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb = 3000):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        shapePC: (3n, n_sp)
        shapeEV: (n_sp, 1)
        expression: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    '''
    x = x.copy()
    assert(shapeMU.shape[0] == shapePC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = shapePC.shape[1]

    n = x.shape[1]
    sigma = shapeEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T.copy()) # 199 x n x 2
    
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    exp_3d = expression
    # 
    b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    #print("pc.shape", np.dot(pc.T, pc).shape)

    #print("reg shape", np.diagflat(1/sigma**2).shape)
    equation_right = np.dot(pc.T, x - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para

def estimate_expression(x, shapeMU, expPC, expEV, shape, s, R, t2d, lamb = 2000):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        expPC: (3n, n_ep)
        expEV: (n_ep, 1)
        shape: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        exp_para: (n_ep, 1) shape parameters(coefficients)
    '''
    x = x.copy()
    assert(shapeMU.shape[0] == expPC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = expPC.shape[1]

    n = x.shape[1]
    sigma = expEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(expPC.T, [dof, n, 3]) 
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T) 
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    shape_3d = shape
    # 
    b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    exp_para = np.dot(np.linalg.inv(equation_left), equation_right)
    
    return exp_para



def estimate_expression_bsm(x, shapeMU, expPC, expEV, s, R, t2d, lamb = 2000):
    x = x.copy()
    assert(shapeMU.shape[0] == expPC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = expPC.shape[1]

    n = x.shape[1]
    sigma = expEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(expPC.T, [dof, n, 3]) 
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T) 
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # 
    b = A.dot(mu_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma**2)

    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    exp_para = np.dot(np.linalg.inv(equation_left), equation_right)
    
    return exp_para

def fit_exp(x, core, wid, ini_exp, max_iter):
    wexp = ini_exp  + np.random.rand(core.shape[2]) * 0.1 - 0.05
    #wexp = np.random.rand(core.shape[2])
    X = tensor.dot.mode_dot(core, wexp.T, 2)
    X = tensor.dot.mode_dot(X, wid.T, 1)
    X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
    x = x.T
    n = x.shape[1]
    fe = 99999
    i = 0
    #for i in range(max_iter):
    while fe > 0.5 and i <  max_iter:

        P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t3d = mesh.transform.P2sRt(P)

        bnds_exp = ((0,1),) * core.shape[2]
        args = (
            x,
            core,
            wid,
            s,
            R,
            t3d[:2]
        )
        res = blendshapes.bfgs(blendshapes.fitting_error_wexp, wexp, args = args, bounds= bnds_exp, options = {
                'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
                'maxiter': 15000, 'iprint': -1, 'maxls': 20
        
        })
        wexp = res.x.reshape((-1))
        fe = np.sqrt(res.fun) / n
        print("expression fitting error in iteration {}:".format(i), fe)
        i = i + 1
    return wexp, s, R, t3d

# ---------------- fit 
def fit_points(x, X_ind, model, n_sp, n_ep, max_iter = 4):
    '''
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''
    x = x.copy().T

    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)

    #-------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]
    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T
        
        #----- estimate pose
        P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t = mesh.transform.P2sRt(P)
        fe = fitting_error(x, X, s, R, t[:2])
        print("fitting error:", fe)
        rx, ry, rz = mesh.transform.matrix2angle(R)
        print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))

        #----- estimate shap
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        ep = estimate_expression(x, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t[:2], lamb = 20)

        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        sp = estimate_shape(x, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t[:2], lamb = 40)
        
        
    return sp, ep, s, R, t


# ---------------- fitting process
def fit_points_for_show(x, X_ind, model, n_sp, n_ep, max_iter = 4):
    '''
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''
    x = x.copy().T

    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)

    #-------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    s = 4e-04
    R = mesh.transform.angle2matrix([0, 0, 0])
    t = [0, 0, 0]
    lsp = []; lep = []; ls = []; lR = []; lt = []
    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T
        lsp.append(sp); lep.append(ep); ls.append(s), lR.append(R), lt.append(t)
        
        #----- estimate pose
        P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t = mesh.transform.P2sRt(P)
        lsp.append(sp); lep.append(ep); ls.append(s), lR.append(R), lt.append(t)

        #----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        ep = estimate_expression(x, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t[:2], lamb = 20)
        lsp.append(sp); lep.append(ep); ls.append(s), lR.append(R), lt.append(t)

        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        sp = estimate_shape(x, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t[:2], lamb = 40)

    # print('ls', ls)
    # print('lR', lR)
    return np.array(lsp), np.array(lep), np.array(ls), np.array(lR), np.array(lt)


#------ fitting process for linear BSM 
def fit_points_BSM(x, X_ind, model, n_ep, max_iter = 4):
    '''
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: BSM
        max_iter: iteration
    Returns:
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''
    x = x.copy().T

    #-- init
    # sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)

    #-------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind]
    shapeEV = model['shapeEV'][:n_ep]
    expPC = model['expPC'][valid_ind, :n_ep]
    print("shapeMU in fit point bsm ", shapeMU.shape)
    print("expPC in fit bsm", expPC.shape)
    for i in range(max_iter):
        X = shapeMU +  expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T
        
        #----- estimate pose
        P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t = mesh.transform.P2sRt(P)
        fe = fitting_error(x, X, s, R, t[:2])
        print("fitting error:", fe)
        rx, ry, rz = mesh.transform.matrix2angle(R)
        print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))

        #----- estimate shape
        # expression
        #shape = shapeMU + expPC.dot(ep)
        #shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        
        ep = estimate_expression_bsm(x, shapeMU, expPC, shapeEV, s, R, t[:2], lamb = 20 )
        #print("exp para", ep[:20])
    return ep, s, R, t


