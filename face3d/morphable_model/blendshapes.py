import numpy as np
import cv2
from skimage import draw
from .. import tensor
from .. import mesh


from scipy.optimize import minimize

def fitting_error_overall(wid, xl, core, wexpl, sl, Rl, t3dl):
    '''
    Args:
        xl: m * (2, n) image points
        core tensor: (3n, n_id, n_exp) 
        Ml: m * (3, 4) camera external matrix
        Ql: m * (3, 3) projection matrix
    Returns:
        fe : fitting error

    '''
    fe = 0
    #vertices = tensor.dot.mode_dot(tensor.dot.mode_dot(core, wid.T, 1), wexp.T, 1)
    m = len(xl)
    for i in range(m):
        X = tensor.dot.mode_dot(core, wexpl[i].T , 2)
        X = tensor.dot.mode_dot(X, wid.T, 1)
        X = np.reshape(X, [int(X.shape[0] / 3), 3]).T # 3 x n
        x = xl[i].T
        n = x.shape[1]
    
        t2d = np.array(t3dl[i][:2])
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = sl[i]*P.dot(Rl[i])
    
        X = A.dot(X)
        X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    
        fe += np.linalg.norm(X - x) ** 2 
    
        #X = np.r_[X, np.ones([1, n])]
    

        #X = Ml[i].dot(X)
        #w = X[2,:]
        #X = X / w
        #X = Ql[i].dot(X)
      
      
   
        # fe += np.linalg.norm(X[:2,:] - x)
    # vertices.shape (4, n)
    #vertices = np.insert(vertices, 3, values = 1, axis = 1).T 
    #fe = np.linag.norm(np.dot(cm, vertices).T[:,:2] - x)
    return fe



def fitting_error_wid(wid, x, core, wexp, s, R, t2d):
    '''
    Args:
        x: (2, n) image points
        vertices: (3, n) 
        M: (3, 4) camera external matrix
        Q: (3, 3) projection matrix

    Returns:
        fe : fitting error

    '''
    #vertices = tensor.dot.mode_dot(tensor.dot.mode_dot(core, wid.T, 1), wexp.T, 1)
    X = tensor.dot.mode_dot(core, wexp.T , 2)
    X = tensor.dot.mode_dot(X, wid.T, 1)
    X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
    n = x.shape[1]
    
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)
    
    X = A.dot(X)
    X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    fe = np.linalg.norm(X - x) ** 2
    
    #fe = np.linalg.norm(X - x) 
    
    #X = np.r_[X, np.ones([1, n])]
    
    #X = M.dot(X)
    #w = X[2,:]
    #X = X / w
    #X = Q.dot(X)
   
    # vertices.shape (4, n)
    #vertices = np.insert(vertices, 3, values = 1, axis = 1).T 
    #fe = np.linag.norm(np.dot(cm, vertices).T[:,:2] - x)
    return fe

def fitting_error_wexp(wexp, x, core, wid,  s, R, t2d):
    '''
    Args:
        x: (2, n) image points
        vertices: (3, n) 
        M: (3, 4) camera external matrix
        Q: (3, 3) projection matrix

    Returns:
        fe : fitting error

    '''
    X = tensor.dot.mode_dot(core, wexp.T , 2)
    X = tensor.dot.mode_dot(X, wid.T, 1)
    #X = tensor.dot.mode_dot(core, wid.T, 1)
    #X = tensor.dot.mode_dot(X, wexp.T , 1)
    X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
    n = x.shape[1]
    
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)
    
    X = A.dot(X)
    X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    fe = np.linalg.norm(X - x) ** 2
    #X = np.r_[X, np.ones([1, n])]
    

    #X = M.dot(X)
    #w = X[2,:]
    #X = X / w
    #X = Q.dot(X)
   
    #fe = np.linalg.norm(X[:2,:] - x)
    
    # vertices.shape (4, n)
    #vertices = np.insert(vertices, 3, values = 1, axis = 1).T 
    #fe = np.linag.norm(np.dot(cm, vertices).T[:,:2] - x)
    return fe
 

def bfgs(fun, weight, args, bounds, options):

    res = minimize(fun, weight, args=args, method='L-BFGS-B', bounds=bounds, options=options)
    return res

def posit(X, x):

    Q = np.zeros(3*3, float).reshape(3, -1) + np.eye(3)

    X = X.T
    x = x.T
    assert (x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert (n >= 4)

    # 2d points
    mean = np.mean(x, 1)  # (2,)

    Q[:2, 2] = mean
    Q[0, 0] = 800  # mm
    Q[1, 1] = 800
    X = X.T
    x = x.T
    X = X[:4]
    x = x[:4]
    retval, rvec, tvec = cv2.solvePnP(X, x, Q, None)
    # if retval:
    #     print("posit succeed")
    # else :
    #     print("posit fail")
    # print(rvec)
    Rca, _ = cv2.Rodrigues(rvec)
    Pca = tvec
    M = np.zeros((3,4))
    M[:3, :3] = Rca
    M[:3, 3:] = Pca
    return M, Q






def generate_blendshapes(model, wid, n_ver):
    n_exp = 47
    core = model['core']
    Uexp = model['Uexp']
    d = np.eye(n_exp)
    expPC = np.zeros([int(3*n_ver), n_exp])
    Uexpd = np.dot(Uexp.T, d)
    pc = tensor.dot.mode_dot(core, wid, 1)
    expPC = np.dot(pc, Uexpd)
    #for i in range(n_exp):
    #    Uexpd = np.dot(Uexp.T, d[:,i])
    #    pc =  tensor.dot.mode_dot(core, wid, 1)
    #    expPC[:,i] = tensor.dot.mode_dot(pc, Uexpd, 1)
    #pmax = expPC.max(axis = 0) 
    #pmin = expPC.min(axis = 0)

    #expPC = (expPC - pmin) / (pmax - pmin)
    expPC = - expPC
    #print(expPC)
    return expPC
    


def fit_id_param_bfgs(xl, X_ind, model, max_iter=4):
    '''
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        max_iter: iteration
    Returns:
    '''
    # -- init
    # -------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')


    core = model['core'][valid_ind, :, :]
    min_E = 1000
    #wid = np.random.rand(core.shape[1])
    wid = np.loadtxt("wid.out")
    #wid = np.ones(core.shape[1])
    wexp0 = np.random.rand(core.shape[2])
    #wexp0 = np.ones(core.shape[2])
    m = len(xl)
    wexpl = [wexp0] * m
 
    for i in range(max_iter):
        fe = 0
        sl = []
        Rl = []
        t3dl = []
        wid_j = np.zeros(core.shape[1])
        #Ml = []
        #Ql = []
        #shapePC_l = []
        for j in range(len(xl)):
            X = tensor.dot.mode_dot(core, wid.T, 1)
            X = tensor.dot.mode_dot(X, wexpl[j].T ,1)
            X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
            x = xl[j].T 
            #----- estimate pose
            #posit_M, Q = posit(X.T, x.T)
            #Ml.append(posit_M)
            #Ql.append(Q)


            P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
            s, R, t3d = mesh.transform.P2sRt(P)
            sl.append(s)
            Rl.append(R)
            t3dl.append(t3d)
            #----- estimate wid & wexp
            # shape wid
            #shapePC = tensor.dot.mode_dot(core, wexpl[j].T ,2)
            
            bnds_id = ((-1,1),) * core.shape[1]
            args = (  # a_star, M, Q, x, X
                x,
                core,
                wexpl[j],
                #Q,
                #posit_M
                s,
                R,
                t3d[:2]
            )
            # use L-BFGS-B algo to get wid
            res = bfgs(fitting_error_wid, wid, args=args, bounds=bnds_id, options={
                'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
                'maxiter': 15000, 'iprint': -1, 'maxls': 20
            })
            wid_j +=  res.x.reshape(-1)
          
            
            print("image {} single wid error:".format(j), res.fun) 
            #wid_j = estimate_weight(x, shapePC, s, R, t[:2])

            # expression wexp
            #expPC = tensor.dot.mode_dot(core, wid_j.T, 1)
            #print("2expPC shape", expPC.shape) 
            
            
            bnds_exp = ((-1,1),) * core.shape[2]
            args = (  # a_star, M, Q, x, X
                x,
                core,
                wid,
                #Q,
                #posit_M
                s,
                R,
                t3d[:2]
            )
            res = bfgs(fitting_error_wexp, wexpl[j], args=args, bounds=bnds_exp, options={
                'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
                'maxiter': 15000, 'iprint': -1, 'maxls': 20
            })
            wexpl[j]  = res.x.reshape((-1))
            print("image {} single exp error:".format(j), res.fun) 
            
 
        bnds_id = ((-1,1),) * core.shape[1]
        wid = wid_j / m
        args = (  # a_star, M, Q, x, X
            xl,
            core,
            wexpl,
            #Ql,
            #Ml
            sl,
            Rl,
            t3dl
        )
        res = bfgs(fitting_error_overall, wid, args=args, bounds=bnds_id, options={
            'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
            'maxiter': 15000, 'iprint': -1, 'maxls': 20
        })
        wid = res.x.reshape((-1))
        print("iter " + str(i+1) + " done. E= ", res.fun)
    return wid, wexpl, sl, Rl, t3dl


def show_fitting_result(core, s, R, t3d, wid, wexp, kpoint_num):
    
    
#    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
#    X_ind_all[1, :] += 1
#    X_ind_all[2, :] += 2
#    valid_ind = X_ind_all.flatten('F')
#
#
#    core = model['core'][valid_ind, :, :]
#    n = X_ind.shape[0]
#    m = len(sl)
#    Xs = []
#    for i in range(m):
    X = tensor.dot.mode_dot(core, wexp.T , 2)
    X = tensor.dot.mode_dot(X, wid.T, 1)
    X = np.reshape(X, [int(X.shape[0] / 3), 3]).T # 3 x n

    t2d = np.array(t3d[:2])
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    X = A.dot(X)
    X = X + np.tile(t2d[:, np.newaxis], [1, kpoint_num]) # 2 x n
     
    X = X.T
    return X

def generate_mesh(core, wid, wexp):
    X = tensor.dot.mode_dot(core, wexp.T , 2)
    X = tensor.dot.mode_dot(X, wid.T, 1)
    return X 
