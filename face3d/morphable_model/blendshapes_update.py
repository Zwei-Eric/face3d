import numpy as np
import cv2
from scipy.optimize import minimize
from .. import tensor
from .. import mesh
from matplotlib import pyplot as plt
from scipy import spatial

def get_valid_ind(X_ind):
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    
    return valid_ind


def find_closest(X, idx, x):
    '''
    X: (n, 2)
    x: (17, 2)

    '''
    ind = np.argsort(X[idx,0])
         
    sorted_h = idx[ind]
    sorted_X = X[idx][ind]
    cls_pids = []
    for pnt in x:
        min_dis = 99999 
        low = 0
        high = ind.shape[0]
        while low < high - 1:
            mid = (high + low) // 2
            if pnt[0]  >= sorted_X[mid,0]:
                low = mid
            else:
                high = mid
        dis = np.sqrt((pnt[0] - sorted_X[mid, 0]) ** 2 + (pnt[1] - sorted_X[mid,1]) **2 )
        cls_pid = sorted_h[mid]
        min_dis = dis
        
        cnt = 1
        while True:
            p = mid + cnt
            if p >= sorted_X.shape[0]:
                break
            dis = np.sqrt((pnt[0] - sorted_X[p, 0]) ** 2 + (pnt[1] - sorted_X[p, 1]) ** 2)
            if dis < min_dis:
                cls_pid = sorted_h[p]
                min_dis = dis
            if np.abs(pnt[0] - sorted_X[p,0]) > min_dis:
                break
            cnt +=1
        
        cnt = 1
        while True:
            p = mid - cnt
            if p < 0:
                break
            dis = np.sqrt((pnt[0] - sorted_X[p, 0]) ** 2 + (pnt[1] - sorted_X[p, 1]) ** 2)
            if dis < min_dis:
                cls_pid = sorted_h[p]
                min_dis = dis
            if np.abs(pnt[0] - sorted_X[p,0]) > min_dis:
                break
            cnt +=1
        cls_pids.append(cls_pid)

    # print(cls_pids)

    return np.asarray(cls_pids)

        




def update_contour_idx(core, wid, wexp, s, R, t2d, x, X_ind, face_ind):
    '''
    face_ind: indices of face without neck and ears


    Returns:
        new_X_ind: updated indices of face mesh contour 
        y: position of face contour points 

    '''
   
    # x: (n x 2)
    X = tensor.dot.mode_dot(core, wexp.T , 2)
    X = tensor.dot.mode_dot(X, wid.T, 1)
    X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
   
    n = X.shape[1]
    
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)
    
    X = A.dot(X)
    X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    X_c = np.zeros(X.shape)
    
    X_c[:,face_ind] = X[:, face_ind]
    
    kp = X_c.T[X_ind]
    
    X = X.T
    X_c = X_c.T 
    mean = np.mean(X_c, axis = 0)

    hull = spatial.ConvexHull(X_c)
    idx = np.append(hull.vertices, X_ind[:17])
    for i in range(3):
        X_c[hull.vertices] = mean
        hull2 = spatial.ConvexHull(X_c)
        idx = np.append(idx, hull2.vertices)
        #for simplex in hull.simplices:
        #    plt.plot(X[simplex, 0], X[simplex, 1],'bo')
        hull = hull2

    plt.plot(X[idx, 0], X[idx, 1], 'bo')
    #for simplex in hull2.simplices:
    #    plt.plot(X[simplex, 0], X[simplex, 1],'bo')
   
    new_ind = find_closest(X, idx ,x[:17,:])
    new_X_ind = X_ind.copy()
    new_X_ind[:17] = new_ind
    y = X[new_ind]
    plt.plot(kp[:,0], kp[:,1], 'co')
    return new_X_ind, y

    
def fitting_error_overall(wid, xl, corel, wexpl, sl, Rl, t2dl):
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
        core = corel[i]
        X = tensor.dot.mode_dot(core, wexpl[i].T , 2)
        X = tensor.dot.mode_dot(X, wid.T, 1)
        X = np.reshape(X, [int(X.shape[0] / 3), 3]).T # 3 x n
        x = xl[i].T
        n = x.shape[1]
        mask = np.ones(n, dtype = bool)
        mask[18:27] = False
    
        t2d = np.array(t2dl[i])
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = sl[i]*P.dot(Rl[i])
    
        X = A.dot(X)
        X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    
        sfe = np.linalg.norm(X.T[mask] - x.T[mask])  
        fe += sfe
    
      
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
    mask = np.ones(n, dtype = bool)
    mask[18:27] = False

    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)
    
    X = A.dot(X)
    X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    fe = np.linalg.norm(X.T[mask] - x.T[mask])  
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
    X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
    n = x.shape[1]
    mask = np.ones(n, dtype = bool)
    mask[18:27] = False
    
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)
    
    X = A.dot(X)
    X = X + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    fe = np.linalg.norm(X.T[mask] - x.T[mask]) 
    

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
    
    Rca, _ = cv2.Rodrigues(rvec)
    Pca = tvec
    M = np.zeros((3,4))
    M[:3, :3] = Rca
    M[:3, 3:] = Pca
    return M, Q






def fit_blendshapes(model, wid, n_ver):
    n_exp = 47
    core = model['core']
    Uexp = model['Uexp']
    d = np.eye(n_exp)
    expPC = np.zeros([int(3*n_ver), n_exp])
    Uexpd = np.dot(Uexp.T, d)
    pc = tensor.dot.mode_dot(core, wid, 1)
    expPC = np.dot(pc, Uexpd)
    
    pmax = expPC.max(axis = 0) 
    pmin = expPC.min(axis = 0)

    expPC = (expPC - pmin) / (pmax - pmin)
    expPC = 1 - expPC
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

    valid_ind = get_valid_ind(X_ind)
    core_base  = model['core']
    min_E = 1000
    #wid = np.random.rand(core_base.shape[1])
    wid = np.loadtxt('wid.out')
    wexp0 = np.random.rand(core_base.shape[2])
    m = len(xl)
    wexpl = [wexp0] * m
    X_inds = [X_ind] * m
    core = core_base[valid_ind, :, :]
    corel = [core] * m
    print("initial indices:", X_ind[:17]) 
    ys = np.zeros([max_iter, m, 17, 2])
    for i in range(max_iter):
        fe = 0
        sl = []
        Rl = []
        tl = []
        wid_j = np.zeros(core_base.shape[1])

        for j in range(len(xl)):
            core = corel[j]
            X = tensor.dot.mode_dot(core, wid.T, 1)
            X = tensor.dot.mode_dot(X, wexpl[j].T ,1)
            X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
            x = xl[j].T 
            #----- estimate pose

            P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
            s, R, t = mesh.transform.P2sRt(P)
            sl.append(s)
            Rl.append(R)
            tl.append(t[:2])
            #----- estimate wid & wexp
            # shape wid
            
            bnds_id = ((0,1),) * core.shape[1]
            args = (  # a_star, M, Q, x, X
                x,
                core,
                wexpl[j],
                s,
                R,
                t[:2]
            )
            # use L-BFGS-B algo to get wid
            res = bfgs(fitting_error_wid, wid, args=args, bounds=bnds_id, options={
                'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
                'maxiter': 15000, 'iprint': -1, 'maxls': 20
            })
            wid_j +=  res.x.reshape(-1)
          
            
            print("image {} single wid error:".format(j), res.fun) 
            
            
            bnds_exp = ((0,1),) * core.shape[2]
            args = (  # a_star, M, Q, x, X
                x,
                core,
                wid,
                s,
                R,
                t[:2]
            )
            res = bfgs(fitting_error_wexp, wexpl[j], args=args, bounds=bnds_exp, options={
                'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
                'maxiter': 15000, 'iprint': -1, 'maxls': 20
            })
            wexpl[j]  = res.x.reshape((-1))
            print("image {} single exp error:".format(j), res.fun) 
            fe += res.fun 
        print("imgae error overall before:", fe) 
        bnds_id = ((0,1),) * core.shape[1]
        wid = wid_j / m
        args = (  # a_star, M, Q, x, X
            xl,
            corel,
            wexpl,
            sl,
            Rl,
            tl,
        )

        res = bfgs(fitting_error_overall, wid, args=args, bounds=bnds_id, options={
            'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000,
            'maxiter': 15000, 'iprint': -1, 'maxls': 20
        })
        wid = res.x.reshape((-1))
        print("iter " + str(i+1) + " done. E= ", res.fun)
        for j in range(len(xl)):
            plt.cla()
            new_X_ind, y = update_contour_idx(core_base, wid, wexpl[j], sl[j], Rl[j], tl[j], xl[j], X_inds[j], model['face_ind'])
            X_inds[j] = new_X_ind
            ys[i][j] = y
            #print('new ind after update', new_X_ind[:17])
            plt.plot(xl[j][:,0], xl[j][:,1], 'r+')
          
            valid_ind = get_valid_ind(new_X_ind)
            corel[j] = core_base[valid_ind,:,:]
            k = 17
            plt.plot( [xl[j][:k,0], y[:k,0]], [xl[j][:k,1], y[:k,1]], color = 'g')
            plt.axis("equal")
            plt.savefig('qz/scatter_img{}_iter{}.jpg'.format(j,i))
            
            #fe = fitting_error_wid(wid, xl[j].T, corel[j], wexpl[j], sl[j], Rl[j], tl[j])
            #print("1: single image {} error after change ind".format(j), fe)
            #fe = np.linalg.norm(xl[j][:k] - y) ** 2
            #print("2: single image {} error after change ind".format(j), fe)
    return wid ,ys


