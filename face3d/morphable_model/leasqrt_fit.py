import numpy as np
import cv2

from .. import tensor
from .. import mesh


def etimate_weight(x, PC, s, R, t2d):
    '''
    x: (2, n) image points
    pc: (3 * n, n_pc) principle component
    '''
    x = x.copy()
    assert(PC.shape[0] == x.shape[1]*3)

    dof = PC.shape[1]

    n = x.shape[1]
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(PC.T, [dof, n, 3])
    pc_3d = np.reshape(pc_3d, [dof*n, 3])
    pc_2d = pc_3d.dot(A.T)
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

    # --- calc b
    # shapeMU
    #
    b = np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc)

    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    param = np.dot(np.linalg.inv(equation_left), equation_right)[:,0]
    #print("param shape", param.shape)
    return param


def estimate_wid_overall(xl, PC_l, sl, Rl, t2dl):
    '''
    xl: m * (n, 2) image points
    pc: (3 * n, n_pc) principle component
    '''
    m = len(xl)
    n = xl[0].shape[0]
    #xs = np.zeros([m, 2, n])
    #for  idx, x in enumerate(xl):
    #    xs[idx] = x.T


    #assert(PC.shape[0] == x.shape[1]*3)
    for i in range(m):
        x = xl[i].T
        PC = PC_l[i]
        dof = PC.shape[1]
        t2d = t2dl[i]
        t2d = np.array(t2d)
        s = sl[i]
        R = Rl[i]
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = s*P.dot(R)

        # --- calc pc
        pc_3d = np.resize(PC.T, [dof, n, 3])
        pc_3d = np.reshape(pc_3d, [dof*n, 3])
        pc_2d = pc_3d.dot(A.T)
        pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

        # --- calc b
        # shapeMU
        #
        b = np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
        b = np.reshape(b.T, [-1, 1]) # 2n x 1

        x = np.reshape(x.T, [-1, 1])
        # --- solve
        if i is 0:
            equation_left = np.dot(pc.T, pc)
            equation_right = np.dot(pc.T, x - b)
        else:
            equation_left += np.dot(pc.T, pc)
            equation_right += np.dot(pc.T, x - b)



    param = np.dot(np.linalg.inv(equation_left), equation_right)[:,0]
    #print("param shape", param.shape)
    return param


# ---------------- fit
def fit_specific_id_param(xl, X_ind, model, max_iter = 4):
    '''
    Args:
        xl: m * (n, 2) list of image points contains m images
        X_ind: (n,) corresponding Model vertex indices
        model: BSM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''

    # -- init
    # -------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    core = model['core'][valid_ind, :, :]
    wid = np.random.rand(core.shape[1])
    wexp0 = np.random.rand(core.shape[2])
    m = len(xl)
    #widl = [wid0] * m
    wexpl = [wexp0] * m
    for i in range(max_iter):
        fe = 0
        sl = []
        Rl = []
        tl = []
        shapePC_l = []
        for j in range(len(xl)):
            X = tensor.dot.mode_dot(core, wid.T, 1)
            X = tensor.dot.mode_dot(X, wexpl[j].T ,1)
            X = np.reshape(X, [int(X.shape[0] / 3), 3]).T
            x = xl[j].T
            #print("spb:x shape", x.shape )
            #print("spb:X shape", X.shape)
            #----- estimate pose
            P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
            s, R, t = mesh.transform.P2sRt(P)
            #fer = fitting_error(x, X, s, R, t[:2])
            #print("fitting error:", fer)
            rx, ry, rz = mesh.transform.matrix2angle(R)
            #print('Iter:{}/{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i,j, s, rx, ry, rz, t[0], t[1]))
            sl.append(s)
            Rl.append(R)
            tl.append(t[:2])
            #----- estimate wid & wexp
            # shape wid
            shapePC = tensor.dot.mode_dot(core, wexpl[j].T ,2)
            # shapePC = np.reshape(shapePC, [int(len(shape)/3), 3]).T
            #print("1shapePC shape", shapePC.shape)
            wid_j = estimate_weight(x, shapePC, s, R, t[:2])

            # expression wexp
            expPC = tensor.dot.mode_dot(core, wid_j.T, 1)
            #print("2expPC shape", expPC.shape)
            wexpl[j] = estimate_weight(x, expPC, s, R, t[:2])

            shapePC = tensor.dot.mode_dot(core, wexpl[j].T ,2)
            #print("3shapePC shape", shapePC.shape)
            shapePC_l.append(shapePC)
            fe += fitting_error(x, X, sl[j], Rl[j], tl[j])
            #print(wid_j[:3])


        wid = estimate_wid_overall(xl, shapePC_l, sl, Rl, tl)

        #print(wid[:2])
        print("iteration %d, fitting error: %f" % (i, fe))
    print("final wid", wid)
    return wid

