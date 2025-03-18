"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2
from scipy.signal import convolve2d

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def rankEnforce(F):
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0 
    F_enforced = U @ np.diag(S) @ Vt 

    return F_enforced


def getSVD(A):
    U, S, Vt = np.linalg.svd(A)
    return U, S, Vt.T


def normalize_pts(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    T = np.array([
        [1/std[0], 0, -mean[0]/std[0]],
        [0, 1/std[1], -mean[1]/std[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def eight_point(pts1, pts2, M):
    pts1_N, T1 = normalize_pts(pts1)
    pts2_N, T2 = normalize_pts(pts2)


    num = len(pts1)
    x2 = np.reshape(pts2_N[:, 0], (num, 1))
    y2 = np.reshape(pts2_N[:, 1], (num, 1))
    x1 = np.reshape(pts1_N[:, 0], (num, 1))
    y1 = np.reshape(pts1_N[:, 1], (num, 1))

    A = np.hstack((
        x2 * x1, 
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        np.ones((num, 1))
    ))


    U, S, V = getSVD(A)

    f = V[:, -1]
    F = np.reshape(f, (3, 3))
    
    return F, T1, T2


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def findPoint(im1, im2, x1, y1, a, b, c):
    windowSize = 4
    halfwidth = windowSize // 2
    im2_ = im2
    im1_ = im1
    mat1 = im1_[y1 - halfwidth: y1 + halfwidth + 1, x1 - halfwidth : x1 + halfwidth + 1]

    h, w = im2_.shape[:2]
    min_ssd = float('inf')
    bestMatch = np.array([-1, -1])
    for x2 in range(w):
        y2 = (- c - a * x2) / b
        if y2 < 2 or y2 >= h - 2:  
            continue
        y2 = np.int32(np.round(y2, decimals=0))
        mat2 = im2_[y2 - 2: y2 + 3, x2 - 2 : x2 + 3]
        if(mat2.shape != mat1.shape):
            continue
        diff = np.int64(mat2 - mat1)
        sqdiff = diff ** 2
        ssd = sum(sum(sum(sqdiff)))
        if(ssd < min_ssd):
            min_ssd = ssd
            bestMatch = np.array([x2, y2])
    return bestMatch[0], bestMatch[1]

def epipolar_correspondences(im1, im2, F, pts):
    x_co = np.reshape(pts[:, 0], (len(pts), 1))
    y_co = np.reshape(pts[:, 1], (len(pts), 1))
    pts_augmented = np.hstack((x_co, y_co, np.ones((len(pts), 1))))
    l = F @ pts_augmented.T
    l = l.T
    a = l[:, 0]
    b = l[:, 1]
    c = l[:, 2]
    x = pts[:, 0]
    y = pts[:, 1]
    corrs = []
    for i in range(len(pts)):
        x_c, y_c = findPoint(im1, im2, x[i], y[i], a[i], b[i], c[i])
        corrs.append(np.array([x_c, y_c]))
    corrs = np.array(corrs)
    return corrs


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)

    S[2] = 0 
    E = U @ np.diag(S) @ Vt
    return E

def getRotationMatrices(K1, K2, F):
    E = essential_matrix(F, K1, K2)

    U, D, Vt = np.linalg.svd(E)

    D_corrected = np.diag([1, 1, 0])
    E = U @ D_corrected @ Vt

    U, D, Vt = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    t = U[:, 2]
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    # t1 = np.zeros((3, 1))
    t1 = -U[:, 2]
    t2 = U[:, 2]
    print("dets\n")
    print(np.linalg.det(-R1))
    print(np.linalg.det(-R2))
    return -R1, -R2, t1, t2

def getP1(K1, R1, t1):
    temM1 = np.hstack((R1, np.reshape(t1, (3, 1))))
    P1 = K1 @ temM1
    return P1


def getPandP2(F, e):
    P = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((np.eye(3), np.zeros((3, 1))))


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""

def compute_reprojection_error(P, pts_2d, pts_3d):
    N = pts_2d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((N, 1))))  

    projected = (P @ pts_3d_hom.T).T
    projected /= projected[:, 2].reshape(-1, 1)  

    error = np.linalg.norm(pts_2d - projected[:, :2], axis=1)
    return np.mean(error)

def triangulate(P1, pts1, P2, pts2):

    N = pts1.shape[0]  
    pts3d = np.zeros((N, 3)) 

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, Vh = np.linalg.svd(A)
        X_homogeneous = Vh[-1]  

        X = X_homogeneous[:3] / X_homogeneous[3]

        pts3d[i] = X
    error = compute_reprojection_error(P1, pts1, pts3d)
    return pts3d, error

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):

    C1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)  
    C2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)  

    r1 = (C1 - C2).flatten()
    print("RSAHPE ", r1.shape)
    r1 /= np.linalg.norm(r1) 
    # r1 /= (r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2) ** 0.5

    r2 = np.cross(R1[2, :].T, r1)  
    r2 /= np.linalg.norm(r2) 

    r3 = np.cross(r2, r1) 
    r3 /= np.linalg.norm(r3)  

    R_rect = np.vstack([r1, r2, r3]) 

    R1p = R_rect
    R2p = R_rect

    K1p = K2 
    K2p = K2

    t1p = -R_rect @ C1
    t2p = -R_rect @ C2

    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):

    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    H, W = im1.shape

    half_win = win_size // 2

    dispM = np.zeros((H, W), dtype=np.float32)

    im1_padded = cv2.copyMakeBorder(im1, half_win, half_win, half_win, half_win, cv2.BORDER_CONSTANT, 0)
    im2_padded = cv2.copyMakeBorder(im2, half_win, half_win, half_win + max_disp, half_win, cv2.BORDER_CONSTANT, 0)

    for y in range(half_win, H + half_win):
        for x in range(half_win, W + half_win):
            ref_block = im1_padded[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]

            best_disp = 0
            min_ssd = float('inf')

            for d in range(max_disp):
                if x - d < half_win:  
                    continue

                target_block = im2_padded[y - half_win:y + half_win + 1, x - half_win - d:x + half_win + 1 - d]

                ssd = np.sum((ref_block - target_block) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d

            dispM[y - half_win, x - half_win] = best_disp

    return dispM

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    
    C1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    C2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.linalg.norm(C1 - C2)

    f = K1[0, 0]

    depthM = np.zeros_like(dispM, dtype=np.float32)
    valid_disp = dispM > 0  
    depthM[valid_disp] = (b * f) / dispM[valid_disp]

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):

    N = x.shape[0]

    X_h = np.hstack((X, np.ones((N, 1))))  
    x_h = np.hstack((x, np.ones((N, 1))))  

    A = []
    for i in range(N):
        X_i = X_h[i, :]
        u, v, w = x_h[i, :]

        A.append([0, 0, 0, 0, -w * X_i[0], -w * X_i[1], -w * X_i[2], -w * X_i[3], v * X_i[0], v * X_i[1], v * X_i[2], v * X_i[3]])
        A.append([w * X_i[0], w * X_i[1], w * X_i[2], w * X_i[3], 0, 0, 0, 0, -u * X_i[0], -u * X_i[1], -u * X_i[2], -u * X_i[3]])

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)

    P = Vt[-1].reshape(3, 4)

    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):

    M = P[:, :3]

    K, R = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)

    T = np.diag(np.sign(np.diag(K)))
    K = K @ T  
    R = T @ R  

    t = np.linalg.inv(K) @ P[:, 3]

    return K, R, t.reshape(3, 1)
