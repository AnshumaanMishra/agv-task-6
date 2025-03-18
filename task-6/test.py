"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def rankEnforce(F):
    U, S, Vt = np.linalg.svd(F)  # Compute SVD
    S[-1] = 0  # Set the smallest singular value to zero
    F_enforced = U @ np.diag(S) @ Vt  # Reconstruct F with rank 2
    # print("\n\n\n", U, np.diag(S), Vt, "\n\n\n", sep='\n')
    # print(U @ np.diag(S))
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
    # print(pts1[:5])
    # print(pts1_N[:5])
    num = len(pts1)
    x2 = np.reshape(pts2_N[:, 0], (num, 1))
    y2 = np.reshape(pts2_N[:, 1], (num, 1))
    x1 = np.reshape(pts1_N[:, 0], (num, 1))
    y1 = np.reshape(pts1_N[:, 1], (num, 1))


    # Contraint Matrix formation
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
    # print(f"f = \n\t{f},\n\nF = \n\t{F}")
    
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
    # print(x1, y1, a, b, c)
    # im1_ = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2_ = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
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
        # print(x2, y2)
        mat2 = im2_[y2 - 2: y2 + 3, x2 - 2 : x2 + 3]
        if(mat2.shape != mat1.shape):
            continue
        diff = np.int64(mat2 - mat1)
        sqdiff = diff ** 2
        ssd = sum(sum(sum(sqdiff)))
        # print(f"SSD = {ssd}, \nmat2 = {mat2}, \nmat1 = {mat1}")
        if(ssd < min_ssd):
            min_ssd = ssd
            bestMatch = np.array([x2, y2])
    return bestMatch[0], bestMatch[1]

def epipolar_correspondences(im1, im2, F, pts):
    print(pts.shape)
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
    # print(np.shape(x_o))
    # print(np.shape(y_o))
    return corrs
    # a = np.reshape(l[:, 0], ())


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

def getRotationMatrices(K1, K2, F):
    E = essential_matrix(F, K1, K2)
    U, D, Vt = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = np.zeros((3, 1))
    t2 = U[:, 2]
    print("T1, grp = ", t1)
    return R1, R2, t1, t2

def getP1(K1, R1, t1):
    print(f"R1 = {R1}, \n\nt1 = {t1}, \n\nK1 = {K1}")
    temM1 = np.hstack((R1, t1))
    P1 = K1 @ temM1
    return P1

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
# def triangulate(P1, pts1, P2, pts2):
#     points = []
#     M = len(pts1)
#     print(P1, P2)
#     pts1 = np.int64(pts1)
#     pts2 = np.int64(pts2)
#     for i in range(M):
#         x1 = pts1[i][0]
#         y1 = pts1[i][0]
#         x2 = pts2[i][0]
#         y2 = pts2[i][0]
#         p3T1 = (P1[2, :])        
#         p2T1 = (P1[1, :])        
#         p1T1 = (P1[0, :])        
#         p3T2 = (P2[2, :])        
#         p2T2 = (P2[1, :])        
#         p1T2 = (P2[0, :])    

#         row1 = y1 * p3T1 - p2T1    
#         row2 = -x1 * p3T1 - p1T1    
#         row3 = y2 * p3T2 - p2T2    
#         row4 = -x2 * p3T2 - p1T2
#         Ai = np.vstack((row1, row2, row3, row4))
#         print(f"\n\nAi = \n {Ai}\n\n")
#         U, D, Vt = np.linalg.svd(Ai)
#         print(print(Vt.T[:, -1]))
#         points.append(Vt.T[:, -1])
#     return points

def triangulate(P1, pts1, P2, pts2):
    """
    Triangulate 3D points from 2D correspondences in two views.

    Args:
        P1: (3x4) Projection matrix for camera 1
        pts1: (Nx2) 2D points in image 1
        P2: (3x4) Projection matrix for camera 2
        pts2: (Nx2) 2D points in image 2

    Returns:
        points_3D: (Nx3) Triangulated 3D points
        reproj_error: (N,) Reprojection error for each point
    """
    points_3D = []
    errors = []
    M = len(pts1)

    pts1 = np.int64(pts1)
    pts2 = np.int64(pts2)

    for i in range(M):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Construct A matrix for triangulation
        A = np.vstack([
            y1 * P1[2, :] - P1[1, :],
            -x1 * P1[2, :] + P1[0, :],
            y2 * P2[2, :] - P2[1, :],
            -x2 * P2[2, :] + P2[0, :]
        ])

        # Solve using SVD
        U, D, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]  # Last column of V
        X_homogeneous /= X_homogeneous[-1]  # Convert to inhomogeneous coordinates
        X = X_homogeneous[:3]  # Extract 3D point

        points_3D.append(X)

        # Compute Reprojection Error
        x1_proj = P1 @ X_homogeneous
        x1_proj /= x1_proj[2]
        x2_proj = P2 @ X_homogeneous
        x2_proj /= x2_proj[2]

        error1 = np.linalg.norm(x1_proj[:2] - pts1[i])
        error2 = np.linalg.norm(x2_proj[:2] - pts2[i])

        errors.append(error1 + error2)

    return np.array(points_3D), np.array(errors)

def extract_extrinsics(P):
    """
    Extracts the rotation matrix (R) and translation vector (t) from a projection matrix P.

    Args:
        P: (3x4) Camera projection matrix

    Returns:
        R: (3x3) Rotation matrix
        t: (3x1) Translation vector
    """
    M = P[:, :3]  # Extract left 3x3 part
    t = P[:, 3]   # Extract right 3x1 translation

    # Use RQ decomposition to get R and intrinsic matrix
    K, R = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)

    return R, t.reshape(3, 1)

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
    # Compute optical centers
    C1 = -np.linalg.inv(R1) @ t1  # Optical center of Camera 1
    C2 = -np.linalg.inv(R2) @ t2  # Optical center of Camera 2
    
    # Compute new X-axis (direction of baseline)
    x_axis = (C2 - C1).flatten()
    x_axis /= np.linalg.norm(x_axis)
    
    # Compute new Z-axis (average of current Z-axes)
    z_axis = (R1[:, 2] + R2[:, 2]) / 2.0
    z_axis /= np.linalg.norm(z_axis)
    
    # Compute new Y-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # Orthonormal rectification rotation matrix
    R_rect = np.vstack([x_axis, y_axis, z_axis])
    
    # Compute rectified rotation matrices
    R1p = R_rect @ R1
    R2p = R_rect @ R2
    
    # Rectification matrices
    M1 = R1p @ np.linalg.inv(R1)
    M2 = R2p @ np.linalg.inv(R2)
    
    # Rectified camera matrices
    K1p = K1  # Keep the same intrinsics
    K2p = K2
    
    # Rectified translation vectors
    t1p = M1 @ t1
    t2p = M2 @ t2
    
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
    """
    Computes the disparity map using the SSD block-matching algorithm.

    Args:
        im1: Left image (H1xW1 grayscale matrix)
        im2: Right image (H2xW2 grayscale matrix)
        max_disp: Maximum disparity value (integer)
        win_size: Window size for block matching (odd integer)

    Returns:
        dispM: Disparity map (H1xW1 matrix)
    """

    # Ensure images are grayscale (convert if necessary)
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    H, W = im1.shape

    # Half window size (for easy indexing)
    half_win = win_size // 2

    # Initialize disparity map
    dispM = np.zeros((H, W), dtype=np.float32)

    # Pad images to handle window borders
    im1_padded = cv2.copyMakeBorder(im1, half_win, half_win, half_win, half_win, cv2.BORDER_CONSTANT, 0)
    im2_padded = cv2.copyMakeBorder(im2, half_win, half_win, half_win + max_disp, half_win, cv2.BORDER_CONSTANT, 0)

    # Compute disparity map using SSD
    for y in range(half_win, H + half_win):
        for x in range(half_win, W + half_win):
            # Extract the reference block in the left image
            ref_block = im1_padded[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]

            best_disp = 0
            min_ssd = float('inf')

            # Search within the disparity range
            for d in range(max_disp):
                if x - d < half_win:  # Ensure valid disparity search
                    continue

                # Extract the block in the right image
                target_block = im2_padded[y - half_win:y + half_win + 1, x - half_win - d:x + half_win + 1 - d]

                # Compute Sum of Squared Differences (SSD)
                ssd = np.sum((ref_block - target_block) ** 2)

                # Update best disparity
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d

            # Assign best disparity value
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
    """
    Computes the depth map from the disparity map.

    Args:
        dispM: Disparity map (H1xW1 matrix)
        K1, K2: Intrinsic camera matrices (3x3)
        R1, R2: Rotation matrices (3x3)
        t1, t2: Translation vectors (3x1)

    Returns:
        depthM: Depth map (H1xW1 matrix)
    """

    # Extract focal length (assuming same focal length for both cameras)
    f = K1[0, 0]  # Focal length from intrinsic matrix K1

    # Compute baseline distance B (distance between optical centers)
    C1 = -R1.T @ t1  # Optical center of first camera
    C2 = -R2.T @ t2  # Optical center of second camera
    B = np.linalg.norm(C2 - C1)  # Euclidean distance between centers

    # Avoid division by zero (set invalid disparities to a small value)
    dispM[dispM == 0] = 1e-6

    # Compute depth map using Z = (B * f) / disparity
    depthM = (B * f) / dispM

    return depthM



"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    """
    Estimates the camera projection matrix P using the Direct Linear Transformation (DLT).

    Args:
        x: 2D image points (Nx2 matrix)
        X: 3D world points (Nx3 matrix)

    Returns:
        P: Camera projection matrix (3x4 matrix)
    """

    # Number of correspondences
    N = x.shape[0]

    # Convert to homogeneous coordinates
    X_h = np.hstack((X, np.ones((N, 1))))  # Convert 3D points to homogeneous (Nx4)
    x_h = np.hstack((x, np.ones((N, 1))))  # Convert 2D points to homogeneous (Nx3)

    # Construct the linear system Ax = 0
    A = []
    for i in range(N):
        X_i = X_h[i, :]  # Homogeneous 3D point (1x4)
        u, v, w = x_h[i, :]  # Homogeneous 2D point

        A.append([0, 0, 0, 0, -w * X_i[0], -w * X_i[1], -w * X_i[2], -w * X_i[3], v * X_i[0], v * X_i[1], v * X_i[2], v * X_i[3]])
        A.append([w * X_i[0], w * X_i[1], w * X_i[2], w * X_i[3], 0, 0, 0, 0, -u * X_i[0], -u * X_i[1], -u * X_i[2], -u * X_i[3]])

    A = np.array(A)  # Convert to matrix form (2N x 12)

    # Solve using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)

    # The solution is the last column of V (or last row of Vt)
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
    """
    Estimates camera intrinsics (K) and extrinsics (R, t) from the projection matrix P.

    Args:
        P: Camera projection matrix (3x4)

    Returns:
        K: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
    """

    # Extract the first 3x3 part of P (which is K * R)
    M = P[:, :3]

    # Perform RQ decomposition to get K and R
    K, R = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)

    # Ensure a positive diagonal in K
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T  # Adjust K
    R = T @ R  # Adjust R to maintain equivalence

    # Extract translation vector t
    t = np.linalg.inv(K) @ P[:, 3]

    return K, R, t.reshape(3, 1)

# def triangulate(P1, pts1, P2, pts2):
#     points = []
#     M = len(pts1)
#     print(P1, P2)
#     pts1 = np.int64(pts1)
#     pts2 = np.int64(pts2)
#     errors = []
#     for i in range(M):
#         x1, y1 = pts1[i]
#         x2, y2 = pts2[i]

#         p3T1 = (P1[2, :])        
#         p2T1 = (P1[1, :])        
#         p1T1 = (P1[0, :])        
#         p3T2 = (P2[2, :])        
#         p2T2 = (P2[1, :])        
#         p1T2 = (P2[0, :])    

#         row1 = y1 * p3T1 - p2T1    
#         row2 = -x1 * p3T1 - p1T1    
#         row3 = y2 * p3T2 - p2T2    
#         row4 = -x2 * p3T2 - p1T2
#         Ai = np.vstack((row1, row2, row3, row4))
#         # print(f"\n\nAi = \n {Ai}\n\n")
#         U, D, Vt = np.linalg.svd(Ai)
#         # print(print(Vt.T[:, -1]))
#         X_h = Vt[-1] / Vt[-1][-1]  # Correct homogeneous division
#         X = X_h[:3]
#         points.append(X)
#         # print(X_h.shape)
#         # print(P1.shape)
#         x1_proj = P1 @ X_h
#         x1_proj /= x1_proj[2]
#         x2_proj = P2 @ X_h
#         x2_proj /= x2_proj[2]

#         error1 = np.linalg.norm(x1_proj[:2] - pts1[i])
#         error2 = np.linalg.norm(x2_proj[:2] - pts2[i])

#         errors.append(error1 + error2)

#     error = np.mean(errors)
#     print(f"Error = {error}")
#     # print(f"Error = {compute_reprojection_error(P1, np.array(points), pts1)}")
#     return points, error

def triangulate(P1, pts1, P2, pts2):
    points = []
    errors = []
    M = len(pts1)

    # Ensure floating point precision
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)

    for i in range(M):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Construct A matrix correctly
        A = np.array([
            y1 * P1[2, :] - P1[1, :],
            x1 * P1[2, :] - P1[0, :],
            y2 * P2[2, :] - P2[1, :],
            x2 * P2[2, :] - P2[0, :]
        ])

        # Normalize rows to avoid numerical instability
        A = A / np.linalg.norm(A, axis=1, keepdims=True)

        # Solve using SVD
        U, D, Vt = np.linalg.svd(A)
        X_h = Vt[-1] / Vt[-1][-1]  # Normalize homogeneous coordinates
        X = X_h[:3]  # Extract 3D point
        points.append(X)

        # Compute reprojection error
        x1_proj = P1 @ X_h
        x1_proj /= x1_proj[2]  # Convert to Euclidean
        
        x2_proj = P2 @ X_h
        x2_proj /= x2_proj[2]  # Convert to Euclidean

        error1 = np.linalg.norm(x1_proj[:2] - pts1[i])
        error2 = np.linalg.norm(x2_proj[:2] - pts2[i])
        errors.append(error1 + error2)

    error = np.mean(errors)
    print(f"Mean Reprojection Error = {error:.4f}")
    
    return np.array(points), error



def triangulate2(P1, pts1, P2, pts2):
    points = []
    M = len(pts1)
    print(P1, P2)
    pts1 = np.int64(pts1)
    pts2 = np.int64(pts2)
    errors = []
    for i in range(M):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        p3T1 = (P1[2, :])        
        p2T1 = (P1[1, :])        
        p1T1 = (P1[0, :])        
        p3T2 = (P2[2, :])        
        p2T2 = (P2[1, :])        
        p1T2 = (P2[0, :])    

        row1 = y1 * p3T1 - p2T1    
        row2 = -x1 * p3T1 - p1T1    
        row3 = y2 * p3T2 - p2T2    
        row4 = -x2 * p3T2 - p1T2
        Ai = np.vstack((row1, row2, row3, row4))

        U, D, Vt = np.linalg.svd(Ai)

        X_h = Vt[-1] / Vt[-1][-1]  
        X = X_h[:3]
        points.append(X)

        x1_proj = P1 @ X_h
        x1_proj /= x1_proj[2]
        x2_proj = P2 @ X_h
        x2_proj /= x2_proj[2]

        error1 = np.linalg.norm(x1_proj[:2] - pts1[i])
        error2 = np.linalg.norm(x2_proj[:2] - pts2[i])

        errors.append(error1 + error2)

    error = np.mean(errors)
    print(f"Error = {error}")

    return points, error
