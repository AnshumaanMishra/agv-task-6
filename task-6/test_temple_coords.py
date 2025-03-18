import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import cv2
import submission as sub
import matplotlib.pyplot as plt


# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load("task-6/data/some_corresp.npz")
print(data.files)

pts1 = data["pts1"]
pts2 = data["pts2"]
im1 = cv2.imread("task-6/data/im1.png")
im2 = cv2.imread("task-6/data/im2.png")

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

M = max(len(im1), len(im1[0]))


# 2. Run eight_point to compute F

F, T1, T2 = sub.eight_point(pts1, pts2, M)
F_enf = sub.rankEnforce(F)
F_deN = T2.T @ F_enf @ T1


# 3. Load points in image 1 from data/temple_coords.npz

pts1_2 = np.load("task-6/data/temple_coords.npz")

# 4. Run epipolar_correspondences to get points in image 2

pts2_c = sub.epipolar_correspondences(im1, im2, F_deN, pts1_2['pts1'])
print(*F_deN, sep="\n")
hlp.epipolarMatchGUI(im1, im2, F_deN)
# hlp.displayEpipolarF(im1, im2, F_deN)


# 5. Compute the camera projection matrix P1

KData = np.load("task-6/data/intrinsics.npz")
K1 = KData['K1']
K2 = KData['K2']
E = sub.essential_matrix(F, K1, K2)

[u, d] = np.linalg.eig(F.T @ F)
# print(u, d)

minval = np.argmin(u)
uu = d[:, 2]
uu /= uu[2]

R1, R2, t1, t2 = sub.getRotationMatrices(K1, K2, F)
P1 =sub.getP1(K1, R1, t1)


# 6. Use camera2 to get 4 camera projection matrices P2

P2s = hlp.camera2(E)

templecoords = np.load("task-6/data/temple_coords.npz")
pts1_t = templecoords['pts1']
pts2_t = sub.epipolar_correspondences(im1, im2, F_deN, pts1_t)


# 7. Run triangulate using the projection matrices

P21pts, err1 = sub.triangulate(P1, pts1_t, P2s[:, :, 0], pts2_t)
P22pts, err2 = sub.triangulate(P1, pts1_t, P2s[:, :, 1], pts2_t)
P23pts, err3 = sub.triangulate(P1, pts1_t, P2s[:, :, 2], pts2_t)
P24pts, err4 = sub.triangulate(P1, pts1_t, P2s[:, :, 3], pts2_t)

# 8. Figure out the correct P2

errors = [err1, err2, err3, err4]
print(errors)
best_P2_index = np.argmin(errors)
best_3D_points = np.array([P21pts, P22pts, P23pts, P24pts])[best_P2_index]
best_P2 = P2s[:, :, best_P2_index]


# 9. Scatter plot the correct 3D points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-20, azim=100)
X, Y, Z = best_3D_points[:, 0], best_3D_points[:, 1], best_3D_points[:, 2]

ax.scatter(X, Y, Z, c='blue', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Triangulated 3D Points')
plt.savefig('triangulated_points.png')

plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

# R1, t1 = sub.extract_extrinsics(P1)
# R2, t2 = sub.extract_extrinsics(best_P2)

np.savez('task-6/data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)
