import cv2 as cv
import numpy as np
import helper as hlp
import submission as sub
import numpy.linalg as la
import skimage.color as col
import matplotlib.pyplot as plt

# 1. Load the images and the parameters

I1 = cv.cvtColor(cv.imread('task-6/data/im1.png'), cv.COLOR_BGR2GRAY).astype(np.float32)
I2 = cv.cvtColor(cv.imread('task-6/data/im2.png'), cv.COLOR_BGR2GRAY).astype(np.float32)

intrinsics = np.load('task-6/data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

extrinsics = np.load('task-6/data/extrinsics.npz')
R1, R2 = extrinsics['R1'], extrinsics['R2']
t1, t2 = extrinsics['t1'], extrinsics['t2']
print(R1, R2, t1, t2, sep="\n")

# 2. Rectify the images and save the paramters

M1, M2, K1p, K2p, R1p, R2p, t1p, t2p = sub.rectify_pair(K1, K2, R1, R2, t1, t2)
np.savez('task-6/data/rectify.npz', M1=M1, M2=M2, K1p=K1p, K2p=K2p, R1p=R1p, R2p=R2p, t1p=t1p, t2p=t2p)

# 3. Warp and display the result
print("Reached")
print(I2.shape, I1.shape)
I1, I2, bb = hlp.warpStereo(I1, I2, M1, M2)
print("Reached")

r, c = I1.shape
I = np.zeros((r, 2*c))
I[:,:c] = I1
I[:,c:] = I2
print(I.shape)
corresp = np.load('task-6/data/some_corresp.npz')
pts1, pts2 = corresp['pts1'][::18].T, corresp['pts2'][::18].T
print("Reached")
pts1, pts2 = hlp._projtrans(M1, pts1), hlp._projtrans(M2, pts2)
print("Reached")
pts2[0,:] = pts2[0,:] + c
fig, ax = plt.subplots(figsize=(16, 9))
# plt.figure(figsize=(10, 10))
# plt.yticks(range(2000, 1, -100))
ax.imshow(I, cmap='gray', aspect='auto')
# ax.set_aspect(1)
ax.scatter(pts1[0,:], pts1[1,:], s=60, c='r', marker='*')
ax.scatter(pts2[0,:], pts2[1,:], s=60, c='r', marker='*')
for p1, p2 in zip(pts1.T, pts2.T):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], '-', c='b')
    print([p1[0],p2[0]], [p1[1],p2[1]])

plt.savefig("rectified.png")
plt.show()
