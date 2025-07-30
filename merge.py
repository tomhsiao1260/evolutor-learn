import sys
import zarr
import cmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import sparse
from st import ST
from wind2d_flat import ImageViewer

def solveAxEqb(A, b):
    print("solving Ax = b", A.shape, b.shape)
    At = A.transpose()
    AtA = At @ A
    # print("AtA", AtA.shape, sparse.issparse(AtA))
    # print("ata", ata.shape, ata.dtype, ata[ata!=0], np.argwhere(ata))
    asum = np.abs(AtA).sum(axis=0)
    # print("asum", np.argwhere(asum==0))
    Atb = At @ b
    # print("Atb", Atb.shape, sparse.issparse(Atb))

    lu = sparse.linalg.splu(AtA.tocsc())
    # print("lu created")
    x = lu.solve(Atb)
    print("x", x.shape, x.dtype, x.min(), x.max())
    return x

def solveV(basew, smoothing_weight, st):
    vecu = st.vector_u
    coh = st.coherence[:,:,np.newaxis]
    wvecu = coh*vecu

    vecu = st.vector_u
    coh = st.coherence[:,:,np.newaxis]
    wvecu = coh*vecu

    shape = wvecu.shape[:2]
    sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=True)
    sparse_grad = ImageViewer.sparseGrad(shape)
    sparse_u_cross_grad = sparse.vstack((sparse_uxg, smoothing_weight*sparse_grad))

    A = sparse_u_cross_grad
    # print("A", A.shape, A.dtype)

    b = -sparse_u_cross_grad @ basew.flatten()
    b[basew.size:] = 0.
    x = solveAxEqb(A, b)
    out = x.reshape(basew.shape)

    h, w = out.shape
    y, x = np.mgrid[:h, :w]
    y, x = y/(h-1), x/(w-1)
    y, x = 2*y-1, 2*x-1
    y, x = y**2, x**2
    y, x = 1-y, 1-x   # center: 1, edge: 0

    mask = np.minimum(y, x)
    out *= mask
    out += basew

    return out

def main():
    input_zarr_dir_img = '/Users/yao/Desktop/Vesuvius/evolutor-visualize/public/scroll.zarr/2/'
    input_zarr_dir = '/Users/yao/Desktop/Vesuvius/evolutor-visualize/public/scroll_v.zarr/2/'
    input_zarr_dir_ = '/Users/yao/Desktop/Vesuvius/evolutor-visualize/public/scroll_v.zarr/3/'

    chunk = 128
    yi, xi = 896, 896
    yj, xj = 896, 1024
    yk, xk = 1024, 896
    yl, xl = 1024, 1024

    z_img = zarr.open(input_zarr_dir_img, mode='r')
    z = zarr.open(input_zarr_dir, mode='r')
    z_ = zarr.open(input_zarr_dir_, mode='r')

    data_i_img = z_img[0, yi:yi+chunk, xi:xi+chunk]
    data_j_img = z_img[0, yj:yj+chunk, xj:xj+chunk]
    data_k_img = z_img[0, yk:yk+chunk, xk:xk+chunk]
    data_l_img = z_img[0, yl:yl+chunk, xl:xl+chunk]

    data_i_img = np.array(data_i_img).astype(np.float32) / 65535.
    data_j_img = np.array(data_j_img).astype(np.float32) / 65535.
    data_k_img = np.array(data_k_img).astype(np.float32) / 65535.
    data_l_img = np.array(data_l_img).astype(np.float32) / 65535.

    data_i = z[0, yi:yi+chunk, xi:xi+chunk]
    data_j = z[0, yj:yj+chunk, xj:xj+chunk]
    data_k = z[0, yk:yk+chunk, xk:xk+chunk]
    data_l = z[0, yl:yl+chunk, xl:xl+chunk]

    data_i = np.array(data_i).astype(np.float32) / 65535.
    data_j = np.array(data_j).astype(np.float32) / 65535.
    data_k = np.array(data_k).astype(np.float32) / 65535.
    data_l = np.array(data_l).astype(np.float32) / 65535.

    data_i_ = z_[0, yi//2:yi//2+chunk//2, xi//2:xi//2+chunk//2]
    data_j_ = z_[0, yj//2:yj//2+chunk//2, xj//2:xj//2+chunk//2]
    data_k_ = z_[0, yk//2:yk//2+chunk//2, xk//2:xk//2+chunk//2]
    data_l_ = z_[0, yl//2:yl//2+chunk//2, xl//2:xl//2+chunk//2]

    data_i_ = np.array(data_i_).astype(np.float32) / 65535.
    data_j_ = np.array(data_j_).astype(np.float32) / 65535.
    data_k_ = np.array(data_k_).astype(np.float32) / 65535.
    data_l_ = np.array(data_l_).astype(np.float32) / 65535.

    st_i = ST(data_i_img)
    st_i.computeEigens()
    smoothing_weight = .1
    data_i_ = cv2.resize(data_i_, (0, 0), fx=2, fy=2)
    data_oi = solveV(data_i_, smoothing_weight, st_i)

    st_j = ST(data_j_img)
    st_j.computeEigens()
    smoothing_weight = .1
    data_j_ = cv2.resize(data_j_, (0, 0), fx=2, fy=2)
    data_oj = solveV(data_j_, smoothing_weight, st_j)

    st_k = ST(data_k_img)
    st_k.computeEigens()
    smoothing_weight = .1
    data_k_ = cv2.resize(data_k_, (0, 0), fx=2, fy=2)
    data_ok = solveV(data_k_, smoothing_weight, st_k)

    st_l = ST(data_l_img)
    st_l.computeEigens()
    smoothing_weight = .1
    data_l_ = cv2.resize(data_l_, (0, 0), fx=2, fy=2)
    data_ol = solveV(data_l_, smoothing_weight, st_l)

    data_ij = np.hstack((data_oi, data_oj))
    data_kl = np.hstack((data_ok, data_ol))
    data = np.vstack((data_ij, data_kl))

    colormap = cmap.Colormap("tab20", interpolation="nearest")
    colored_image = colormap(data)

    plt.imshow(colored_image, aspect='equal')  
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    sys.exit(main())

