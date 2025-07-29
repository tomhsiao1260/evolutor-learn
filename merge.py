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

def apply_piecewise_linear(data, x, b):
    flat_data = data.flatten()
    interp_vals = np.interp(flat_data, x, b)

    return interp_vals.reshape(data.shape)

def apply_piecewise_linear_extrap(data, x, b):
    f = interp1d(x, b, kind='linear', fill_value='extrapolate', assume_sorted=True)
    flat_data = data.flatten()
    interp_vals = f(flat_data)
    return interp_vals.reshape(data.shape)

def sparseBoundary(shape):
    nrf, ncf = shape
    nr = nrf-1
    nc = ncf-1
    n1df_flat = np.arange(nrf*ncf)
    n1df = np.reshape(n1df_flat, shape)

    boundary_indices = np.concatenate([
        n1df[0, :],
        n1df[-1, :],
        n1df[1:-1, 0],
        n1df[1:-1, -1],
    ])

    data = np.ones(len(boundary_indices))
    row = np.zeros(len(boundary_indices))
    col = boundary_indices
    sparse_boundary = sparse.coo_array((data, (row, col)), shape=(nrf*ncf, nrf*ncf))
    return sparse_boundary

def solveV(basew_, basew, smoothing_weight, st):
    vecu = st.vector_u
    coh = st.coherence[:,:,np.newaxis]
    wvecu = coh*vecu

    vecu = st.vector_u
    coh = st.coherence[:,:,np.newaxis]
    wvecu = coh*vecu

    shape = wvecu.shape[:2]
    sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=True)
    sparse_grad = ImageViewer.sparseGrad(shape)
    sparse_boundary = sparseBoundary(shape)
    # sparse_u_cross_grad = sparse.vstack((sparse_uxg, sparse_boundary))
    sparse_u_cross_grad = sparse.vstack((sparse_uxg, smoothing_weight*sparse_grad))
    # sparse_u_cross_grad = sparse.vstack((sparse_uxg, sparse_boundary, smoothing_weight*sparse_grad))

    A = sparse_u_cross_grad
    # print("A", A.shape, A.dtype)

    mask = np.ones_like(basew_, dtype=bool)
    # mask[:, :] = False
    mask[30:-30, 30:-30] = False
    mask = mask.flatten()

    b = -sparse_u_cross_grad @ basew_.flatten()
    # b[:basew_.size] = 0.
    # b[:basew_.size][mask] = 0.
    b[basew_.size:] = 0.
    x = solveAxEqb(A, b)
    out = x.reshape(basew_.shape)
    out += basew_

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
    data_oi = solveV(data_i_, data_i, smoothing_weight, st_i)

    st_j = ST(data_j_img)
    st_j.computeEigens()
    smoothing_weight = .1
    data_j_ = cv2.resize(data_j_, (0, 0), fx=2, fy=2)
    data_oj = solveV(data_j_, data_j, smoothing_weight, st_j)

    st_k = ST(data_k_img)
    st_k.computeEigens()
    smoothing_weight = .1
    data_k_ = cv2.resize(data_k_, (0, 0), fx=2, fy=2)
    data_ok = solveV(data_k_, data_k, smoothing_weight, st_k)

    st_l = ST(data_l_img)
    st_l.computeEigens()
    smoothing_weight = .1
    data_l_ = cv2.resize(data_l_, (0, 0), fx=2, fy=2)
    data_ol = solveV(data_l_, data_l, smoothing_weight, st_l)

    h, w = chunk, chunk
    y, x = np.mgrid[0:h, 0:w]
    y, x = (y+1e-5)/(h+1e-5), (x+1e-5)/(w+1e-5)

    data_oi_t = apply_piecewise_linear(data_oi, data_oi[0, :], data_i_[0, :])
    data_oi_b = apply_piecewise_linear(data_oi, data_oi[-1, :], data_i_[-1, :])
    data_oi_l = apply_piecewise_linear(data_oi, data_oi[:, 0], data_i_[:, 0])
    data_oi_r = apply_piecewise_linear(data_oi, data_oi[:, -1], data_i_[:, -1])
    data_oi = 1/y * data_oi_t + 1/(1-y) * data_oi_b + 1/x * data_oi_l + 1/(1-x) * data_oi_r
    data_oi /= 1/y + 1/(1-y) + 1/x + 1/(1-x)

    data_oj_t = apply_piecewise_linear(data_oj, data_oj[0, :], data_j_[0, :])
    data_oj_b = apply_piecewise_linear(data_oj, data_oj[-1, :], data_j_[-1, :])
    data_oj_l = apply_piecewise_linear(data_oj, data_oj[:, 0], data_j_[:, 0])
    data_oj_r = apply_piecewise_linear(data_oj, data_oj[:, -1], data_j_[:, -1])
    data_oj = 1/y * data_oj_t + 1/(1-y) * data_oj_b + 1/x * data_oj_l + 1/(1-x) * data_oj_r
    data_oj /= 1/y + 1/(1-y) + 1/x + 1/(1-x)

    data_ok_t = apply_piecewise_linear(data_ok, data_ok[0, :], data_k_[0, :])
    data_ok_b = apply_piecewise_linear(data_ok, data_ok[-1, :], data_k_[-1, :])
    data_ok_l = apply_piecewise_linear(data_ok, data_ok[:, 0], data_k_[:, 0])
    data_ok_r = apply_piecewise_linear(data_ok, data_ok[:, -1], data_k_[:, -1])
    data_ok = 1/y * data_ok_t + 1/(1-y) * data_ok_b + 1/x * data_ok_l + 1/(1-x) * data_ok_r
    data_ok /= 1/y + 1/(1-y) + 1/x + 1/(1-x)

    data_ol_t = apply_piecewise_linear(data_ol, data_ol[0, :], data_l_[0, :])
    data_ol_b = apply_piecewise_linear(data_ol, data_ol[-1, :], data_l_[-1, :])
    data_ol_l = apply_piecewise_linear(data_ol, data_ol[:, 0], data_l_[:, 0])
    data_ol_r = apply_piecewise_linear(data_ol, data_ol[:, -1], data_l_[:, -1])
    data_ol = 1/y * data_ol_t + 1/(1-y) * data_ol_b + 1/x * data_ol_l + 1/(1-x) * data_ol_r
    data_ol /= 1/y + 1/(1-y) + 1/x + 1/(1-x)

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
