import sys
import zarr
import cmap
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import sparse

def apply_piecewise_linear(data, x, b):
    flat_data = data.flatten()
    interp_vals = np.interp(flat_data, x, b)

    return interp_vals.reshape(data.shape)

def apply_piecewise_linear_extrap(data, x, b):
    f = interp1d(x, b, kind='linear', fill_value='extrapolate', assume_sorted=True)
    flat_data = data.flatten()
    interp_vals = f(flat_data)
    return interp_vals.reshape(data.shape)

def main():
    input_zarr_dir = '/Users/yao/Desktop/Vesuvius/evolutor-visualize/public/scroll_v.zarr/2/'
    input_zarr_dir_ = '/Users/yao/Desktop/Vesuvius/evolutor-visualize/public/scroll_v.zarr/3/'

    chunk = 128
    yi, xi = 896, 896
    yj, xj = 896, 1024
    yk, xk = 1024, 896
    yl, xl = 1024, 1024

    z = zarr.open(input_zarr_dir, mode='r')
    z_ = zarr.open(input_zarr_dir_, mode='r')

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

    data_i -= np.min(data_i)
    data_i /= np.max(data_i)
    data_i *= np.max(data_i_) - np.min(data_i_)
    data_i += np.min(data_i_)

    data_j -= np.min(data_j)
    data_j /= np.max(data_j)
    data_j *= np.max(data_j_) - np.min(data_j_)
    data_j += np.min(data_j_)

    data_k -= np.min(data_k)
    data_k /= np.max(data_k)
    data_k *= np.max(data_k_) - np.min(data_k_)
    data_k += np.min(data_k_)

    data_l -= np.min(data_l)
    data_l /= np.max(data_l)
    data_l *= np.max(data_l_) - np.min(data_l_)
    data_l += np.min(data_l_)

    hi, wi = data_i.shape
    yi, xi = np.mgrid[:hi, :wi]
    yi, xi = (yi+0.0001)/hi, (xi+0.0001)/wi
    # data_i_t = apply_piecewise_linear(data_i, data_i[0, :][::2], data_i_[0, :])
    # data_i_d = apply_piecewise_linear(data_i, data_i[-1, :][::2], data_i_[-1, :])
    # data_i_l = apply_piecewise_linear(data_i, data_i[:, 0][::2], data_i_[:, 0])
    # data_i_r = apply_piecewise_linear(data_i, data_i[:, -1][::2], data_i_[:, -1])
    # data_i = 1/(1-yi)*data_i_d + 1/(1-xi)*data_i_r
    # data_i = data_i_r
    # data_i /= 1/(1-yi) + 1/(1-xi)
    # data_i = 1/yi*data_i_t + 1/(1-yi)*data_i_d + 1/xi*data_i_l + 1/(1-xi)*data_i_r
    # data_i /= 1/yi + 1/(1-yi) + 1/xi + 1/(1-xi)

    hj, wj = data_j.shape
    yj, xj = np.mgrid[:hj, :wj]
    yj, xj = (yj+0.0001)/hj, (xj+0.0001)/wj
    data_j_t = apply_piecewise_linear(data_j, data_j[0, :][::2], data_j_[0, :])
    data_j_d = apply_piecewise_linear(data_j, data_j[-1, :][::2], data_j_[-1, :])
    data_j_l = apply_piecewise_linear(data_j, data_j[:, 0], data_i[:, -1])
    # data_j_l = apply_piecewise_linear(data_j, data_j[:, 0][::2], data_j_[:, 0])
    data_j_r = apply_piecewise_linear(data_j, data_j[:, -1][::2], data_j_[:, -1])
    data_j = 1/(1-yj)*data_j_d + 1/xj*data_j_l
    data_j = data_j_l
    # data_j /= 1/(1-yj) + 1/xj
    # data_j = 1/yj*data_j_t + 1/(1-yj)*data_j_d + 1/xj*data_j_l + 1/(1-xj)*data_j_r
    # data_j /= 1/yj + 1/(1-yj) + 1/xj + 1/(1-xj)

    # bi = np.hstack((data_i_[0, :], data_i_[-1, :], data_i_[:, 0], data_i_[:, -1]))
    # xi = np.hstack((data_i[0, :], data_i[-1, :], data_i[:, 0], data_i[:, -1]))
    # xi = xi[::2]
    # data_i = apply_piecewise_linear(data_i, xi, bi)

    # bj = np.hstack((data_j_[0, :], data_j_[-1, :], data_j_[:, 0], data_j_[:, -1]))
    # xj = np.hstack((data_j[0, :], data_j[-1, :], data_j[:, 0], data_j[:, -1]))
    # xj = xj[::2]
    # data_j = apply_piecewise_linear(data_j, xj, bj)

    bk = np.hstack((data_k_[:, -1]))
    xk = np.hstack((data_k[:, -1]))
    xk = xk[::2]
    data_k = apply_piecewise_linear(data_k, xk, bk)

    bl = np.hstack((data_l_[:, 0]))
    xl = np.hstack((data_l[:, 0]))
    xl = xl[::2]
    data_l = apply_piecewise_linear(data_l, xl, bl)

    # bk = np.hstack((data_k_[0, :], data_k_[-1, :], data_k_[:, 0], data_k_[:, -1]))
    # xk = np.hstack((data_k[0, :], data_k[-1, :], data_k[:, 0], data_k[:, -1]))
    # xk = xk[::2]
    # data_k = apply_piecewise_linear(data_k, xk, bk)

    # bl = np.hstack((data_l_[0, :], data_l_[-1, :], data_l_[:, 0], data_l_[:, -1]))
    # xl = np.hstack((data_l[0, :], data_l[-1, :], data_l[:, 0], data_l[:, -1]))
    # xl = xl[::2]
    # data_l = apply_piecewise_linear(data_l, xl, bl)

    data_ij = np.hstack((data_i, data_j))
    data_kl = np.hstack((data_k, data_l))
    data_ij_ = np.hstack((data_i_, data_j_))
    data_kl_ = np.hstack((data_k_, data_l_))
    # data_ij = apply_piecewise_linear_extrap(data_ij, data_ij[-1, :][::2], data_ij_[-1, :])
    # data_kl = apply_piecewise_linear_extrap(data_kl, data_kl[0, :][::2], data_kl_[0, :])
    # data_ij = np.hstack((data_i_, data_j_))
    # data_kl = np.hstack((data_k_, data_l_))
    data = np.vstack((data_ij, data_kl))

    colormap = cmap.Colormap("tab20", interpolation="nearest")
    colored_image = colormap(data)

    plt.imshow(colored_image, aspect='equal')  
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
