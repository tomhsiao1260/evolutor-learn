import sys
import zarr
import cmap
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

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

def main():
    input_zarr_dir = '/Users/yao/Desktop/Vesuvius/evolutor-visualize/public/scroll_u.zarr/2/'

    chunk = 128

    yi, xi = 896, 896
    yj, xj = 896, 1024
    yk, xk = 1024, 896
    yl, xl = 1024, 1024

    z = zarr.open(input_zarr_dir, mode='r')

    data_i = z[0, yi:yi+chunk, xi:xi+chunk]
    data_j = z[0, yj:yj+chunk, xj:xj+chunk]
    data_k = z[0, yk:yk+chunk, xk:xk+chunk]
    data_l = z[0, yl:yl+chunk, xl:xl+chunk]

    data_i = np.array(data_i).astype(np.float32) / 65535.
    data_j = np.array(data_j).astype(np.float32) / 65535.
    data_k = np.array(data_k).astype(np.float32) / 65535.
    data_l = np.array(data_l).astype(np.float32) / 65535.

    data_ij = np.hstack((data_i, data_j))
    data_kl = np.hstack((data_k, data_l))
    data = np.vstack((data_ij, data_kl))

    colormap = cmap.Colormap("tab20", interpolation="nearest")
    colored_image = colormap(data)

    plt.imshow(colored_image, aspect='equal')  
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
