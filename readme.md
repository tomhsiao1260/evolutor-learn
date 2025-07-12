# Introduction

Learn from [evolutor](https://github.com/KhartesViewer/evolutor)


# synth2d.py

Used to generate test TIFF data.

# wind2d.py

Able to convert a specified TIFF image into a circular shape (requires center point umbilicus).

### MainWindow

- Generation of the main interface, utilizing QWidget and QGridLayout from PyQt5.
- Initialization of the ImageViewer class, loading the original TIFF image data.
- Initialization of the structure tensor class, calculating the related eigenvalues of the image.

### ImageViewer

- loadTIFF: Logic details for loading TIFF images.  
- setDefaults: Set default values for circular center and zoom.  
- setZoom: Update the zoom value.  
- rectIntersection: Calculate the visible area in the view.  
- dataToZoomedRGB: Generate the color image to render in the view area after color mapping.  
- drawAll: Core rendering logic. Render the image on the PyQt5 application via QImage.  
- ixyToWxy: Convert absolute coordinates to window display coordinates.
- solveWindingOneStep: Core undeform operation logic.
- createRadiusArray: Initial radius array (distance to umbilicus).
- createThetaArray: Initial theta array (angle to umbilicus).
- solveRadius0: Caculate pre-deformation radius array r0.
- solveRadius1: Caculate pre-deformation radius array r1.
- solveTheta: Caculate pre-deformation theta array.
- sparseVecOpGrad: sparse matrix that represents the operator vec2d cross grad or vec2d dot grad.
- sparseGrad: sparse matrix that represents the 2D grad operator.
- sparseUmbilical: sparse matrix that represents the umbilicus location.
- sparseDiagonal: sparse matrix that represents the diagonal array.
- solveAxEqb: solve Ax = b (least square method find x).
- alignUVVec: align u vector with the gradient of the radius.
- synthesizeUVecArray: create u vectors and coherence from a given radius array.
- computeGrad: Caculate gradient x, y for a given array.
- xformXY: given radius and theta, compute x and y.
- warpImage: transform the image for a given src and dest via FastPiecewiseAffineTransform.

### process_cl_args

some input parameters

- umbilicus: Coordinates of the circular center point.  
- decimation: Default value is 8, representing sampling density.

# st.py

Core logic of the structure tensor.

### ST

- saveImage: Save the image file.  
- computeEigens: Calculate the related eigenvalues of the image, including `lambda_u`, `lambda_v`, `vector_u`, `vector_v`, and `grad`.  
- saveEigens: Combine related eigenvalues and save as an NRRD file.  
- loadEigens: Load eigenvalue data from an NRRD file (e.g. ``.._e.nrrd`).
- loadOrCreateEigens: Logic handling (load, save, compute).

### Overlay

A class to store different structure tensor during the calculation and display them via drawAll function.

# Details

More details for specific functions.

### computeEigens()

- `lambda_u` (1-dim) is the max eigenvalues of eigenvectors `vector_u` (2-dim)
- `lambda_v` (1-dim) is the min eigenvalues of eigenvectors `vector_v` (2-dim)
- `grad` is the gradient vector with Gaussian filter (2-dim)

- `isotropy` metric (1-dim) is between 0 (edge) and 1 (random). Formula: `lv / lu`
- `linearity` metric (1-dim) is between 0 (random) and 1 (edge). Formula: `(lu - lv) / lu`
- `coherence` metric (1-dim) is between 0 (random) and 1 (edge). Formula: `coherence = ((lu - lv) / (lu + lv)) ** 2`

### drawAll()

A portion of the code will draw the calculated eigenvectors onto the original image, following a general method as outlined below:

First, generate the coordinate arrays for the sampling points, `dpw` and `dpir`, then sample `vector_u`, `vector_v`, and `coherence` using the interpolator.
```python
uvs = st.vector_u_interpolator(dpir)
vvs = st.vector_v_interpolator(dpir)
coherence = st.linearity_interpolator(dpir)
```

Next, calculate the length of the normal vectors:
```python
lvecs = 25. * uvs * coherence[:, :, np.newaxis]
```

And calculate the length of the tangent vectors:
```python
lvecs = linelen * vvs * coherence[:, :, np.newaxis]
```

### solveWindingOneStep()

Core undeform operation logic.

`u` (from original image) -> `r` (from umbilicus) -> `r0` (from `u`, `r`) -> `u` (align `u` to `r0`) -> `r1` (from `r0` and `u`) -> `th0` -> `r1` (adjust) -> `th1` (from `r1`)

### solveRadius0()

Caculate pre-deformation radius array r0 (more description in original repo).

```markdown
# r: initial radius
# r0: pre-deformation radius
r0 = r + r0'

# constrain we need (r should align u)
u cross (grad r0) = 0
# thus
u cross (grad r0') = -u cross (grad r0)
```


Then, solve Ax=b
```python
A = sparse.vstack((sparse_uxg, smoothing_weight*sparse_grad, sparse_umb))

b = -sparse_u_cross_grad @ basew.flatten()
b[basew.size:] = 0.

x = self.solveAxEqb(A, b)

# Then, get the final result via r0 = r + r0'
```

The matrix `A` and flatten vector `b` are constructed by stacking the results from the `sparseVecOpGrad` and `sparseGrad` methods. The `sparseUmbilical` is to ensure that the `r0` at the umbilicus is set to 0.

### solveRadius1()

Generate `r1` which use `u` to refine the value of the pre-deformation radius `r0`.

```markdown
# constrain we need (simple circular geometry)
u dot (grad r1) = 1
u cross (grad r1) = 0

# more contrain (smooth in x, y direction, radius 0 at umbilicus)
```

Then, solve Ax=b
```python
A = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, sparse_umb))

b = np.zeros((A.shape[0]), dtype=np.float64)
b[:rad0.size] = 1.*coh.flatten()*decimation*icw

x = self.solveAxEqb(A, b)
```

Once r1 is computed, an adjustment factor is calculated. Averaged over the entire image, r1 / r should equal 1. The actual ratio is calculated (in the center part of the image), and r1 is multiplied by whatever factor is needed to bring the average r1 / r ratio to 1.

### solveTheta()

Caculate pre-deformation theta array.

```markdown
# t is tangent vector, n is normal vector

# constrain we need
(r1 grad th0) dot t = 1
# change its form a bit
(r1 grad th0) cross n = 1

# more contrain for smoothness
```

Then, solve Ax=b
```python
# sparse_u_cross_g, b_cross -> (r1 grad th0) cross n = 1
A = sparse.vstack((sparse_u_cross_g, dot_weight*sparse_u_dot_g, smoothing_weight*sparse_grad, theta_weight*sparse_theta))

b_all = np.concatenate((b_cross, dot_weight*b_dot, smoothing_weight*b_grad, theta_weight*b_theta))

x = self.solveAxEqb(A, b)
```

