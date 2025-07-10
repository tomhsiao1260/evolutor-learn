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
- solveRadius0: Caculate pre-deformation radius array r0.
- sparseVecOpGrad: sparse matrix that represents the operator vec2d cross grad or vec2d dot grad.
- sparseGrad: sparse matrix that represents the 2D grad operator.
- sparseUmbilical: sparse matrix that represents the umbilicus location.
- solveAxEqb: solve Ax = b (least square method find x).
- alignUVVec: align u vector with the gradient of the radius.

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

`vector_u` (from original image) -> `r` (from umbilicus) -> `r0` (from `vector_u`, `r`) -> `vector_u` (align `vector_u` to `r0`)

### solveRadius0()

Caculate pre-deformation radius array r0 (more description in original repo).

```markdown
# r: initial radius
# r0: pre-deformation radius
r0 = r + r0\'

# constrain we need (r should align u)
u cross (grad r0) = 0
# thus
u cross (grad r0\') = -u cross (grad r0)
# solve Ax=b
A: u cross grad operator
b: -u cross (grad r0)
x: r0\'

# then get the result r0
r0 = r + r0\'
```

The matrix `A` and flatten vector `b` are constructed by stacking the results from the `sparseVecOpGrad` and `sparseGrad` methods. The `sparseUmbilical` is to ensure that the `r0` at the umbilicus is set to 0.


