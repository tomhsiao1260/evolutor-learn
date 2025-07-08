## Introduction

Learn from [evolutor](https://github.com/KhartesViewer/evolutor)


## synth2d.py

Used to generate test TIFF data.

## wind2d.py

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

### process_cl_args

some input parameters

- umbilicus: Coordinates of the circular center point.  
- decimation: Default value is 8, representing sampling density.

## st.py

Core logic of the structure tensor.

### ST

- saveImage: Save the image file.  
- computeEigens: Calculate the related eigenvalues of the image, including `lambda_u`, `lambda_v`, `vector_u`, `vector_v`, and `grad`.  
- saveEigens: Combine related eigenvalues and save as an NRRD file.  
- loadEigens: Load eigenvalue data from an NRRD file (e.g. ``.._e.nrrd`).
- loadOrCreateEigens: Logic handling (load, save, compute).


### Eigenvalues Info

- `lambda_u` (1-dim) is the max eigenvalues of eigenvectors `vector_u` (2-dim)
- `lambda_v` (1-dim) is the min eigenvalues of eigenvectors `vector_v` (2-dim)
- `grad` is the gradient vector with Gaussian filter (2-dim)

- `isotropy` metric (1-dim) is between 0 (edge) and 1 (random). Formula: `lv / lu`
- `linearity` metric (1-dim) is between 0 (random) and 1 (edge). Formula: `(lu - lv) / lu`
- `coherence` metric (1-dim) is between 0 (random) and 1 (edge). Formula: `coherence = ((lu - lv) / (lu + lv)) ** 2`

