import cv2
import numpy as np

class ST(object):

    # assumes image is a floating-point numpy array
    def __init__(self, image):
        self.image = image
        self.lambda_u = None
        self.lambda_v = None
        self.vector_u = None
        self.vector_v = None
        self.isotropy = None
        self.linearity = None
        self.coherence = None
        # self.vector_u_interpolator_ = None
        # self.vector_v_interpolator_ = None
        self.lambda_u_interpolator_ = None
        self.lambda_v_interpolator_ = None
        self.vector_u_interpolator_ = None
        self.vector_v_interpolator_ = None
        self.grad_interpolator_ = None
        self.isotropy_interpolator_ = None
        self.linearity_interpolator_ = None
        self.coherence_interpolator_ = None

    def saveImage(self, fname):
        timage = (self.image*65535).astype(np.uint16)
        cv2.imwrite(str(fname), timage)