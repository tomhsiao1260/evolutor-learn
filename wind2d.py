import sys
import cv2
import argparse
import pathlib
import math
from scipy import sparse

from st import ST

from PyQt5.QtWidgets import (
        QApplication,
        QGridLayout,
        QLabel,
        QMainWindow,
        QWidget,
        )
from PyQt5.QtCore import (
        QPoint,
        Qt,
        )
from PyQt5.QtGui import (
        QImage,
        QPixmap,
        )

import cv2
import numpy as np
import cmap

class MainWindow(QMainWindow):

    def __init__(self, app, parsed_args):
        super(MainWindow, self).__init__()
        self.app = app
        self.st = None

        grid = QGridLayout()
        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)
        self.viewer = ImageViewer(self)
        grid.addWidget(self.viewer, 0, 0)
        self.viewer.setDefaults()

        no_cache = parsed_args.no_cache
        self.viewer.no_cache = no_cache
        tifname = pathlib.Path(parsed_args.input_tif)
        cache_dir = parsed_args.cache_dir
        decimation = parsed_args.decimation
        self.viewer.decimation = decimation
        umbstr = parsed_args.umbilicus
        window_width = parsed_args.window
        maxrad = parsed_args.maxrad

        if cache_dir is None:
            cache_dir = tifname.parent
        else:
            cache_dir = pathlib.Path(cache_dir)
        cache_file_base = cache_dir / tifname.stem
        self.viewer.cache_dir = cache_dir
        self.viewer.cache_file_base = cache_file_base
        # nrrdname = cache_dir / (tifname.with_suffix(".nrrd")).name

        print("loading tif", tifname)
        # loadTIFF also sets default umbilicus location
        self.viewer.loadTIFF(tifname)
        if umbstr is not None:
            words = umbstr.split(',')
            if len(words) != 2:
                print("Could not parse --umbilicus argument")
            else:
                self.viewer.umb = np.array((float(words[0]),float(words[1])))
        umb = self.viewer.umb
        self.viewer.umb_maxrad = np.sqrt((umb*umb).sum())
        if maxrad is None:
            maxrad = self.viewer.umb_maxrad
        self.viewer.overlay_maxrad = maxrad

        self.st = ST(self.viewer.image)

        part = "_e.nrrd"
        if decimation is not None and decimation > 1:
            part = "_d%d%s"%(decimation, part)
        if window_width is not None:
            part = "_w%d%s"%(window_width, part)
        nrrdname = cache_file_base.with_name(cache_file_base.name + part)
        if no_cache:
            print("computing structural tensors")
            self.st.computeEigens()
        else:
            print("computing/loading structural tensors")
            self.st.loadOrCreateEigens(nrrdname)

        self.viewer.drawAll()

    def keyPressEvent(self, e):
        self.viewer.keyPressEvent(e)

class ImageViewer(QLabel):

    def __init__(self, main_window):
        super(ImageViewer, self).__init__()
        self.setMouseTracking(True)
        self.main_window = main_window
        self.image = None
        self.zoom = 1.
        self.center = (0,0)
        self.bar0 = (0,0)
        self.mouse_start_point = QPoint()
        self.center_start_point = None
        self.is_panning = False
        self.dip_bars_visible = True
        self.warp_dot_size = 3
        self.umb = None
        self.overlays = []
        self.overlay_data = None
        self.overlay_name = ""
        self.decimation = 1
        self.overlay_colormap = "viridis"
        self.overlay_interpolation = "linear"
        self.overlay_maxrad = None
        self.overlay_alpha = None
        self.overlay_scale = None
        self.overlay_defaults = None
        self.src_dots = None
        self.dest_dots = None

    def loadTIFF(self, fname):
        try:
            image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED).astype(np.float64)
            image /= 65535.
        except Exception as e:
            print("Error while loading",fname,e)
            return
        self.image = image
        self.image_mtime = fname.stat().st_mtime
        self.setDefaults()
        self.umb = np.array((image.shape[1]/2, image.shape[0]/2))
        self.drawAll()

    def setDefaults(self):
        if self.image is None:
            return
        ww = self.width()
        wh = self.height()
        # print("ww,wh",ww,wh)
        iw = self.image.shape[1]
        ih = self.image.shape[0]
        self.center = (iw//2, ih//2)
        zw = ww/iw
        zh = wh/ih
        zoom = min(zw, zh)
        self.setZoom(zoom)
        print("center",self.center[0],self.center[1],"zoom",self.zoom)

    def setZoom(self, zoom):
        # TODO: set min, max zoom
        prev = self.zoom
        self.zoom = zoom
        if prev != 0:
            bw,bh = self.bar0
            cw,ch = self.center
            bw -= cw
            bh -= ch
            bw /= zoom/prev
            bh /= zoom/prev
            self.bar0 = (bw+cw, bh+ch)

    # class function
    def rectIntersection(ra, rb):
        (ax1, ay1, ax2, ay2) = ra
        (bx1, by1, bx2, by2) = rb
        # print(ra, rb)
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        if (x1<x2) and (y1<y2):
            r = (x1, y1, x2, y2)
            # print(r)
            return r

    # input: 2D float array, range 0.0 to 1.0
    # output: RGB array, uint8, with colors determined by the
    # colormap and alpha, zoomed in based on the current
    # window size, center, and zoom factor
    def dataToZoomedRGB(self, data, alpha=1., colormap="gray", interpolation="linear", scale=1.):
        if scale is None:
            scale = 1.
        if colormap in self.colormaps:
            colormap = self.colormaps[colormap]
        cm = cmap.Colormap(colormap, interpolation=interpolation)

        iw = data.shape[1]
        ih = data.shape[0]
        z = self.zoom / scale
        # zoomed image width, height:
        ziw = max(int(z*iw), 1)
        zih = max(int(z*ih), 1)
        # viewing window width, height:
        ww = self.width()
        wh = self.height()
        # print("di ww,wh",ww,wh)
        # viewing window half width
        whw = ww//2
        whh = wh//2
        cx,cy = self.center
        cx *= scale
        cy *= scale

        # Pasting zoomed data slice into viewing-area array, taking
        # panning into account.
        # Need to calculate the interesection
        # of the two rectangles: 1) the panned and zoomed slice, and 2) the
        # viewing window, before pasting
        ax1 = int(whw-z*cx)
        ay1 = int(whh-z*cy)
        ax2 = ax1+ziw
        ay2 = ay1+zih
        bx1 = 0
        by1 = 0
        bx2 = ww
        by2 = wh
        ri = ImageViewer.rectIntersection((ax1,ay1,ax2,ay2), (bx1,by1,bx2,by2))
        outrgb = np.zeros((wh,ww,3), dtype=np.uint8)
        if ri is not None:
            (x1,y1,x2,y2) = ri
            # zoomed data slice
            x1s = int((x1-ax1)/z)
            y1s = int((y1-ay1)/z)
            x2s = int((x2-ax1)/z)
            y2s = int((y2-ay1)/z)
            zslc = cv2.resize(data[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            cslc = cm(np.remainder(zslc, 1.))
            outrgb[y1:y2, x1:x2, :] = (255*cslc[:,:,:3]*alpha).astype(np.uint8)
        return outrgb

    def saveCurrentOverlay(self):
        name = self.overlay_name
        no = Overlay(name, self.overlay_data, self.overlay_maxrad, self.overlay_colormap, self.overlay_interpolation, self.overlay_alpha, self.overlay_scale)
        index = Overlay.findIndexByName(self.overlays, name)
        if index < 0:
            self.overlays.append(no)
        else:
            self.overlays[index] = no

    def getNextOverlay(self):
        name = self.overlay_name

        no = Overlay.findNextItem(self.overlays, name)
        self.makeOverlayCurrent(no)

    def makeOverlayCurrent(self, overlay):
        self.saveCurrentOverlay()
        self.overlay_data = overlay.data
        self.overlay_name = overlay.name
        self.overlay_colormap = overlay.colormap
        self.overlay_interpolation = overlay.interpolation
        self.overlay_maxrad = overlay.maxrad
        self.overlay_alpha = overlay.alpha
        self.overlay_scale = overlay.scale

    def setOverlayByName(self, name):
        o = Overlay.findItemByName(self.overlays, name)
        if o is None:
            return
        self.makeOverlayCurrent(o)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_W:
            self.solveWindingOneStep()
            self.drawAll()
        elif e.key() == Qt.Key_A:
            self.getNextOverlay()
            self.drawAll()

    def drawAll(self):
        if self.image is None:
            return
        total_alpha = .8
        main_alpha = total_alpha

        outrgb = self.dataToZoomedRGB(self.image, alpha=main_alpha)
        st = self.main_window.st

        ww = self.width()
        wh = self.height()

        scale = 1.

        if st is not None and self.dip_bars_visible and scale == 1:
            dh = 15
            w0i,h0i = self.wxyToIxy((0,0))
            w0i -= self.bar0[0]
            h0i -= self.bar0[1]
            dhi = 2*dh/self.zoom
            w0i = int(math.floor(w0i/dhi))*dhi
            h0i = int(math.floor(h0i/dhi))*dhi
            w0i += self.bar0[0]
            h0i += self.bar0[1]
            w0,h0 = self.ixyToWxy((w0i,h0i))
            dpw = np.mgrid[h0:wh:2*dh, w0:ww:2*dh].transpose(1,2,0)
            # switch from y,x to x,y coordinates
            dpw = dpw[:,:,::-1]
            # print ("dpw", dpw.shape, dpw.dtype, dpw[0,5])
            dpi = self.wxysToIxys(dpw)
            # interpolators expect y,x ordering
            dpir = dpi[:,:,::-1]
            # print ("dpi", dpi.shape, dpi.dtype, dpi[0,5])
            uvs = st.vector_u_interpolator(dpir)
            vvs = st.vector_v_interpolator(dpir)
            # print("vvs", vvs.shape, vvs.dtype, vvs[0,5])
            # coherence = st.coherence_interpolator(dpir)
            coherence = st.linearity_interpolator(dpir)
            # testing
            # coherence[:] = .5
            # print("coherence", coherence.shape, coherence.dtype, coherence[0,5])
            linelen = 25.

            lvecs = linelen*vvs*coherence[:,:,np.newaxis]
            x0 = dpw
            x1 = dpw+lvecs

            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            # cv2.polylines(outrgb, lines, False, (255,255,0), 1)

            lvecs = linelen*uvs*coherence[:,:,np.newaxis]

            # x1 = dpw+lvecs
            x1 = dpw+.6*lvecs
            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            cv2.polylines(outrgb, lines, False, (0,255,0), 1)

            xm = dpw-.5*lvecs
            xp = dpw+.5*lvecs
            lines = np.concatenate((xm,xp), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            # cv2.polylines(outrgb, lines, False, (0,255,0), 1)

            points = dpw.reshape(-1,1,1,2).astype(np.int32)
            cv2.polylines(outrgb, points, True, (0,255,255), 3)

        if self.umb is not None:
            wumb = self.ixyToWxy(self.umb)
            cv2.circle(outrgb, wumb, 3, (255,0,255), -1)

        bytesperline = 3*outrgb.shape[1]
        # print(outrgb.shape, outrgb.dtype)
        qimg = QImage(outrgb, outrgb.shape[1], outrgb.shape[0],
                      bytesperline, QImage.Format_RGB888)
        # print("created qimg")
        pixmap = QPixmap.fromImage(qimg)
        # print("created pixmap")
        self.setPixmap(pixmap)
        # print("set pixmap")

    colormaps = {
            "gray": "matlab:gray",
            "viridis": "bids:viridis",
            "bwr": "matplotlib:bwr",
            "cool": "matlab:cool",
            "bmr_3c": "chrisluts:bmr_3c",
            "rainbow": "gnuplot:rainbow",
            "spec11": "colorbrewer:Spectral_11",
            "set12": "colorbrewer:Set3_12",
            "tab20": "seaborn:tab20",
            "hsv": "matlab:hsv",
            }

    def ixyToWxy(self, ixy):
        ix,iy = ixy
        cx,cy = self.center
        z = self.zoom
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        wx = int(z*(ix-cx)) + wcx
        wy = int(z*(iy-cy)) + wcy
        return (wx,wy)

    def wxyToIxy(self, wxy):
        wx,wy = wxy
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        dx,dy = wx-wcx, wy-wcy
        cx,cy = self.center
        z = self.zoom
        ix = cx + dx/z
        iy = cy + dy/z
        return (ix, iy)

    def wxysToIxys(self, wxys):
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2

        # dxys = wx-wcx, wy-wcy
        dxys = wxys.copy()
        dxys[...,0] -= wcx
        dxys[...,1] -= wcy
        cx,cy = self.center
        z = self.zoom
        ixys = np.zeros(wxys.shape)
        ixys[...,0] = cx + dxys[...,0]/z
        ixys[...,1] = cy + dxys[...,1]/z
        return ixys

    # This is where the undeform operation takes place.
    # The name of the function doesn't really make sense.
    def solveWindingOneStep(self):
        im = self.image
        if im is None:
            return

        rad = self.createRadiusArray()

        smoothing_weight = .1
        rad0 = self.solveRadius0(rad, smoothing_weight)

        self.alignUVVec(rad0)

        # copy uvec AFTER it has been aligned
        st = self.main_window.st
        uvec = st.vector_u.copy()
        coh = st.coherence.copy()

        self.overlay_data = rad0
        self.overlay_name = "rad0"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

        # hess
        smoothing_weight = .2
        # grad
        # smoothing_weight = .01
        cross_weight = 0.95
        rad1 = self.solveRadius1(rad0, smoothing_weight, cross_weight)

        # find which locations in the image have
        # high coherency and a low radius
        cargs = np.argsort(coh.flatten())
        min_coh = coh.flatten()[cargs[len(cargs)//4]]
        rargs = np.argsort(rad1.flatten())
        max_rad1 = rad1.flatten()[rargs[len(rargs)//4]]

        crb = np.logical_and(coh > min_coh, rad1 < max_rad1)
        crb = np.logical_and(crb, rad > 0)
        rs = rad[crb]
        r1s = rad1[crb]
        # using the locations found above, find the average
        # ratio between r1 (pre-deformation radius) and
        # current radius.
        mr1r = np.median(r1s/rs)
        print("mr1r", mr1r)
        # apply this ratio as a correction to r1
        rad1 /= mr1r

        self.overlay_data = rad1
        self.overlay_name = "rad1"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

        self.setOverlayByName("rad1")

        dot_weight = .001
        smoothing_weight = .4
        theta_weight = .0001

        th0uvec, th0coh = self.synthesizeUVecArray(rad1)

        theta0 = self.solveTheta(rad1, th0uvec, th0coh, dot_weight, smoothing_weight, theta_weight)

        self.overlay_data = theta0
        self.overlay_name = "theta0"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = 3.
        self.saveCurrentOverlay()

        gradx, grady = self.computeGrad(theta0)
        gradx *= rad1
        grady *= rad1
        grad = np.sqrt(gradx*gradx+grady*grady)

        # u cross (rad1 grad theta0)
        gxu = -gradx*th0uvec[:,:,1] + grady*th0uvec[:,:,0]

        cargs = np.argsort(coh.flatten())
        min_coh = coh.flatten()[cargs[len(cargs)//4]]
        rargs = np.argsort(rad1.flatten())
        max_rad1 = rad1.flatten()[rargs[len(rargs)//4]]

        crb = np.logical_and(coh > min_coh, rad1 < max_rad1)

        # gxu (defined above) should average out to 1.0 over the image;
        # find the deviation and apply it as a correction factor to rad1
        mgxu = np.median(gxu[crb])

        print("mgxu", mgxu)

        rad1 /= mgxu

    def createRadiusArray(self):
        umb = self.umb
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        # print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
        rad = np.sqrt(radsq)
        # print("rad", rad.shape)
        return rad

    def createThetaArray(self):
        umb = self.umb
        umb[1] += .5
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        # print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        theta = np.arctan2(iys-umb[1], ixs-umb[0])
        # print("theta", theta.shape, theta.min(), theta.max())
        return theta

    # given a radius array, create u vectors from the
    # normalized gradients of that array.
    def synthesizeUVecArray(self, rad):
        gradx, grady = self.computeGrad(rad)
        uvec = np.stack((gradx, grady), axis=2)
        # print("suvec", uvec.shape)
        luvec = np.sqrt((uvec*uvec).sum(axis=2))
        lnz = luvec != 0
        # print("ll", uvec.shape, luvec.shape, lnz.shape)
        uvec[lnz] /= luvec[lnz][:,np.newaxis]
        coh = np.full(rad.shape, 1.)
        coh[:,-1] = 0
        coh[-1,:] = 0
        return uvec, coh

    def solveRadius0(self, basew, smoothing_weight):
        st = self.main_window.st
        decimation = self.decimation
        print("rad0 smoothing", smoothing_weight)
        print("decimation", decimation)

        vecu = st.vector_u
        coh = st.coherence[:,:,np.newaxis]
        wvecu = coh*vecu
        if decimation > 1:
            wvecu = wvecu[::decimation, ::decimation, :]
            basew = basew.copy()[::decimation, ::decimation]
        shape = wvecu.shape[:2]
        sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=True)
        sparse_grad = ImageViewer.sparseGrad(shape)
        sparse_umb = ImageViewer.sparseUmbilical(shape, np.array(self.umb)/decimation)
        sparse_u_cross_grad = sparse.vstack((sparse_uxg, smoothing_weight*sparse_grad, sparse_umb))

        A = sparse_u_cross_grad
        # print("A", A.shape, A.dtype)

        b = -sparse_u_cross_grad @ basew.flatten()
        b[basew.size:] = 0.
        x = self.solveAxEqb(A, b)
        out = x.reshape(basew.shape)
        out += basew
        # print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (vecu.shape[1], vecu.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out

    def solveRadius1(self, rad0, smoothing_weight, cross_weight):
        st = self.main_window.st
        print("rad1 smoothing", smoothing_weight, "cross_weight", cross_weight)
        decimation = self.decimation
        # print("decimation", decimation)

        icw = 1.-cross_weight

        uvec = st.vector_u
        coh = st.coherence.copy()

        # TODO: for testing
        # mask = self.createMask()
        ## coh = coh.copy()*mask
        # coh *= mask

        coh = coh[:,:,np.newaxis]

        wuvec = coh*uvec
        if decimation > 1:
            wuvec = wuvec[::decimation, ::decimation, :]
            coh = coh.copy()[::decimation, ::decimation, :]
            rad0 = rad0.copy()[::decimation, ::decimation] / decimation
        shape = wuvec.shape[:2]
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=False)
        sparse_grad = ImageViewer.sparseGrad(shape)
        sgx, sgy = ImageViewer.sparseGrad(shape, interleave=False)
        hxx = sgx.transpose() @ sgx
        hyy = sgy.transpose() @ sgy
        hxy = sgx @ sgy
        # print("sgx", sgx.shape, "hxx", hxx.shape, "hxy", hxy.shape)

        # print("grad", sparse_grad.shape, "hess", sparse_hess.shape)
        sparse_umb = ImageViewer.sparseUmbilical(shape, np.array(self.umb)/decimation)
        # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*sparse_grad, sparse_umb))
        # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, smoothing_weight*hxy, sparse_umb))
        sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, sparse_umb))

        A = sparse_all
        # print("A", A.shape, A.dtype)

        b = np.zeros((A.shape[0]), dtype=np.float64)
        # NOTE multiplication by decimation factor
        b[:rad0.size] = 1.*coh.flatten()*decimation*icw
        x = self.solveAxEqb(A, b)
        out = x.reshape(rad0.shape)
        # print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out

    def solveTheta(self, rad, uvec, coh, dot_weight, smoothing_weight, theta_weight):
        print("theta dot_weight", dot_weight, "smoothing", smoothing_weight, "theta_weight", theta_weight)
        st = self.main_window.st
        decimation = self.decimation
        # print("decimation", decimation)

        theta = self.createThetaArray()
        oldshape = rad.shape
        coh = coh[:,:,np.newaxis]
        weight = coh.copy()
        # TODO: for testing only!
        # mask = self.createMask()
        # coh *= mask
        # weight = coh*coh*coh
        # weight[:,:] = 1.
        wuvec = weight*uvec
        rwuvec = rad[:,:,np.newaxis]*wuvec
        if decimation > 1:
            wuvec = wuvec[::decimation, ::decimation, :]
            theta = theta[::decimation, ::decimation]
            weight = weight[::decimation, ::decimation, :]
            # Note that rad is divided by decimation
            rad = rad.copy()[::decimation, ::decimation] / decimation
            # recompute rwuvec to account for change in rad
            rwuvec = rad[:,:,np.newaxis]*wuvec
        shape = theta.shape
        sparse_grad = ImageViewer.sparseGrad(shape, rad)
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=False)
        sparse_theta = ImageViewer.sparseDiagonal(shape)
        sparse_all = sparse.vstack((sparse_u_cross_g, dot_weight*sparse_u_dot_g, smoothing_weight*sparse_grad, theta_weight*sparse_theta))
        # print("sparse_all", sparse_all.shape)

        umb = np.array(self.umb)
        decimated_umb = umb/decimation
        iumb = decimated_umb.astype(np.int32)
        # bc: branch cut
        bc_rad = rad[iumb[1], :iumb[0]]
        bc_rwuvec = rwuvec[iumb[1], :iumb[0]]
        bc_dot = 2*np.pi*bc_rwuvec[:,1]
        bc_grad = 2*np.pi*bc_rad
        bc_cross = 2*np.pi*bc_rwuvec[:,0]
        bc_f0 = shape[1]*iumb[1]
        bc_f1 = bc_f0 + iumb[0]

        b_dot = np.zeros((sparse_u_dot_g.shape[0]), dtype=np.float64)
        b_dot[bc_f0:bc_f1] += bc_dot.flatten()
        b_cross = weight.flatten()
        b_cross[bc_f0:bc_f1] += bc_cross.flatten()
        b_grad = np.zeros((sparse_grad.shape[0]), dtype=np.float64)
        b_grad[2*bc_f0+1:2*bc_f1+1:2] += bc_grad.flatten()
        b_theta = theta.flatten()
        b_all = np.concatenate((b_cross, dot_weight*b_dot, smoothing_weight*b_grad, theta_weight*b_theta))
        # print("b_all", b_all.shape)

        x = self.solveAxEqb(sparse_all, b_all)
        out = x.reshape(shape)
        # print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            outl = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
            outn = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_NEAREST)
        # return outl,outn
        return outl

    def computeGrad(self, arr):
        decimation = self.decimation
        oldshape = arr.shape
        if decimation > 1:
            arr = arr.copy()[::decimation, ::decimation]
        shape = arr.shape
        sparse_grad = ImageViewer.sparseGrad(shape)

        # NOTE division by decimation
        grad_flat = (sparse_grad @ arr.flatten()) / decimation
        grad = grad_flat.reshape(shape[0], shape[1], 2)
        gradx = grad[:,:,0]
        grady = grad[:,:,1]
        if decimation > 1:
            gradx = cv2.resize(gradx, (oldshape[1], oldshape[0]), interpolation=cv2.INTER_LINEAR)
            grady = cv2.resize(grady, (oldshape[1], oldshape[0]), interpolation=cv2.INTER_LINEAR)
        return gradx, grady

    # set is_cross True if op is cross product, False if
    # op is dot product
    # Creates a sparse matrix that represents the operator
    # vec2d cross grad  or  vec2d dot grad
    # depending on the is_cross flag
    @staticmethod
    def sparseVecOpGrad(vec2d, is_cross):
        # full number of rows, columns of image;
        # it is assumed that the image and vec2d
        # are the same size, except each vec2d element
        # has 2 components.
        nrf, ncf = vec2d.shape[:2]
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, vec2d.shape[:2])
        # No immediate effect, since n1df is a view of n1df_flat
        n1df_flat = None
        # n1d is like n1df but shrunk by 1 in row and column directions
        n1d = n1df[:nr, :nc]
        # No immediate effect, since n1d is a view of n1df
        n1df = None
        # flat array (size nrf-1 times ncf-1) where each element
        # contains a position in the original nrf by ncf array. 
        n1d_flat = n1d.flatten()
        # No immediate effect, since n1d_flat is a view of n1d
        n1d = None
        # diag3 is the diagonal matrix of n1d_flat, in 3-column sparse format.
        # float32 is not precise enough for carrying indices of large
        # flat matrices, so use default (float64)
        diag3 = np.stack((n1d_flat, n1d_flat, np.zeros(n1d_flat.shape)), axis=1)
        # print("diag3", diag3.shape, diag3.dtype)
        # clean up memory
        n1d_flat = None

        vec2d_flat = vec2d[:nr, :nc].reshape(-1, 2)
        # print("vec2d_flat", vec2d_flat.shape)

        dx0 = diag3.copy()

        dx1 = diag3.copy()
        dx1[:,1] += 1
        if is_cross:
            dx0[:,2] = vec2d_flat[:,1]
            dx1[:,2] = -vec2d_flat[:,1]
        else:
            dx0[:,2] = -vec2d_flat[:,0]
            dx1[:,2] = vec2d_flat[:,0]

        ddx = np.concatenate((dx0, dx1), axis=0)
        # print("ddx", ddx.shape, ddx.dtype)

        # clean up memory
        dx0 = None
        dx1 = None

        dy0 = diag3.copy()

        dy1 = diag3.copy()
        dy1[:,1] += ncf
        if is_cross:
            dy0[:,2] = -vec2d_flat[:,0]
            dy1[:,2] = vec2d_flat[:,0]
            pass
        else:
            dy0[:,2] = -vec2d_flat[:,1]
            dy1[:,2] = vec2d_flat[:,1]

        ddy = np.concatenate((dy0, dy1), axis=0)
        # print("ddy", ddy.shape, ddy.dtype)

        # clean up memory
        dy0 = None
        dy1 = None

        # print("ddx,ddy", ddx.max(axis=0), ddy.max(axis=0))

        uxg = np.concatenate((ddx, ddy), axis=0)
        # print("uxg", uxg.shape, uxg.dtype, uxg[:,0].max(), uxg[:,1].max())
        ddx = None
        ddy = None
        sparse_uxg = sparse.coo_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(nrf*ncf, nrf*ncf))
        # sparse_uxg = sparse.csc_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(nrf*ncf, nrf*ncf))
        # print("sparse_uxg", sparse_uxg.shape, sparse_uxg.dtype)
        return sparse_uxg

    # create a sparse matrix that represents the 2D grad operator.
    # if interleave is true, the output is a single sparse matrix
    # with interleaved x and y components of the grad.
    # if interleave is false, separate sparse matrices are created
    # for the x and y components of the grad
    @staticmethod
    def sparseGrad(shape, multiplier=None, interleave=True):
        # full number of rows, columns of image
        nrf, ncf = shape
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, shape)
        n1df_flat = None

        # n1dfr is full in the row direction, but shrunk by 1 in column dir
        n1dfr = n1df[:, :nc]
        # n1dfc is full in the column direction, but shrunk by 1 in row dir
        n1dfc = n1df[:nr, :]
        n1df = None
        n1dfr_flat = n1dfr.flatten()
        n1dfr = None
        n1dfc_flat = n1dfc.flatten()
        n1dfc = None

        mfr = None
        mfc = None
        if multiplier is not None:
            mfr = multiplier.flatten()[n1dfr_flat]
            mfc = multiplier.flatten()[n1dfc_flat]

        # float32 is not precise enough for carrying indices of large
        # flat matrices, so use default (float64)
        diag3fr = np.stack((n1dfr_flat, n1dfr_flat, np.zeros(n1dfr_flat.shape)), axis=1)
        n1dfr_flat = None
        diag3fc = np.stack((n1dfc_flat, n1dfc_flat, np.zeros(n1dfc_flat.shape)), axis=1)
        n1dfc_flat = None

        dx0g = diag3fr.copy()
        if mfr is not None:
            dx0g[:,2] = -mfr
        else:
            dx0g[:,2] = -1.

        dx1g = diag3fr.copy()
        dx1g[:,1] += 1
        if mfr is not None:
            dx1g[:,2] = mfr
        else:
            dx1g[:,2] = 1.

        ddxg = np.concatenate((dx0g, dx1g), axis=0)
        # print("ddx", ddx.shape, ddx.dtype)

        # clean up memory
        diag3fr = None
        dx0g = None
        dx1g = None

        dy0g = diag3fc.copy()
        if mfc is not None:
            dy0g[:,2] = -mfc
        else:
            dy0g[:,2] = -1.

        dy1g = diag3fc.copy()
        dy1g[:,1] += ncf
        if mfc is not None:
            dy1g[:,2] = mfc
        else:
            dy1g[:,2] = 1.

        ddyg = np.concatenate((dy0g, dy1g), axis=0)

        # clean up memory
        diag3fc = None
        dy0g = None
        dy1g = None

        if interleave:
            ddxg[:,0] *= 2
            ddyg[:,0] *= 2
            ddyg[:,0] += 1

            grad = np.concatenate((ddxg, ddyg), axis=0)
            # print("grad", grad.shape, grad.min(axis=0), grad.max(axis=0), grad.dtype)
            sparse_grad = sparse.coo_array((grad[:,2], (grad[:,0], grad[:,1])), shape=(2*nrf*ncf, nrf*ncf))
            return sparse_grad
        else:
            sparse_grad_x = sparse.coo_array((ddxg[:,2], (ddxg[:,0], ddxg[:,1])), shape=(nrf*ncf, nrf*ncf))
            sparse_grad_y = sparse.coo_array((ddyg[:,2], (ddyg[:,0], ddyg[:,1])), shape=(nrf*ncf, nrf*ncf))
            return sparse_grad_x, sparse_grad_y

    @staticmethod
    def sparseUmbilical(shape, umb):
        nrf, ncf = shape
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, shape)
        umbpt = n1df[int(umb[1]), int(umb[0])]
        umbzero = np.array([[0, umbpt, 1.]])
        sparse_umb = sparse.coo_array((umbzero[:,2], (umbzero[:,0], umbzero[:,1])), shape=(nrf*ncf, nrf*ncf))
        return sparse_umb

    @staticmethod
    def sparseDiagonal(shape):
        nrf, ncf = shape
        ix = np.arange(nrf*ncf)
        ones = np.full((nrf*ncf), 1.)
        sparse_diag = sparse.coo_array((ones, (ix, ix)), shape=(nrf*ncf, nrf*ncf))
        return sparse_diag

    @staticmethod
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

    # given an array filled with radius values, align
    # the structure tensor u vector with the gradient of
    # the radius
    def alignUVVec(self, rad0):
        st = self.main_window.st
        uvec = st.vector_u
        shape = uvec.shape[:2]
        sparse_grad = self.sparseGrad(shape)
        delr_flat = sparse_grad @ rad0.flatten()
        delr = delr_flat.reshape(uvec.shape)
        # print("delr", delr[iy,ix])
        dot = (uvec*delr).sum(axis=2)
        # print("dot", dot[iy,ix])
        print("dots", (dot<0).sum())
        print("not dots", (dot>=0).sum())
        st.vector_u[dot<0] *= -1
        st.vector_v[dot<0] *= -1

        # Replace vector interpolator by simple interpolator
        st.vector_u_interpolator = ST.createInterpolator(st.vector_u)
        st.vector_v_interpolator = ST.createInterpolator(st.vector_v)

class Overlay():
    def __init__(self, name, data, maxrad, colormap="viridis", interpolation="linear", alpha=None, scale=None):
        self.name = name
        self.data = data
        self.colormap = colormap
        self.interpolation = interpolation
        self.maxrad = maxrad
        self.alpha = alpha
        self.scale = scale

    @staticmethod
    def findIndexByName(overlays, name):
        for i,item in enumerate(overlays):
            if item.name == name:
                return i
        return -1

    @staticmethod
    def findNextItem(overlays, cur_name):
        index = Overlay.findIndexByName(overlays, cur_name)
        if index < 0:
            return overlays[0]
        return overlays[(index+1)%len(overlays)]

    @staticmethod
    def findItemByName(overlays, name):
        index = Overlay.findIndexByName(overlays, name)
        if index < 0:
            return None
        return overlays[index]

class Tinter():

    def __init__(self, app, parsed_args):
        window = MainWindow(app, parsed_args)
        self.app = app
        self.window = window
        window.show()

# From https://stackoverflow.com/questions/11713006/elegant-command-line-argument-parsing-for-pyqt

def process_cl_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Test determining winding numbers using structural tensors")
    parser.add_argument("input_tif",
                        help="input tiff slice")
    parser.add_argument("--cache_dir",
                        default=None,
                        help="directory where the cache of the structural tensor data is or will be stored; if not given, directory of input tiff slice is used")
    parser.add_argument("--no_cache",
                        action="store_true",
                        help="Don't use cached structural tensors")
    parser.add_argument("--diagnostics",
                        action="store_true",
                        help="Create diagnostic overlays")
    parser.add_argument("--umbilicus",
                        default=None,
                        help="umbilicus location in x,y, for example 3960,2280")
    parser.add_argument("--window",
                        type=int,
                        default=None,
                        help="size of window centered around umbilicus")
    parser.add_argument("--colormap",
                        default="viridis",
                        help="colormap")
    parser.add_argument("--interpolation",
                        default="linear",
                        help="interpolation, either linear or nearest")
    parser.add_argument("--maxrad",
                        type=float,
                        default=None,
                        help="max expected radius, in pixels (if not given, will be calculated from umbilicus position)")
    parser.add_argument("--decimation",
                        type=int,
                        default=8,
                        help="decimation factor (default is no decimation)")
    parser.add_argument("--warp_decimation",
                        type=int,
                        default=32,
                        help="decimation factor for warping")

    # I decided not to use parse_known_args because
    # I prefer to get an error message if an argument
    # is unrecognized
    # parsed_args, unparsed_args = parser.parse_known_args()
    # return parsed_args, unparsed_args
    parsed_args = parser.parse_args()
    return parsed_args

# python wind2d.py ./evol1/circle.tif --umbilicus 549,463
if __name__ == '__main__':
    parsed_args = process_cl_args()
    qt_args = sys.argv[:1] 
    app = QApplication(qt_args)

    tinter = Tinter(app, parsed_args)
    sys.exit(app.exec())
