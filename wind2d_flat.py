import sys
import cv2
import argparse
import pathlib
import math

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
        self.viewer.loadTIFF(tifname)

        self.viewer.overlay_maxrad = maxrad
        self.viewer.warp_decimation = parsed_args.warp_decimation

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

# python wind2d_flat.py ./evol3/cell_yxz_006_008_004.tif
if __name__ == '__main__':
    parsed_args = process_cl_args()
    qt_args = sys.argv[:1]
    app = QApplication(qt_args)

    tinter = Tinter(app, parsed_args)
    sys.exit(app.exec())



