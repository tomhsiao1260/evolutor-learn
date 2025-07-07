import sys
import argparse
from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QPushButton,
        QVBoxLayout,
        QMessageBox,
        )

class Tinter():

    def __init__(self, app, parsed_args):
        window = QWidget()
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