"""LEPTON READER APPLICATION"""


#! IMPORTS


import sys

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw

from leptonreader import CameraWidget


#! MAIN


if __name__ == "__main__":

    # highdpi scaling
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

    # app generation
    app = qtw.QApplication(sys.argv)
    try:
        camera = CameraWidget()
    except Exception as exc:
        raise RuntimeError from exc
    camera.show()
    sys.exit(app.exec_())
