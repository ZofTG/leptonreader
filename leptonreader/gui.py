"""LEPTONREADER GUI MODULE"""


#! IMPORTS


import os
import time
from typing import Tuple

import cv2
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import qimage2ndarray

from .core import Camera
from .assets import *

__all__ = ["CameraWidget"]


#! CONSTANTS


FONT = qtg.QFont("Arial", 12)


#! FUNCTIONS


def _QIcon(path, size=40):
    """
    return a QIcon with the required size from file.

    Parameters
    ----------
    path: str
        the file to be used as icon.

    size: int
        the dimension of the icon.

    Returns
    -------
    icon: QIcon
        the icon.
    """
    # check the entries
    assert os.path.exists(path), "path must be a valid file."
    assert isinstance(size, int), "size must be an int."

    # get the icon
    qimage = qtg.QImage()
    qimage.loadFromData(qtc.QByteArray.fromBase64(path.encode("ascii")))
    pixmap = qtg.QPixmap(qimage)
    scaled = pixmap.scaledToHeight(
        size,
        mode=qtc.Qt.TransformationMode.SmoothTransformation,
    )
    return qtg.QIcon(scaled)


#! CLASSES


class _RecordingWidget(qtw.QWidget):
    """
    Initialize a PySide2 widget capable showing a checkable button for
    recording things and showing the recording time.
    """

    button = None
    label = None
    start_time = None
    timer = None
    label_format = "{:02d}:{:02d}:{:02d}"
    started = qtc.pyqtSignal()
    stopped = qtc.pyqtSignal()
    _size = 50

    def __init__(self):
        super().__init__()

        # generate the output layout
        layout = qtw.QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        # recording time
        self.label = qtw.QLabel(self.label_format.format(0, 0, 0, 0))
        self.label.setFont(FONT)

        # rec button
        rec_icon = _QIcon(REC, self._size)
        self.button = qtw.QPushButton()
        self.button.setFlat(True)
        self.button.setCheckable(True)
        self.button.setContentsMargins(0, 0, 0, 0)
        self.button.setIcon(rec_icon)
        self.button.setFixedHeight(self._size)
        self.button.setFixedWidth(self._size)
        self.button.clicked.connect(self.clicked)

        # generate the output
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # start the timer runnable
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update_time)

    def update_time(self):
        """
        timer function
        """
        if self.start_time is not None:
            now = time.time()
            delta = now - self.start_time
            h = int(delta // 3600)
            m = int((delta - h * 3600) // 60)
            s = int((delta - h * 3600 - m * 60) // 1)
            self.label.setText(self.label_format.format(h, m, s))
        else:
            self.label.setText(self.label_format.format(0, 0, 0))

    def clicked(self):
        """
        function handling the clicking of the recording button.
        """
        if self.button.isChecked():  # the recording starts
            self.start_time = time.time()
            self.timer.start(1000)
            self.started.emit()

        else:  # the recording stops
            self.start_time = None
            self.stopped.emit()


class _HoverWidget(qtw.QWidget):
    """
    defines a hover pane to be displayed over a matplotlib figure.
    """

    # class variable
    labels = {}
    formatters = {}
    artists = {}
    layout = None

    def __init__(self):
        super().__init__()
        self.layout = qtw.QGridLayout()
        self.layout.setHorizontalSpacing(20)
        self.layout.setVerticalSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        flags = qtc.Qt.FramelessWindowHint | qtc.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def add_label(
        self,
        name: str,
        unit: str,
        digits: int,
    ):
        """
        add a new label to the hover.

        Parameters
        ----------
        name: str
            the name of the axis

        unit: str
            the unit of measurement to be displayed.

        digits: int
            the number of digits to be displayed by the hover.
        """
        # check the entries
        assert isinstance(name, str), "name must be a str."
        assert isinstance(unit, str), "unit must be a str."
        assert isinstance(digits, int), "digits must be an int."
        assert digits >= 0, "digits must be >= 0."

        # add the new label
        n = len(self.labels)
        name_label = qtw.QLabel(name)
        name_label.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignRight)
        name_label.setFont(FONT)
        self.layout.addWidget(name_label, n, 0)
        self.labels[name] = qtw.QLabel("")
        self.labels[name].setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignLeft)
        self.labels[name].setFont(FONT)
        self.layout.addWidget(self.labels[name], n, 1)
        self.setLayout(self.layout)
        self.formatters[name] = lambda x: self.unit_formatter(x, unit, digits)

    def update(self, **labels):
        """
        update the hover parameters.

        Parameters
        ----------
        labels: any
            keyworded values to be updated
        """
        for label, value in labels.items():
            self.labels[label].setText(self.formatters[label](value))

    def unit_formatter(
        self,
        x: Tuple[int, float],
        unit: str = "",
        digits: int = 1,
    ):
        """
        return the letter linked to the order of magnitude of x.

        Parameters
        ----------
        x: int, float
            the value to be visualized

        unit: str
            the unit of measurement

        digits: int
            the number of digits required when displaying x.
        """

        # check the entries
        assert isinstance(x, (int, float)), "x must be a float or int."
        assert isinstance(unit, str), "unit must be a str."
        assert isinstance(digits, int), "digits must be an int."
        assert digits >= 0, "digits must be >= 0."

        if unit != "":
            # get the magnitude
            mag = np.floor(np.log10(abs(x))) * np.sign(x)

            # scale the magnitude
            unit_letters = ["p", "n", "μ", "m", "", "k", "M", "G", "T"]
            unit_magnitudes = np.arange(-12, 10, 3)
            index = int((min(12, max(-12, mag)) + 12) // 3)

            # get the value
            v = x / (10.0 ** unit_magnitudes[index])
            letter = unit_letters[index]
        else:
            v = x
            letter = ""

        # return the value formatted
        return ("{:0." + str(digits) + "f} {}{}").format(v, letter, unit)


class _ImageWidget(qtw.QLabel):
    """
    Generate a QWidget incorporating a matplotlib Figure.
    Animated artists can be provided to ensure optimal performances.

    Parameters
    ----------
    hover_offset_x: float
        the percentage of the screen width that offsets the hover with respect
        to the position of the mouse.

    hover_offset_y: float
        the percentage of the screen height that offsets the hover with respect
        to the position of the mouse.

    colormap: cv2.COLORMAP
        the colormap to be applied
    """

    # class variables
    hover = None
    hover_offset_x_perc = None
    hover_offset_y_perc = None
    colormap = None
    data = None

    def __init__(
        self,
        hover_offset_x=0.02,
        hover_offset_y=0.02,
        colormap=cv2.COLORMAP_JET,
    ):
        super().__init__()
        self.hover = _HoverWidget()
        self.hover.add_label("x", "", 0)
        self.hover.add_label("y", "", 0)
        self.hover.add_label("temperature", "°C", 1)
        self.hover.setVisible(False)
        self.hover_offset_x_perc = hover_offset_x
        self.hover_offset_y_perc = hover_offset_y
        self.colormap = colormap
        self.setMouseTracking(True)

    def enterEvent(self, event=None):
        """
        override enterEvent.
        """
        self.hover.setVisible(True)
        self.update_hover(event)

    def mouseMoveEvent(self, event=None):
        """
        override moveEvent.
        """
        self.update_hover(event)

    def leaveEvent(self, event=None):
        """
        override leaveEvent.
        """
        self.hover.setVisible(False)

    def _adjust_view(self):
        """
        private method used to resize the widget
        """
        if self.data is not None:
            if self.pixmap() is not None:
                self.pixmap().scaled(*self.data.shape)

            # get the target width
            screen_size = qtw.QDesktopWidget().screenGeometry(-1).size()
            max_w = screen_size.width()
            min_w = self.parent().minimumSizeHint().width()
            new_w = self.sizeHint().width()
            w = min(max_w, max(min_w, new_w))

            # get the target height
            ratio = self.data.shape[0] / self.data.shape[1]
            new_h = int(round(w / ratio))
            max_h = screen_size.height()
            max_h -= self.parent().size().height() - self.size().height()
            h = min(new_h, max_h)

            # convert the data into image
            img = self.data - np.min(self.data)
            img /= np.max(self.data) - np.min(self.data)
            img = np.expand_dims(img * 255, 2).astype(np.uint8)
            img = np.concatenate([img, img, img], axis=2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.applyColorMap(img, self.colormap)

            # adjust the image shape
            if new_h <= max_h:
                img = qimage2ndarray.array2qimage(img).scaledToWidth(w)
            else:
                img = qimage2ndarray.array2qimage(img).scaledToHeight(h)
            self.setPixmap(qtg.QPixmap.fromImage(img))

    def update_hover(self, event=None):
        """
        update the hover position.

        Parameters
        ----------
        event: matplotlib.backend_bases.Event
            the event causing the hover update.

        values: any
            the values to be used for updating the hover.
        """
        if self.data is not None:
            x_img = self.size().width()
            y_img = self.size().height()
            x_data = int((event.x() // (x_img / self.data.shape[1])))
            y_data = int(event.y() // (y_img / self.data.shape[0]))
            v_data = float(self.data[y_data, x_data])
            y = event.y() + int(round(self.hover_offset_x_perc * x_img))
            x = event.x() + int(round(self.hover_offset_y_perc * y_img))
            pnt = self.mapToGlobal(qtc.QPoint(x, y))
            self.hover.move(pnt.x(), pnt.y())
            self.hover.update(x=x, y=y, temperature=v_data)

    def update_view(self, data=None):
        """
        update the image displayed by the object.

        Parameters
        ----------
        data: numpy 2D array
            the data to be displayed.
        """
        self.data = data
        self._adjust_view()


class CameraWidget(qtw.QWidget):
    """
    Initialize a PyQt5 widget capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.

    Parameters
    ----------

    hover_offset_x: float
        the percentage of the screen width that offsets the hover with respect
        to the position of the mouse.

    hover_offset_y: float
        the percentage of the screen height that offsets the hover with respect
        to the position of the mouse.

    colormap: str
        the colormap to be applied
    """

    # private variables
    _size = 50
    timer = None
    zoom_spinbox = None
    frequency_spinbox = None
    thermal_image = None
    fps_label = None
    rotation_button = None
    recording_pane = None
    opt_pane = None
    device = None

    def _create_box(self, title, obj):
        """
        create a groupbox with the given title and incorporating obj.

        Parameters
        ----------
        title: str
            the box title

        obj: QWidget
            the object to be included

        Returns
        -------
        box: QGroupBox
            the box.
        """
        # check the entries
        assert isinstance(title, str), "title must be a string."
        assert isinstance(obj, qtw.QWidget), "obj must be a QWidget."

        # generate the input layout
        layout = qtw.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(obj)
        pane = qtw.QGroupBox(title)
        pane.setLayout(layout)
        pane.setFont(FONT)
        return pane

    def start(self):
        """
        start the timer.
        """
        try:
            self.timer.stop()
            self.timer.start(int(round(1000.0 * self.device._dt)))
        except Exception:
            pass

    def show(self):
        """
        make the widget visible.
        """
        self.device.capture(save=False)
        self.start()
        super().show()

    def update_frequency(self):
        """
        set the sampling frequency.
        """
        self.device.interrupt()
        fs = self.frequency_spinbox.value()
        self.device.set_sampling_frequency(fs)
        self.device.capture(save=False)
        self.start()

    def rotate(self):
        """
        set the rotation angle.
        """
        self.device.set_angle(self.device.angle + 90)
        self.resizeEvent()

    def start_recording(self):
        """
        function handling what happens at the start of the recording.
        """
        self.device.interrupt()
        self.device.capture(save=True)

    def stop_recording(self):
        """
        function handling what happens at the stop of the recording.
        """
        self.device.interrupt()
        if len(self.device._data) > 0:
            # let the user decide where to save the data
            file_filters = "H5 (*.h5)"
            file_filters += ";;NPZ (*.npz)"
            file_filters += ";;JSON (*.json)"
            options = qtw.QFileDialog.Options()
            options |= qtw.QFileDialog.DontUseNativeDialog
            path, ext = qtw.QFileDialog.getSaveFileName(
                parent=self,
                filter=file_filters,
                directory=self.path,
                options=options,
            )

            # prepare the data
            if len(path) > 0:
                path = path.replace("/", os.path.sep)
                ext = ext.split(" ")[0].lower()
                if not path.endswith(ext):
                    path += "." + ext

                # save data
                self.setCursor(qtc.Qt.WaitCursor)
                try:
                    self.device.save(path)
                    folders = path.split(os.path.sep)[:-1]
                    self.path = os.path.sep.join(folders)

                except TypeError as err:
                    msgBox = qtw.QMessageBox()
                    msgBox.setIcon(qtw.QMessageBox.Warning)
                    msgBox.setText(err)
                    msgBox.setFont(FONT)
                    msgBox.setWindowTitle("ERROR")
                    msgBox.setStandardButtons(qtw.QMessageBox.Ok)
                    msgBox.exec()

                # reset the camera buffer and restart the data streaming
                finally:
                    self.setCursor(qtc.Qt.ArrowCursor)
                    self.device.clear()
        else:
            msgBox = qtw.QMessageBox()
            msgBox.setIcon(qtw.QMessageBox.Warning)
            msgBox.setText("NO DATA HAVE BEEN COLLECTED.")
            msgBox.setFont(FONT)
            msgBox.setWindowTitle("ERROR")
            msgBox.setStandardButtons(qtw.QMessageBox.Ok)
            msgBox.exec()

        # restart sampling
        self.device.capture(save=False)
        self.start()

    def _update_view(self):
        """
        update the last frame and display it.
        """
        tic = time.time()
        # NOTE: rotation is handled by LeptonCamera as it directly affects
        # the way the data are collected
        if self.device.last_reading is not None:
            img = np.squeeze(self.device.last_reading[1])
            self.thermal_image.update_view(img)
        toc = time.time()
        fps = 0 if toc == tic else (1 / (toc - tic))
        self.fps_label.setText("FPS: {:0.1f}".format(fps))

    def resizeEvent(self, event=None):
        w = self.thermal_image.sizeHint().width()
        h = self.sizeHint().height()
        self.resize(w, h)

    def __init__(
        self,
        hover_offset_x=0.02,
        hover_offset_y=0.02,
        colormap=cv2.COLORMAP_HOT,
    ):
        """
        constructor
        """
        super().__init__()

        # actual path
        self.path = os.getcwd()

        # find the Lepton Camera device
        self.device = Camera()

        # sampling frequency
        self.frequency_spinbox = qtw.QDoubleSpinBox()
        self.frequency_spinbox.setFont(FONT)
        self.frequency_spinbox.setDecimals(1)
        self.frequency_spinbox.setMinimum(1.0)
        self.frequency_spinbox.setSingleStep(0.1)
        self.frequency_spinbox.setMaximum(8.5)
        self.frequency_spinbox.setValue(8.5)
        self.frequency_spinbox.valueChanged.connect(self.update_frequency)
        freq_box = self._create_box("Frequency (Hz)", self.frequency_spinbox)

        # camera rotation
        rotation_icon = _QIcon(ROTATION, self._size)
        self.rotation_button = qtw.QPushButton(icon=rotation_icon)
        self.rotation_button.setFlat(True)
        self.rotation_button.setFixedHeight(self._size)
        self.rotation_button.setFixedWidth(self._size)
        self.rotation_button.clicked.connect(self.rotate)
        rotation_box = self._create_box("Rotate 90°", self.rotation_button)

        # recording
        self.recording_pane = _RecordingWidget()
        self.recording_pane.started.connect(self.start_recording)
        self.recording_pane.stopped.connect(self.stop_recording)
        recording_box = self._create_box("Data recording", self.recording_pane)

        # setup the options panel
        opt_pane = qtw.QWidget()
        opt_layout = qtw.QGridLayout()
        opt_layout.setSpacing(2)
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_layout.addWidget(freq_box, 0, 0)
        opt_layout.addWidget(rotation_box, 0, 1)
        opt_layout.addWidget(recording_box, 0, 2)
        opt_pane.setLayout(opt_layout)
        opt_pane.setFixedHeight(int(round(self._size * 1.5)))

        # thermal image
        self.thermal_image = _ImageWidget(
            hover_offset_x=hover_offset_x,
            hover_offset_y=hover_offset_y,
            colormap=colormap,
        )

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.thermal_image)
        layout.addWidget(opt_pane)
        central_widget = qtw.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # status bar
        size = FONT.pixelSize()
        family = FONT.family()
        self.fps_label = qtw.QLabel()
        self.fps_label.setFont(qtg.QFont(family, size // 2))
        self.statusBar().addPermanentWidget(self.fps_label)

        # icon and title
        self.setWindowIcon(_QIcon(MAIN, self._size))
        self.setWindowTitle("LeptonWidget")

        # stream handlers
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self._update_view)
        self.update_frequency()

        # avoid resizing
        self.statusBar().setSizeGripEnabled(False)
