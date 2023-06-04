"""core leptonreader module"""

#! IMPORTS

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Tuple

import h5py
import numpy as np
from IR16Filters import IR16Capture, NewBytesFrameEvent
from Lepton import CCI
from scipy.ndimage import rotate

__all__ = ["TIMESTAMP_FORMAT", "read_file", "Camera"]


#! CONSTANTS


TIMESTAMP_FORMAT = "%Y-%b-%d %H:%M:%S.%f"


#! FUNCTIONS


def read_file(filename: str):
    """
    read the recorded data from file.

    Parameters
    ----------
    filename: a valid filename path

    Returns
    -------
    obj: dict
        a dict where each key is a timestamp which contains the
        corresponding image frame

    Notes
    -----
    the file extension is used to desume which file format is used.
    Available formats are:
        - ".h5" (gzip format with compression 9)
        - ".npz" (compressed numpy format)
        - ".json"
    """

    # check filename and retrieve the file extension
    assert isinstance(filename, str), "'filename' must be a str object."
    extension = filename.split(".")[-1].lower()

    # check the extension
    valid_extensions = np.array(["npz", "json", "h5"])
    txt = "file extension must be any of " + str(valid_extensions)
    assert extension in valid_extensions, txt

    # check if the file exists
    assert os.path.exists(filename), "{} does not exists.".format(filename)

    # timestamps parsing method
    def to_datetime(txt: Any):
        return datetime.strptime(str(txt), TIMESTAMP_FORMAT)

    # obtain the readed objects
    if extension == "json":  # json format
        with open(filename, "r") as buf:
            obj = json.load(buf)
        timestamps = map(to_datetime, list(obj.keys()))
        samples = np.squeeze(list(obj.values())).astype(np.float16)

    elif extension == "npz":  # npz format
        with np.load(filename, allow_pickle=True) as obj:
            timestamps = map(to_datetime, obj["timestamps"])
            samples = np.squeeze(obj["samples"]).astype(np.float16)

    elif extension == "h5":  # h5 format
        with h5py.File(filename, "r") as obj:
            timestamps = map(to_datetime, obj["timestamps"][:])  # type: ignore
            samples = np.squeeze(obj["samples"][:])  # type: ignore
            samples = samples.astype(np.float16)

    else:
        raise TypeError(f"{extension} files not supported.")

    return dict(zip(timestamps, samples))


#! CLASSES


class Camera:
    """
    Initialize a Lepton camera object capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.
    """

    # class variables
    _device = None
    _reader = None
    _data = {}
    _path = ""
    _first = None
    _last = None
    _dt = 200
    _angle = 0
    _sampling_frequency = 8.0

    def __init__(self):
        """
        constructor
        """
        # find a valid device
        devices = []
        for i in CCI.GetDevices():
            if i.Name.startswith("PureThermal"):
                devices += [i]

        # if multiple devices are found,
        # allow the user to select the preferred one
        if len(devices) > 1:
            print("Multiple Pure Thermal devices have been found.\n")
            for i, d in enumerate(devices):
                print("{}. {}".format(i, d))
            while True:
                idx = input("Select the index of the required device.")
                if isinstance(idx, int) and idx in range(len(devices)):
                    self._device = devices[idx]
                    break
                else:
                    print("Unrecognized input value.\n")

        # if just one device is found, select it
        elif len(devices) == 1:
            self._device = devices[0]

        # tell the user that no valid devices have been found.
        else:
            self._device = None

        # open the found device
        txt = "No devices called 'PureThermal' have been found."
        assert self._device is not None, txt
        self._device = self._device.Open()
        self._device.sys.RunFFCNormalization()

        # set the gain mode
        self._device.sys.SetGainMode(CCI.Sys.GainMode.HIGH)

        # set radiometric
        try:
            self._device.rad.SetTLinearEnableStateChecked(True)
        except Exception:
            print("this lepton does not support tlinear")

        # setup the buffer
        self._reader = IR16Capture()
        callback = NewBytesFrameEvent(self._add_frame)
        self._reader.SetupGraphWithBytesCallback(callback)

        # path init
        self._path = os.path.sep.join(__file__.split(os.path.sep)[:-4])

        # set the sampling frequency
        self.set_sampling_frequency(8.5)

        # set the rotation angle
        self.set_angle(0)

    def _add_frame(self, array: bytearray, width: int, height: int):
        """
        add a new frame to the buffer of readed data.
        """
        dt = datetime.now()  # time data
        img = np.fromiter(array, dtype="uint16").reshape(height, width)  # parse
        img = (img - 27315.0) / 100.0  # centikelvin --> celsius conversion
        img = rotate(img, angle=self.angle, reshape=True)  # rotation
        img = np.expand_dims(img, 2).astype(np.float16)
        self._last = (dt, img)  # update the last reading

    def capture(
        self,
        save: bool = True,
        n_frames: Tuple[int, None] = None, # type: ignore
        seconds: Tuple[float, int, None] = None, # type: ignore
    ):
        """
        record a series of frames from the camera.

        Parameters
        ----------
        save: bool
            if true the data are stored, otherwise nothing except
            "last" is updated.

        n_frames: None / int
            if a positive int is provided, n_frames are captured.
            Otherwise, all the frames collected are saved until the
            stop command is given.

        seconds: None / int
            if a positive int is provided, data is sampled for the indicated
            amount of seconds.
        """

        # check input
        assert isinstance(save, bool), "save must be a bool."
        if seconds is not None:
            txt = "'seconds' must be a float or int."
            assert isinstance(seconds, (float, int)), txt
        if n_frames is not None:
            txt = "'n_frames' must be an int."
            assert isinstance(n_frames, int), txt

        # start reading data
        self._reader.RunGraph()
        while self._last is None:
            pass

        # store the last data according to the given sampling
        # frequency
        if save:
            self._first = self._last
            self._data[self._last[0]] = self._last[1]

            def store():
                old = list(self._data.keys())[-1]
                processing_time = 0
                n = 0
                while self._first is not None and self._last is not None:
                    dt = (self._last[0] - old).total_seconds()
                    if dt >= self._dt - processing_time:
                        tic = time.time()
                        self._data[self._last[0]] = self._last[1]
                        old = self._last[0]
                        toc = time.time()
                        tm = toc - tic
                        processing_time = processing_time * n + tm
                        n += 1
                        processing_time /= n

            thread = threading.Thread(target=store)
            thread.start()

        # continue reading until a stopping criterion is met
        if seconds is not None:

            def stop_reading_sec(time):
                while self._first is None:
                    pass
                dt = 0
                while dt < time and self._last is not None:
                    dt = (self._last[0] - self._first[0]).total_seconds()
                self.interrupt()

            thread = threading.Thread(target=stop_reading_sec, args=[seconds])
            thread.run()

        elif n_frames is not None:

            def stop_reading_frm(n_frames):
                while len(self._data) < n_frames:
                    pass
                self.interrupt()

            thread = threading.Thread(target=stop_reading_frm, args=[n_frames])
            thread.run()

    def interrupt(self):
        """
        stop reading from camera.
        """
        self._reader.StopGraph()
        self._first = None

    @property
    def sampling_frequency(self):
        """
        return the actual sampling frequency
        """
        return float(self._sampling_frequency)

    def set_sampling_frequency(self, sampling_frequency: float):
        """
        set the sampling frequency value and update the _dt argument.

        Parameters
        ----------
        sampling_frequency: float, int
            the new sampling frequency
        """

        # check the input
        txt = "'sampling frequency' must be a value in the (0, 8.5] range."
        assert isinstance(sampling_frequency, (int, float)), txt
        assert 0 < sampling_frequency <= 8.5, txt
        self._sampling_frequency = np.round(sampling_frequency, 1)
        self._dt = 1.0 / self.sampling_frequency

    def set_angle(self, angle: float):
        """
        set the rotation angle in degrees.

        Parameters
        ----------
        angle: float
            the rotation angle in degrees.
        """
        assert isinstance(angle, (int, float)), "'angle' must be a float."
        self._angle = angle

    @property
    def angle(self):
        """
        return the rotation angle
        """
        return self._angle

    def is_recording(self):
        return self._first is not None

    def clear(self):
        """
        clear the current object memory and buffer
        """
        self._data = {}
        self._last = None
        self._first = None

    @property
    def buffer(self):
        """
        return the sampled data as dict with
        timestamps as keys and the sampled data as values.
        """
        return self._data

    @property
    def last_reading(self):
        """return the last reading"""
        return self._last

    def save(self, filename: str):
        """
        store the recorded data to file.

        Parameters
        ----------
        filename: a valid filename path

        Notes
        -----
        the file extension is used to desume which file format is required.
        Available formats are:

            - ".h5" (gzip format with compression 9)
            - ".npz" (compressed numpy format)
            - ".json"

        If an invalid file format is found a TypeError is raised.
        """

        # check filename and retrieve the file extension
        assert isinstance(filename, str), "'filename' must be a str object."
        extension = filename.split(".")[-1].lower()

        # ensure the folders exist
        if not os.path.exists(filename):
            root = os.path.sep.join(filename.split(os.path.sep)[:-1])
            os.makedirs(root, exist_ok=True)

        # prepare the data to be saved
        times = [x.strftime(TIMESTAMP_FORMAT) for x in list(self.buffer.keys())]
        samples = [x.tolist() for x in list(self.buffer.values())]

        # save the data
        if extension == "json":  # json format
            with open(filename, "w") as buf:
                json.dump(dict(zip(times, samples)), buf)

        elif extension == "npz":  # npz format
            np.savez(filename, timestamps=times, samples=samples)

        elif extension == "h5":  # h5 format
            h5f = h5py.File(filename, "w")
            h5f.create_dataset(
                "timestamps",
                data=times,
                compression="gzip",
                compression_opts=9,
            )
            h5f.create_dataset(
                "samples",
                data=samples,
                compression="gzip",
                compression_opts=9,
            )
            h5f.close()

        else:  # unsupported formats
            raise TypeError(f"{extension} extension not supported")
