---
author: Luca Zoffoli
jupyter:
  kernelspec:
    display_name: python38
    language: python
    name: python3
  language_info:
    name: python
    version: 3.8.0
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
title: README
---

python integration for FLIR Lepton 3.5 with GUI

<br>
<br>

# GETTING READY

1.  Install the python package via *pip*

         pip install "leptonreader @ git+https://github.com/ZofTG/leptonreader.git" --no-build-isolation

2.  Navigate to the installation folder and make sure to unlock the SDK
    dlls:

    -   Navigate to the x64 or x86 depending on if you have 64 bit (x64)
        or 32 bit (x86) python.
    -   Right click on LeptonUVC.dll and select Properties.
    -   In the general tab there may be a section called \"Security\" at
        the bottom. If there is, check \"Unblock\" and hit apply.
    -   Repeat for ManagedIR16Filters.dll.

3.  Install the latest redistributable according to your system
    architecture
    ([64bit](https://aka.ms/vs/17/release/vc_redist.x64.exe),
    [32bit](https://aka.ms/vs/17/release/vc_redist.x86.exe))

4.  Install the latest .NET runtime according to your system
    architecture
    ([64bit](https://dotnet.microsoft.com/en-us/download/dotnet/thank-you/runtime-desktop-7.0.5-windows-x64-installer),
    [32bit](https://dotnet.microsoft.com/en-us/download/dotnet/thank-you/runtime-desktop-7.0.5-windows-x86-installer))

5.  Physically connect a purethermal board to your PC.

<br>
<br>

# USAGE

<br>

## USING *Camera* INTERFACE

That provides an instance of the connected *Lepton Camera*. This object can be used to obtain data directly from the *Lepton* device. The following code provides an example of the *Camera* class usage.

``` python
# import packages
import os
from datetime import datetime
from random import randint
from time import sleep
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import leptonreader

# initialize the camera object
camera = leptonreader.Camera()

# capture 10 frames
camera.capture(save=True, n_frames=10)

# save the frames in "h5 format"
frames_file = os.path.sep.join([os.getcwd(), "frames.h5"])
camera.save(frames_file)
camera.clear()  # ensure to free the stored data buffer

# capture 2 seconds with sampling frequency set at 3Hz
camera.set_sampling_frequency(3)
camera.capture(save=True, seconds=2)

# save the collected data in "npz format"
seconds_file = os.path.sep.join([os.getcwd(), "seconds.npz"])
camera.save(seconds_file)
camera.clear()

# collect data for a random amount of time in "json format"
camera.set_sampling_frequency(8)
camera.capture(save=True)
sleep(randint(1, 5))
camera.interrupt()
random_file = os.path.sep.join([os.getcwd(), "random.json"])
camera.save(random_file)

# since we have not cleared the camera buffer (yet) let's have a look
# at the available frames.
# NOTE: The camera buffer is a dict where each key is a
# datetime.datetime object denoting the timestamp and each value is a
# 3D numpy array with shape (height, width, 1)
def plot_acquisitions(data:Dict[datetime, np.ndarray]):
    nframes = len(data)
    _, ax = plt.subplots(nframes, 1)
    for i, obj in enumerate(data.items()):
        timestamp, temparr = obj
        title = timestamp.strftime(leptonreader.TIMESTAMP_FORMAT)
        values = np.squeeze(temparr)
        ax[i].imshow(values)
        ax[i].set_title(title)
    plt.tight_layout()
    plt.show()

plot_acquisitions(camera.buffer)
```

<br>

## USING A GUI APPLICATION

This allows to obtain an easy to use graphical framework that provides the same functionalities of *Camera*.

``` python
# import the packages
import sys
import PyQt5.QWidgets as qtw
import PyQt5.QtCore as qtc

# launch the application
if __name__ == "__main__":

    # highdpi scaling
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

    # app generation
    app = qtw.QApplication(sys.argv)
    camera = leptonreader.CameraWidget()
    camera.show()
    sys.exit(app.exec_())
```

<br>

## READING FROM FILE

The collected data can be easily retrieved and disposed at will. For instance in this case the collected data will be plotted for visual inspection.

``` python
plot_acquisitions(leptonreader.read_file(seconds_file))
```
