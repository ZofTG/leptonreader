# imports
import os
import platform
import sys

import clr  # needs the "pythonnet" package

# check whether python is running as 64bit or 32bit
# to import the right .NET dll
if platform.architecture()[0] == "64bit":
    sys.path.append(os.path.sep.join([os.getcwd(), "lepton_sdk_purethermal_windows10_1.0.2", "x64",]))
elif platform.architecture()[0] == "32bit":
    sys.path.append(os.path.sep.join([os.getcwd(), "lepton_sdk_purethermal_windows10_1.0.2", "x86",]))
else:
    raise TypeError(f"the {platform.architecture()[0]} architecture is not supported.",)
clr.AddReference("LeptonUVC")
clr.AddReference("ManagedIR16Filters")

from .core import *
from .gui import *
