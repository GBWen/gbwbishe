__author__ = 'root'

import numpy as np
import math
import httplib
from pydvid import voxels, general
import Local_Neuroseg


def readStack(x, y, z, r):
    xmin = 100
    xmax = 1123
    ymin = 100
    ymax = 1123
    zmin = 2600
    zmax = 2697

    if (Local_Neuroseg.IS_IN_CLOSE_RANGE(x, xmin, xmax)) & (Local_Neuroseg.IS_IN_CLOSE_RANGE(y, ymin, ymax)) & (Local_Neuroseg.IS_IN_CLOSE_RANGE(z, zmin, zmax)):
        box = np.zeros(6)
        box[0] = math.floor(x - r)
        box[1] = math.floor(y - r)
        box[2] = math.floor(z - r)
        box[3] = math.ceil(x + r)
        box[4] = math.ceil(y + r)
        box[5] = math.ceil(z + r)
        boxSize = int((box[5] - box[2]) * (box[4] - box[1]) * (box[3] - box[0]))

        connection = httplib.HTTPConnection("localhost:8000", timeout=5.0)
        uuid = "5c2d"
        dvid_volume = voxels.VoxelsAccessor(connection, uuid, "grayscale")
        stack = dvid_volume.get_ndarray((0, int(box[0]), int(box[1]), int(box[2])), (1, int(box[3]), int(box[4]), int(box[5])))
        stack = stack.reshape(boxSize)
        return stack
    else:
        return np.zeros(0)

def registerToRawStack(offset, locseg):
    pos = np.zeros(3)
    Local_Neuroseg.Local_Neuroseg_Center(locseg, pos)
    pos[0] -= offset[0]
    pos[1] -= offset[1]
    pos[2] -= offset[2]
    Local_Neuroseg.Set_Neuroseg_Position(locseg, pos, 2)

def registerToStack(offset, locseg):
    pos = np.zeros(3)
    Local_Neuroseg.Local_Neuroseg_Center(locseg, pos)
    pos[0] += offset[0]
    pos[1] += offset[1]
    pos[2] += offset[2]
    Local_Neuroseg.Set_Neuroseg_Position(locseg, pos, 2)
