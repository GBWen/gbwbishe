__author__ = 'gbw'

import numpy as np
import math
import httplib
from pydvid import voxels
import Local_Neuroseg

def readStack(x, y, z, r, address, port, uuid):
    xmin = 0
    xmax = 99
    ymin = 0
    ymax = 99
    zmin = 0
    zmax = 99

    if (Local_Neuroseg.IS_IN_CLOSE_RANGE(x, xmin, xmax)) & (Local_Neuroseg.IS_IN_CLOSE_RANGE(y, ymin, ymax)) & (Local_Neuroseg.IS_IN_CLOSE_RANGE(z, zmin, zmax)):
        box = np.zeros(6)
        box[0] = math.floor(x - r)
        box[1] = math.floor(y - r)
        box[2] = math.floor(z - r)
        box[3] = math.ceil(x + r) + 1
        box[4] = math.ceil(y + r) + 1
        box[5] = math.ceil(z + r) + 1


        boxSize = int((box[5] - box[2]) * (box[4] - box[1]) * (box[3] - box[0]))

        connection = httplib.HTTPConnection(address + ":" + port, timeout=5.0)
        # uuid = "ae87"
        dvid_volume = voxels.VoxelsAccessor(connection, uuid, "grayscale")

        stack = dvid_volume.get_ndarray((0, int(box[0]), int(box[1]), int(box[2])), (1, int(box[3]), int(box[4]), int(box[5])))
        stack2 = np.zeros(boxSize)
        t, a, b, c = stack.shape
        index = 0
        for i in range(c):
            for j in range(b):
                for k in range(a):
                    stack2[index] = stack[0][k][j][i]
                    k += 1
                    index += 1
                j += 1
            i += 1
        return stack2.reshape(1, int(box[3] - box[0]), int(box[4] - box[1]), int(box[5] - box[2]))
    else:
        return np.zeros(0)

# def readBlock(box, dvidurl, uuid):
#     boxSize = int((box[5] - box[2]) * (box[4] - box[1]) * (box[3] - box[0]))
#     connection = httplib.HTTPConnection(dvidurl, timeout=5.0)
#     dvid_volume = voxels.VoxelsAccessor(connection, uuid, "grayscale")
#     stack = dvid_volume.get_ndarray((0, int(box[0]), int(box[1]), int(box[2])), (1, int(box[3]), int(box[4]), int(box[5])))
#     stack = stack.reshape(boxSize)
#     return stack

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
