__author__ = 'gbw'

import numpy as np
import Dvid_Access
import Stack_Bwdist_L_U16_2
import Stack_Local_Max2

# def IMAGE_ARRAY_HIST_M(stack, minmax_index):
#
#
#
# def hist(stack):
#     minmax_index = np.zeros(2)
#     IMAGE_ARRAY_HIST_M(stack, minmax_index)
#
# def SubtractBackground(stackData, minFr, maxIter):
#     hist1 = hist(stackData)
#
#
#
# def computeSeedPosition(stack):
#     if stack.size != 0:
#         # startProgress();
#         bw = stack
#         # SubtractBackground(bw, 0.5, 3)
#         # binarize(bw, bw)
#         # C_Stack::translate(bw, GREY, 1)
#
#         print "Removing noise ..."
#
#         # Stack *mask = bwsolid(bw);
#         # advanceProgress(0.05);

if __name__ == '__main__':
    dvidurl = "localhost:8000"
    uuid = "09c0"
    box = np.zeros(6)
    box[0] = 100
    box[1] = 100
    box[2] = 2600
    box[3] = 200
    box[4] = 200
    box[5] = 2700

    stack = Dvid_Access.readBlock(box, dvidurl, uuid)

    # if stack.size != 0:
    #     computeSeedPosition(stack)



