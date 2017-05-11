__author__ = 'root'

import numpy as np
import math

import Dvid_Access
import Local_Neuroseg

if __name__ == '__main__':
    # x = 110
    # y = 112
    # z = 2635
    # r = 1.0
    #
    # locseg = Local_Neuroseg.New_Local_Neuroseg()
    # Local_Neuroseg.Set_Local_Neuroseg(locseg, x, y, z, r)
    #
    # ball = np.zeros(4)
    # Local_Neuroseg.Local_Neuroseg_Ball_Bound(locseg, ball)
    # stack = Dvid_Access.readStack(ball[0], ball[1], ball[2], ball[3])
    #
    # score = -1.0
    #
    # if stack.size > 0:
    #     fws_n = 2
    #     fws_options = np.array([0, 1])
    #     fws_pos_adjust = 1
    #
    #     stack_offset = np.zeros(3)
    #     stack_offset[0] = math.floor(x - r)
    #     stack_offset[1] = math.floor(y - r)
    #     stack_offset[2] = math.floor(z - r)
    #
    #     Dvid_Access.registerToRawStack(stack_offset, locseg)
    #     # Local_Neuroseg_Optimize_W(locseg, stack->c_stack(), 1.0, 0, fws)
    #     fws_pos_adjust = 0
    #     Local_Neuroseg.Flip_Local_Neuroseg(locseg)
    #     # Fit_Local_Neuroseg_W(locseg, stack->c_stack(), 1.0, fws);
    #     Local_Neuroseg.Flip_Local_Neuroseg(locseg)
    #     # Fit_Local_Neuroseg_W(locseg, stack->c_stack(), 1.0, fws);
    #     Dvid_Access.registerToStack(stack_offset, locseg)



    file_url = "/home/gbw/PycharmProjects/DvidSpark/smalldata"
    file = open(file_url + "/input_sort2.txt")
    try:
        str = file.readline()
        arr = str.split(' ')
        locseg_seg_r1 = float(arr[0])
        locseg_seg_c = float(arr[1])
        locseg_seg_theta = float(arr[2])
        locseg_seg_psi = float(arr[3])
        locseg_seg_h = float(arr[4])
        locseg_seg_curvature = float(arr[5])
        locseg_seg_alpha = float(arr[6])
        locseg_seg_scale = float(arr[7])
        locseg_pos = np.zeros(3)
        locseg_pos[0] = float(arr[8])
        locseg_pos[1] = float(arr[9])
        locseg_pos[2] = float(arr[10])
        z_scale = float(arr[11])

        str = file.readline()
        arr = str.split(' ')
        stack_depth = int(arr[0])
        stack_width = int(arr[1])
        stack_height = int(arr[2])

        area = stack_depth * stack_height * stack_width
        row = file.readline()
        words = row.split(' ')
        stack = []
        for i in range(area):
            stack.append(int(words[i]))
    finally:
         file.close()

    fws_n = 2
    fws_options = np.array([0, 1])
    fws_pos_adjust = 1
    fws_scores = np.zeros(2)

    locseg = [locseg_seg_r1, locseg_seg_c, locseg_seg_theta, locseg_seg_psi,\
                    locseg_seg_h, locseg_seg_curvature, locseg_seg_alpha, locseg_seg_scale,\
                    locseg_pos[0], locseg_pos[1], locseg_pos[2], z_scale]

    score = Local_Neuroseg.Fit_Local_Neuroseg_W(locseg, stack, 1.0, fws_n, fws_options, fws_scores, fws_pos_adjust)

    print score