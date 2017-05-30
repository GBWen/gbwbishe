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
    file = open(file_url + "/input_sort1.txt")
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

    stack = np.array(stack)
    stack = stack.reshape(1, 100, 100, 100)

    locseg = [locseg_seg_r1, locseg_seg_c, locseg_seg_theta, locseg_seg_psi,\
                    locseg_seg_h, locseg_seg_curvature, locseg_seg_alpha, locseg_seg_scale,\
                    locseg_pos[0], locseg_pos[1], locseg_pos[2], z_scale]

    # locseg = [1.500000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 57.000000, 4.000000, 0.000000, 1.000000]
    # locseg = [3.464102, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 85.000000, 82.000000, 9.000000, 1.000000]
    # locseg = [1.500000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 57.000000, 4.000000, 0.000000, 1.000000]
    # locseg = [3.316625, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 89.000000, 84.000000, 6.000000, 1.000000]
    # locseg = [1.500000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 44.000000, 16.000000, 16.000000, 1.000000]
    # locseg = [1.500000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 57.000000, 4.000000, 0.000000, 1.000000]
    # locseg = [1.500000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 44.000000, 16.000000, 16.000000, 1.000000]

    locseg = [3.316625, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 1.000000, 79.000000, 79.000000, 13.000000, 1.000000]

    score = Local_Neuroseg.Local_Neuroseg_Optimize_W(locseg, stack, 1.0, 0)


    print locseg
    print score

    locseg = [ 3.36044415e+00,   0.00000000e+00,   5.55667959e+00,
         1.47975557e-01,   1.10000000e+01,   0.00000000e+00,
         0.00000000e+00,   2.28587831e-01,   4.96417918e+02,
         1.02754568e+02,   2.60250408e+03,   4.51453950e-01,  1.0]

    # score = 1
    #
    # print "Tracing:"
    #
    # m_locsegList = []
    # if score > 0.3:
    #     m_locsegList.append(locseg)
    #
    #     tailSeg = m_locsegList[len(m_locsegList)-1]
    #     headSeg = m_locsegList[0]
    #
    #     while (len(tailSeg) > 0) | (len(headSeg) > 0):
    #         if (len(tailSeg) > 0):
    #             # print "head"
    #             locseg = Local_Neuroseg.Next_Local_Neuroseg(tailSeg, 0.5)
    #             score = Local_Neuroseg.fit(locseg)
    #             print locseg
    #             print score
    #             if (score >= 0.3) & (Local_Neuroseg.hitTraced(locseg, m_locsegList) == False):
    #                 m_locsegList.append(locseg)
    #                 tailSeg = locseg
    #             else:
    #                 tailSeg = np.zeros(0)
    #         if (len(headSeg) > 0):
    #             # print "tail"
    #             Local_Neuroseg.Flip_Local_Neuroseg(headSeg)
    #             locseg = Local_Neuroseg.Next_Local_Neuroseg(headSeg, 0.5)
    #             Local_Neuroseg.Flip_Local_Neuroseg(locseg)
    #             score = Local_Neuroseg.fit(locseg)
    #             print locseg
    #             print score
    #             if (score >= 0.3) & (Local_Neuroseg.hitTraced(locseg, m_locsegList) == False):
    #                 Local_Neuroseg.Flip_Local_Neuroseg(locseg)
    #                 m_locsegList.append(locseg)
    #                 headSeg = locseg
    #             else:
    #                 headSeg = np.zeros(0)
    #
    #     print "Ans"
    #     print m_locsegList