__author__ = 'root'

import numpy as np
import Dvid_Access
import Local_Neuroseg
import Optimize

# def fit(locseg):
#     ball = np.zeros(4)
#     Local_Neuroseg.Local_Neuroseg_Ball_Bound(locseg, ball)
#     stack = Dvid_Access.readStack(ball[0], ball[1], ball[2], ball[3])
#     score = -1
#
#     if stack.size > 0:
#         # To do
#         # registerToRawStack(stack->getOffset(), locseg);
#         # Fit_Local_Neuroseg_W(locseg, stack->c_stack(), 1.0, m_fitWorkspace);
#         # registerToStack(stack->getOffset(), locseg);
#         # score = getFitScore();
#         score = 1
#     return score

if __name__ == '__main__':
    x = 57
    y = 68
    z = 34
    r = 5
    # 773 424 2654
    # 43 45 67 10.198

    ball = np.zeros(4)
    ball[0] = x
    ball[1] = y
    ball[2] = z
    ball[3] = r
    # print ball

    locseg = Local_Neuroseg.Spark_Optimize(ball)

    # locseg = [19.401664, 0.000000, 0.204780, 1.567118, 11.000000, 0.000000, 0.000000, 0.545005, 31.294245, 47.087712, 61.798826, 0.5]

    score = locseg[11]

    print locseg
    print score
    print

    # score = 1
    # locseg = [2.089561, 0.000000, 5.417318, 0.943172, 11.000000, 0.000000, 0.000000, 1.001310, 29.319142, 44.712037, 26.884238]
    # print locseg

    # locseg = [5.473846, 0.000000, 1.084860, 5.256380, 11.000000, 0.000000, 0.000000, 1.032702, 59.531654, 71.000877, 31.016171]

    m_locsegList = []
    if score >= 0.3:
        m_locsegList.append(locseg)

        tailSeg = [0 for i in range(11)]
        headSeg = [0 for i in range(11)]
        for i in range(11):
            tailSeg[i] = m_locsegList[len(m_locsegList)-1][i]
            headSeg[i] = m_locsegList[0][i]

        while (len(tailSeg) > 0) | (len(headSeg) > 0):
            if len(tailSeg) > 0:

                locseg = Local_Neuroseg.Next_Local_Neuroseg(tailSeg, 0.5)

                score = Local_Neuroseg.fit(locseg)

                if (score >= 0.3) & (Local_Neuroseg.hitTraced(locseg, m_locsegList) == False):
                    m_locsegList.append(locseg)
                    for i in range(11):
                        tailSeg[i] = locseg[i]
                else:
                    tailSeg = np.zeros(0)
            if len(headSeg) > 0:
                Local_Neuroseg.Flip_Local_Neuroseg(headSeg)
                locseg = Local_Neuroseg.Next_Local_Neuroseg(headSeg, 0.5)
                Local_Neuroseg.Flip_Local_Neuroseg(headSeg)
                score = Local_Neuroseg.fit(locseg)

                if (score >= 0.3) & (Local_Neuroseg.hitTraced(locseg, m_locsegList) == False):
                    Local_Neuroseg.Flip_Local_Neuroseg(locseg)
                    m_locsegList.insert(0, locseg)
                    for i in range(11):
                        headSeg[i] = locseg[i]
                else:
                    headSeg = np.zeros(0)
    for i in range(len(m_locsegList)):
        print i+1, 0, m_locsegList[i][8], m_locsegList[i][9], m_locsegList[i][10], m_locsegList[i][0], i

    # print "ANS"
    # print m_locsegList
    # return m_locsegList

    # m_locsegList = []
    # locseg = [19.401664, 0.000000, 0.204780, 1.567118, 11.000000, 0.000000, 0.000000, 0.545005, 31.294245, 47.087712, 61.798826]
    # m_locsegList.append(locseg)
    # locseg = [19.401664, 0.000000, 0.204780, 1.567118, 11.000000, 0.000000, 0.000000, 0.545005, 32.310997, 47.083972, 66.694355]
    # m_locsegList.append(locseg)
    # locseg = [18.827740, 0.000000, 2.635793, 1.352821, 11.000000, 0.000000, 0.000000, 0.550854, 33.327748, 47.080233, 71.589884]
    # print Local_Neuroseg.hitTraced(locseg, m_locsegList)



