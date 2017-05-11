__author__ = 'root'

import numpy as np
import Dvid_Access
import Local_Neuroseg
import Optimize

def fit(locseg):
    ball = np.zeros(4)
    Local_Neuroseg.Local_Neuroseg_Ball_Bound(locseg, ball)
    stack = Dvid_Access.readStack(ball[0], ball[1], ball[2], ball[3])
    score = -1

    if stack.size > 0:
        # To do
        # registerToRawStack(stack->getOffset(), locseg);
        # Fit_Local_Neuroseg_W(locseg, stack->c_stack(), 1.0, m_fitWorkspace);
        # registerToStack(stack->getOffset(), locseg);
        # score = getFitScore();
        score = 1
    return score

if __name__ == '__main__':
    x = 110
    y = 112
    z = 2635
    r = 1.0

    locseg = Local_Neuroseg.New_Local_Neuroseg()
    Local_Neuroseg.Set_Local_Neuroseg(locseg, x, y, z, r)

    score = optimize(locseg)

    m_locsegList = list()
    if score > 0.3:
        m_locsegList.append(locseg)

        tailSeg = m_locsegList[len(m_locsegList)-1]
        headSeg = m_locsegList[0]

        while (tailSeg.size > 0) | (headSeg.size > 0):
            if (tailSeg.size > 0):
                locseg = Local_Neuroseg.Next_Local_Neuroseg(tailSeg, 0.5)
                score = fit(locseg)
                if (score >= 0.3) & (Local_Neuroseg.hitTraced(locseg) == False):
                    m_locsegList.append(locseg)
                    tailSeg = locseg
                else:
                    tailSeg = np.zeros(0)
            if (headSeg.size > 0):
                Local_Neuroseg.Flip_Local_Neuroseg(headSeg)
                locseg = Local_Neuroseg.Next_Local_Neuroseg(headSeg, 0.5)
                Local_Neuroseg.Flip_Local_Neuroseg(headSeg)
                score = fit(locseg)
                if (score >= 0.3) & (Local_Neuroseg.hitTraced(locseg) == False):
                    Local_Neuroseg.Flip_Local_Neuroseg(locseg)
                    m_locsegList.append(locseg)
                    headSeg = locseg
                else:
                    headSeg = np.zeros(0)



