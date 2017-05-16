__author__ = 'gbw'

import numpy as np
import math
import Dvid_Access
# try:
#     import Dvid_Access
# except Exception,e:
#     print 'not found Dvid_Access here.'
#     sys.exit(1)

def New_Local_Neuroseg():
    return np.zeros(11)

def Set_Local_Neuroseg(locseg, x, y, z, r):
    locseg[0] = locseg_seg_r1 = r + r
    locseg[1] = locseg_seg_c = 0.0
    locseg[2] = locseg_seg_theta = 0.0
    locseg[3] = locseg_seg_psi = 0.0
    locseg[4] = locseg_seg_h = 11.0
    locseg[5] = locseg_seg_curvature = 0.0
    locseg[6] = locseg_seg_alpha = 0.0
    locseg[7] = locseg_seg_scale = 1.0
    locseg_pos = np.zeros(3)
    locseg[8] = locseg_pos[0] = x
    locseg[9] = locseg_pos[1] = y
    locseg[10] = locseg_pos[2] = z

def Neuroseg_Ball_Range(seg):
    r = NEUROSEG_RB(seg)
    if seg[7] > 1.0:
        r *= seg[7]
    return math.sqrt(seg[4] * seg[4] + r * r)

def Local_Neuroseg_Ball_Bound(locseg, ball):
    Local_Neuroseg_Center(locseg, ball)
    ball[3] = Neuroseg_Ball_Range(locseg) / 2.0


def NEUROSEG_COEF(seg):
    # (((seg)->h == 1.0) ? (seg)->c : dmax2((seg)->c, (NEUROSEG_MIN_R - (seg)->r1) / ((seg)->h - 1.0)))
    ans = seg[1]
    if seg[0] == 1.0:
        return ans
    else:
        if (0.5 - seg[0]) / (seg[0] - 1.0) > ans:
            ans = (0.5 - seg[0]) / (seg[0] - 1.0)
    return ans

def Next_Neuroseg(seg1, seg2, pos_step):
    seg2 = seg1
    seg2[0] = seg1[0] + pos_step * NEUROSEG_COEF(seg1) * (seg1[0] - 1.0)

    if seg2[0] < 0.5:
        seg2[0] = 0.5
    seg2[4] = 11.0
    seg2[1] = NEUROSEG_COEF(seg2)

    return seg2

def Rotate_XZ(input, n, theta, psi, inverse):
    output = input
    size = len(input)
    Ar0 = math.cos(theta)
    Ar1 = math.sin(theta)
    Ar2 = math.cos(psi)
    Ar3 = math.sin(psi)
    offset = 0
    offsety = 1
    offsetz = 2
    result = np.zeros(3)

    # print n
    # print input.size

    if inverse == 0:
        for i in range(n):
            result[2] = Ar1 * input[(offsety + offset) % size] + Ar0 * input[(offsetz + offset) % size]
            result[0] = input[(offsetz + offset) % size] * Ar1 - input[(offsety + offset) % size] * Ar0
            result[1] = input[offset] * Ar3 - result[0] * Ar2
            result[0] = input[offset] * Ar2 + result[0] * Ar3
            for j in range(3):
                output[j + offset] = result[j]
            offset += 3
    else:
        for i in range(n):
            result[0] = Ar2 * input[offset] + Ar3 * input[(offsety + offset) % size]
            result[1] = input[(offsety + offset) % size] * Ar2 - input[offset] * Ar3
            result[2] = input[(offsetz + offset) % size] * Ar0 - result[1] * Ar1
            result[1] = input[(offsetz + offset) % size] * Ar1 + result[1] * Ar0
            for j in range(3):
                output[j + offset] = result[j]
            offset += 3
    return output

def Rotate_Z(input, n, alpha, inverse):
    output = input

    if alpha != 0.0:
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        inOffset = 0
        outOffset = 0

        if inverse == 0:
            for i in range(n):
                tmp = input[0 + inOffset] * cos_a - input[1 + inOffset] * sin_a
                output[1 + outOffset] = input[0 + inOffset] * sin_a + input[1 + inOffset] * cos_a
                output[0 + outOffset] = tmp
                output[2 + outOffset] = input[2 + inOffset]
                inOffset += 3
                outOffset += 3
        else:
            for i in range(n):
                tmp = input[0 + inOffset] * cos_a + input[1 + inOffset] * sin_a
                output[1 + outOffset] = -input[0 + inOffset] * sin_a + input[1 + inOffset] * cos_a
                output[0 + outOffset] = tmp
                output[2 + outOffset] = input[2 + inOffset]
                inOffset += 3
                outOffset += 3
    else:
        output[0] = input[0]
        output[1] = input[1]
        output[2] = input[2]

    return output

def Neuroseg_Axis_Offset(seg, axis_offset, pos_offset):
    pos_offset[0] = 0.0
    pos_offset[1] = 0.0
    pos_offset[2] = axis_offset
    pos_offset = Rotate_XZ(pos_offset, 1, seg[2], seg[3], 0)

def Local_Neuroseg_Bottom(locseg, pos):
    pos[0] = locseg[8]
    pos[1] = locseg[9]
    pos[2] = locseg[10]

def Neuropos_Absolute_Coordinate(pos, apos):
    apos[0] += pos[0]
    apos[1] += pos[1]
    apos[2] += pos[2]

def Local_Neuroseg_Axis_Position(locseg, apos, axis_offset):
    Neuroseg_Axis_Offset(locseg, axis_offset, apos)
    bottom = np.zeros(3)
    Local_Neuroseg_Bottom(locseg, bottom)
    Neuropos_Absolute_Coordinate(bottom, apos)

def Set_Neuropos(locseg, x, y, z):
    locseg[8] = x
    locseg[9] = y
    locseg[10] = z

def Neuropos_Translate(pos, apos):
    pos[8] += apos[0]
    pos[9] += apos[1]
    pos[10] += apos[2]

def Set_Neuroseg_Position(locseg, pos, ref):
    Set_Neuropos(locseg, pos[0], pos[1], pos[2])
    NEUROSEG_BOTTOM = 0
    NEUROSEG_TOP = 1
    NEUROSEG_CENTER = 2
    Neuropos_Reference = 0
    apos = np.zeros(3)

    if ref != Neuropos_Reference:
        axis_offset = 0
        if Neuropos_Reference == NEUROSEG_BOTTOM :
            if ref == NEUROSEG_TOP:
                axis_offset = -locseg[4] + 1.0
            if ref == NEUROSEG_CENTER:
                axis_offset = -(locseg[4] - 1.0) / 2.0
        elif Neuropos_Reference == NEUROSEG_TOP:
            if ref == NEUROSEG_BOTTOM:
                axis_offset = locseg[4] - 1.0
            if ref == NEUROSEG_CENTER:
                axis_offset = (locseg[4] - 1.0) / 2.0
        elif Neuropos_Reference == NEUROSEG_CENTER:
            if ref == NEUROSEG_BOTTOM:
                axis_offset = (locseg[4] - 1.0) / 2.0
            if ref == NEUROSEG_TOP:
                axis_offset = -(locseg[4] - 1.0) / 2.0

        # print axis_offset
        # print locseg
        # print pos

        Neuroseg_Axis_Offset(locseg, axis_offset, apos)

        # print apos

        Neuropos_Translate(locseg, apos)

def Next_Local_Neuroseg(locseg1, pos_step):
    locseg2 = New_Local_Neuroseg()
    locseg2 = Next_Neuroseg(locseg1, locseg2, pos_step)

    bottom = np.zeros(3)
    Local_Neuroseg_Axis_Position(locseg1, bottom, pos_step * (locseg1[0] - 1.0))
    Set_Neuroseg_Position(locseg2, bottom, 0)

    return locseg2

def Local_Neuroseg_Center(locseg, ball):
    pos = np.zeros(3)
    pos[0] = ball[0]
    pos[1] = ball[1]
    pos[2] = ball[2]
    Local_Neuroseg_Axis_Position(locseg, pos, (locseg[4] - 1.0) / 2.0)
    ball[0] = pos[0]
    ball[1] = pos[1]
    ball[2] = pos[2]

def NEUROSEG_R2(seg):
    return seg[0] + (seg[4] - 1.0) * NEUROSEG_COEF(seg)

def NEUROSEG_RB(seg):
    if seg[1] <=0:
        return seg[0]
    else:
        return NEUROSEG_R2(seg)

def IS_IN_CLOSE_RANGE(x, minv, maxv):
    return (x >= minv) & (x <= maxv)

def NEUROSEG_RADIUS(seg, z):
    return seg[0] + z * NEUROSEG_COEF(seg)

def Neuroseg_Hit_Test(seg, x, y, z):
    if (z >= -0.5) & (z <= seg[4] - 0.5):
        d2 = (x * x) / (seg[7] * seg[7]) + y * y
        r = NEUROSEG_RADIUS(seg, z)
        if d2 <= r * r:
            return True
    return False

def Local_Neuroseg_Hit_Test(locseg, x, y, z):
    tmp_pos = np.zeros(3)
    Local_Neuroseg_Bottom(locseg, tmp_pos)

    tmp_pos[0] = x - tmp_pos[0]
    tmp_pos[1] = y - tmp_pos[1]
    tmp_pos[2] = z - tmp_pos[2]

    tmp_pos = Rotate_XZ(tmp_pos, 1, locseg[2], locseg[3], 1)
    tmp_pos = Rotate_Z(tmp_pos, 1, locseg[2], 1)

    return Neuroseg_Hit_Test(locseg, tmp_pos[0], tmp_pos[1], tmp_pos[2])

def hitTraced(locseg, m_locsegList):
    hit = False
    top = np.zeros(3)
    Local_Neuroseg_Top(locseg, top)
    for seg in m_locsegList:
        if (Local_Neuroseg_Hit_Test(seg, top[0], top[1], top[2])):
            hit = True
            break
    return hit

def Local_Neuroseg_Top(locseg, pos):
    Local_Neuroseg_Axis_Position(locseg, pos, locseg[4] - 1.0)

def Flip_Local_Neuroseg(locseg):
    pos = np.zeros(3)
    Local_Neuroseg_Top(locseg, pos)
    Set_Neuroseg_Position(locseg, pos, 0)

    TZ_PI = 3.14159265358979323846264338328
    locseg[2] += TZ_PI
    locseg[0] = NEUROSEG_R2(locseg)
    locseg[1] = -locseg[1]

def darray_sqsum(d1, length):
    result = 0.0
    for i in range(length):
        result += d1[i] * d1[i]
    return result

def darray_norm(d1, length):
    return math.sqrt(darray_sqsum(d1, length))

def Local_Neuroseg_Var(locseg, param):
    for i in range(11):
        param[i] = locseg[i]

def Local_Neuroseg_Param_Array(locseg, z_scale, param):
    Local_Neuroseg_Var(locseg, param)

def Local_Neuroseg_Set_Var(locseg, var_index, value):
    locseg[var_index] = value

def Normalize_Radian(r):
    TZ_PI = 3.14159265358979323846264338328
    TZ_2PI = 6.2831853071795862319959269
    norm_r = r
    if (r < 0.0) | (r >= TZ_2PI):
        norm_r = r - math.floor(r / TZ_2PI) * 2.0 * TZ_PI
    return norm_r

def Construct_Geo3d_Scalar_Field(size):
    field_size = size
    field_points = np.zeros((field_size, 3))
    # print field_points
    field_values = np.zeros(field_size)
    return field_size, field_points, field_values

def neurofield(x, y):
    t = x * x + y * y
    return (1 - t) * math.exp(-t)

def NEUROSEG_PLANE_FIELD(points, values, length, x, y):
    points[length][0] = x
    points[length][1] = y
    values[length] = neurofield(x, y)
    length += 1
    if x != 0.0:
        points[length][0] = -x
        points[length][1] = y
        values[length] = values[length - 1]
        length += 1
    if y != 0.0:
        points[length][0] = x
        points[length][1] = -y
        values[length] = values[length - 1]
        length += 1
        if x != 0.0:
            points[length][0] = -x
            points[length][1] = -y
            values[length] = values[length - 1]
            length += 1
    return points, values, length

def Neuroseg_Slice_Field(points, values):
    start = 0.2
    end = 1.65
    y = 0.0
    x = 0.0
    step = 0.2
    offset = 0
    range = (end - 0.05) * (end - 0.05) + 0.1
    length = 0

    values[length] = neurofield(x, y)
    points[length][0] = 0.0
    points[length][1] = 0.0
    length += 1

    for x in np.arange(start, end, step):
        points[length][0] = x
        points[length][1] = y
        values[length] = neurofield(x, y)
        length += 1
        points[length][0] = -x
        points[length][1] = y
        values[length] = values[length - 1]
        length += 1


    x = 0.0
    for y in np.arange(start, end, step):
        points[length][0] = x
        points[length][1] = y
        values[length] = neurofield(x, y)
        length += 1
        points[length][0] = x
        points[length][1] = -y
        values[length] = values[length - 1]
        length += 1

    for y in np.arange(start, 0.85, step):
        for x in np.arange(start, end, step):
            if x * x + y * y < range:
                points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y)

    y = 1.0
    for x in np.arange(start, 1.65, step):
        if x * x + y * y < range:
            points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y)

    for y in np.arange(1.2, 1.45, step):
        for x in np.arange(0.2, 1.45, step):
            if x * x + y * y < range:
                points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y)

    if end >= 1.6:
        y = 1.6
        for x in np.arange(start, 1.05, step):
            if x * x + y * y < range:
                points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y)
    return length

def Geo3d_Point_Array_Bend(points, n, c):
    d = 0
    for i in range(n):
        d = c - points[i][1]
        if d == 0.0:
            points[i][1] = 0.0
            points[i][2] = 0.0
        else:
            points[i][2] /= d
            points[i][1] += d - d * math.cos(points[i][2])
            points[i][2] = math.sin(points[i][2])
        points[i][2] *= d

def Neuroseg_Field_S_Fast(seg, field_size, field_points, field_values):
    NEUROSEG_DEFAULT_H = 11
    nslice = NEUROSEG_DEFAULT_H
    NEUROSEG_SLICE_FIELD_LENGTH = 277
    # New_Geo3d_Scalar_Field
    if field_size == 0:
        field_size, field_points, field_values = Construct_Geo3d_Scalar_Field(nslice * NEUROSEG_SLICE_FIELD_LENGTH)

    z_start = 0.0
    z_step = (seg[4] - 1.0) / (nslice - 1)
    # points = field_points
    # values = field_values
    coef = NEUROSEG_COEF(seg) * z_step
    r = seg[0]
    z = z_start

    length = Neuroseg_Slice_Field(field_points, field_values)
    field_size = length

    weight = 0.0
    r_scale = r * seg[7]
    sqrt_r = math.sqrt(r)
    sqrt_sqrt_scale = math.sqrt(math.sqrt(seg[7]))
    sqrt_r_scale = sqrt_r * sqrt_sqrt_scale

    # points = field_points
    # values = field_values

    if (coef != 0.0) & (nslice > 1):
    # memcpy(field->values + length, field->values, sizeof(double) * length);
        for i in range(length):
            field_values[length + i] = field_values[i]

    for i in range(length):
        field_points[i][2] = z

        pt = field_points[i]
        if field_values[i] >= 0:
            pt[0] *= r_scale
            pt[1] *= r
            field_values[i] *= sqrt_r_scale
        else:
            norm = math.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
            alpha = norm - 1
            pt[0] *= r_scale / norm
            pt[1] *= r / norm

            enorm = pt[0] * pt[0] + pt[1] * pt[1]
            if enorm < 1.0:
                alpha /= math.sqrt(enorm)
            elif enorm > 4.0:
                alpha /= math.sqrt(enorm)
                alpha += alpha
            pt[0] *= 1.0 + alpha
            pt[1] *= 1.0 + alpha
        weight += math.fabs(field_values[i])
        field_points[i] = pt

    for i in range(length):
        field_values[i] /= weight

    if (coef != 0.0) & (nslice > 1):
    # memcpy(field->values + length, field->values, sizeof(double) * length);
        for i in range(length):
            field_values[length + i] = field_values[i]

    pointsOffset = 0
    valuesOffset = 0
    points = field_points
    values = field_values

    for j in range(1, nslice):
        z += z_step
        if coef != 0.0:
            r += coef
            r_scale = r * seg[7]
            sqrt_r = math.sqrt(r)
            sqrt_r_scale = sqrt_r * sqrt_sqrt_scale

        pointsOffset += length
        valuesOffset += length
        field_size += length

        for i in range(length):
            points[i + pointsOffset] = field_points[i]

        for i in range(length):
            points[i + pointsOffset][2] = z

        if j > 0:
            for i in range(length):
                values[i + valuesOffset] = field_values[i]

        if coef != 0.0:
            weight = 0.0
            for i in range(length):
                points[i + pointsOffset][2] = z

                pt = points[i + pointsOffset]
                if values[i + pointsOffset] >= 0.0:
                    pt[0] *= r_scale
                    pt[1] *= r
                    values[i + pointsOffset] *= sqrt_r_scale
                else:
                    norm = math.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
                    alpha = norm - 1
                    rnorm = r / norm
                    pt[0] *= rnorm * seg[7]
                    pt[1] *= rnorm

                    enorm = pt[0] * pt[0] + pt[1] * pt[1]
                    if enorm < 1.0:
                        alpha /= math.sqrt(enorm)
                    elif enorm > 4.0:
                        alpha *= 2.0 / math.sqrt(enorm)
                    pt[0] *= 1.0 + alpha
                    pt[1] *= 1.0 + alpha
                weight += math.fabs(values[i + pointsOffset])
                points[i + pointsOffset] = pt

            for i in range(length):
                values[i + pointsOffset] /= weight

    field_points = points
    field_values = values

    # print field_size
    # print field_values[400:410]
    # print field_points[400:410]

    if seg[6] != 0.0:
        tmp_points = field_points.reshape(field_points.size)
        tmp_points = Rotate_Z(field_points[0], field_size, seg[7], 0)
        field_points = tmp_points.reshape((field_points.size / 3, 3))

    TZ_PI = 3.14159265358979323846264338328
    if seg[5] >= 0.2:
        curvature = seg[5]
        if seg[5] > TZ_PI:
            seg[5] = TZ_PI
        Geo3d_Point_Array_Bend(field_points, field_size, seg[4] / curvature)

    if (seg[2] != 0.0) | (seg[3] != 0.0):
        tmp_points = field_points.reshape(field_points.size)
        tmp_points = Rotate_XZ(tmp_points, field_size, seg[2], seg[3], 0)
        field_points = tmp_points.reshape((field_points.size / 3, 3))

    return field_size, field_points, field_values

def Neuroseg_Bottom(seg, pos):
    pos[0] = 0.0
    pos[1] = 0.0
    pos[2] = 0.0

def local_neuroseg_field_shift(locseg, offset):
    pos = np.zeros((2,3))
    # if Neuropos_Reference == 0
    Neuroseg_Bottom(locseg, pos[0])
    Local_Neuroseg_Bottom(locseg, pos[1])

    offset[0] = pos[1][0] - pos[0][0]
    offset[1] = pos[1][1] - pos[0][1]
    offset[2] = pos[1][2] - pos[0][2]

def Geo3d_Point_Array_Translate(points, n, dx, dy, dz):
    for i in range(n):
        points[i][0] += dx
        points[i][1] += dy
        points[i][2] += dz

def Local_Neuroseg_Field_S(locseg, field_size, field_points, field_values):
    field_size, field_points, field_values = Neuroseg_Field_S_Fast(locseg, field_size,field_points,field_values)
    offset = np.zeros(3)
    local_neuroseg_field_shift(locseg, offset)
    Geo3d_Point_Array_Translate(field_points, field_size, offset[0], offset[1], offset[2])
    return field_size, field_points, field_values

def Stack_Point_Sampling(stack, x, y, z):
    # print stack.shape

    width, height, depth = stack.shape
    stack_array = stack.reshape(width * height * depth)

    if (x >= width - 1) | (x <= 0) | (y >= height - 1) | (y <= 0) | (z >= depth - 1) | (z <= 0):
        return np.nan
    else:
        sum = 0.0
        x_low = int(x)
        y_low = int(y)
        z_low = int(z)
        wx_high = x - x_low
        wx_low = 1.0 - wx_high
        wy_high = y - y_low
        wy_low = 1.0 - wy_high
        wz_high = z - z_low
        wz_low = 1.0 - wz_high

        area = width * height
        offset = area * z_low + width * y_low + x_low

        if offset < stack_array.size:
            sum = wx_low * float(stack_array[offset])
            offset += 1
            sum += wx_high * float(stack_array[offset])
            sum *= wy_low * wz_low
            offset += width
            tmp_sum = wx_high * float(stack_array[offset])
            offset -= 1
            tmp_sum += wx_low * float(stack_array[offset])
            sum += tmp_sum * wy_high * wz_low
            offset += area
            tmp_sum = wx_low * float(stack_array[offset])
            offset += 1
            tmp_sum += wx_high * float(stack_array[offset])
            sum += tmp_sum * wy_high * wz_high
            offset -= width
            tmp_sum = wx_high * float(stack_array[offset])
            offset -= 1
            tmp_sum += wx_low * float(stack_array[offset])
            sum += tmp_sum * wy_low * wz_high

        return sum

def Geo3d_Scalar_Field_Stack_Score(field_size, points, values, stack, z_scale, fs_n, fs_options, fs_scores):
    signal = np.zeros(field_size)
    if z_scale == 1.0:
        for i in range(field_size):
            # print i
            signal[i] = Stack_Point_Sampling(stack, points[i][0], points[i][1], points[i][2])

    score = 0.0

    # print field_size
    # print values[600:605]
    # print points[600:605]
    # print signal[600:605]

    if (fs_n != 0):
        for j in range(fs_n):
            # STACK_FIT_DOT
            if fs_options[j] == 0:
                d = 0.0
                for i in range(field_size):
                    p = values[i] * signal[i]
                    if p == p:
                        d += p
                fs_scores[j] = d
                # print "===="
                # print fs_scores[j]
                # for i in range(1500,1505):
                #     print values[i] * signal[i]
            # STACK_FIT_CORRCOEF:
            elif fs_options[j] == 1:
                sum1 = 0
                sum2 = 0
                for i in range(field_size):
                    if values[i] == values[i]:
                        sum1 += values[i]
                for i in range(field_size):
                    if signal[i] == signal[i]:
                        sum2 += signal[i]
                mu1 = sum1 / field_size
                mu2 = sum2 / field_size

                r = v1 = v2 = 0.0
                for i in range(field_size):
                    if (values[i] == values[i]) & (signal[i] == signal[i]):
                        sd1 = values[i] - mu1
                        sd2 = signal[i] - mu2
                        r += sd1 * sd2
                        v1 += sd1 * sd1
                        v2 += sd2 * sd2

                if (v1 == 0) | (v2 == 0):
                    fs_scores[j] = 0
                else:
                    fs_scores[j] = r/math.sqrt(v1*v2)

    score = fs_scores[0]
    return score

def Local_Neuroseg_Score_W(locseg, stack, z_scale, fs_n, fs_options, fs_scores):
    field_size = 0
    field_points = np.zeros(0)
    field_values = np.zeros(0)
    field_size, field_points, field_values = Local_Neuroseg_Field_S(locseg, field_size, field_points, field_values)

    score = 0.0

    # if (ws->mask == NULL)
    score = Geo3d_Scalar_Field_Stack_Score(field_size, field_points, field_values, stack, z_scale, fs_n, fs_options, fs_scores)

    return score

def Local_Neuroseg_Score_R(var, stack, fs_n, fs_options, fs_scores):
    seg = New_Local_Neuroseg()
    for i in range(11):
        Local_Neuroseg_Set_Var(seg, i, var[i])
    z_scale = var[11]

    # print seg
    score = Local_Neuroseg_Score_W(seg, stack, 1.0, fs_n, fs_options, fs_scores)
    return score

def update_variable(vs_nvar, vs_var, vs_var_index, vs_link, index, delta):
    vs_var[vs_var_index[index]] += delta
    if len(vs_link) != 0:
        for i in range(vs_nvar):
            remain = vs_link[i]
            while remain > 0:
                vs_var[remain % 100 - 1] = vs_var[vs_var_index[i]]
                remain /= 100

def perceptor_gradient_partial(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, stack, delta, score, fs_n, fs_options, fs_scores):
    update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, delta)
    right_score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)
    update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, -delta)
    grad = (right_score - score) / delta

    if grad < 0.0:
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, -delta)
        left_score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)
        if left_score < score:
            grad = 0.0
        else:
            grad = (score - left_score) / delta
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, delta)
    elif grad > 0.0:
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, -delta)
        left_score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)
        if left_score > score:
            grad = (score - left_score) / delta
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, delta)
    else:
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, -delta)
        left_score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)
        grad = (score - left_score) / delta
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, index, delta)
    return grad

def Local_Neuroseg_Validate(vs_var, var_min, var_max):
    for i in range(11):
        if vs_var[i] < var_min[i]:
            vs_nvar = var_min[i]
        elif vs_var[i] > var_max[i]:
            vs_var[i] = var_max[i]
        if vs_var[1] < 0.0:
            if (0.5 - vs_var[0]) / vs_var[4] > vs_var[1]:
                vs_var[1] = (0.5 - vs_var[0]) / vs_var[4]

def LINE_SEARCH(vs_nvar, vs_var, vs_var_index, vs_link, cf_varmin, cf_varmax, stack, fs_n, fs_options, fs_scores, weigh, direction, lsw_min_direction, lsw_score, lsw_alpha, lsw_start_grad, lsw_c1, lsw_ro):
    STOP_GRADIENT = 1e-1
    for i in range(vs_nvar):
        if weigh.size != 0:
            direction[i] *= weigh[i]

    # print weigh
    # print direction

    sum = 0.0
    for i in range(vs_nvar):
        sum += direction[i] * direction[i]
    direction_length = math.sqrt(sum)

    improved = True

    if direction_length > lsw_min_direction:
        alpha = 0.0
        org_var = np.zeros(vs_nvar)
        for i in range(vs_nvar):
            org_var[i] = vs_var[vs_var_index[i]]
        start_score = lsw_score
        alpha = lsw_alpha / direction_length

        gd_dot = 0.0
        for i in range(vs_nvar):
            gd_dot += lsw_start_grad[i] * direction[i]
        gd_dot_c1 = gd_dot * lsw_c1

        # print start_score
        # print org_var
        # print gd_dot_c1
        # print alpha

        wolfe1 = 0.0

        while True:
            for i in range(vs_nvar):
                vs_var[vs_var_index[i]] = alpha * direction[i]
                vs_var[vs_var_index[i]] += org_var[i]

            Local_Neuroseg_Validate(vs_var, cf_varmin, cf_varmax)

            # print vs_var

            # Variable_Set_Update_Link(Variable_Set *vs)
            if len(vs_link) != 0:
                for i in range(vs_nvar):
                    remain = vs_link[i]
                    while remain > 0:
                        vs_var[remain % 100 - 1] = vs_var[vs_var_index[i]]
                        remain /= 100

            lsw_score = Local_Neuroseg_Score_R(vs_var, stack, fs_n, fs_options, fs_scores)

            # print lsw_score
            # print alpha, direction_length, alpha * direction_length, STOP_GRADIENT

            alpha *= lsw_ro
            if alpha * direction_length < STOP_GRADIENT:
                for i in range(vs_nvar):
                     vs_var[vs_var_index[i]] = org_var[i]
                # Variable_Set_Update_Link(vs);
                if len(vs_link) != 0:
                    for j in range(vs_nvar):
                        remain = vs_link[j]
                        while remain > 0:
                            vs_var[remain % 100 - 1] = vs_var[vs_var_index[j]]
                            remain /= 100

                lsw_score = start_score
                improved = False
                break

            # print start_score + wolfe1
            # print lsw_score

            if alpha / lsw_ro * gd_dot_c1 > wolfe1:
                wolfe1 = alpha / lsw_ro * gd_dot_c1

            if lsw_score >= start_score + wolfe1:
                break
    else:
        improved = False
    return lsw_score, improved

def Fit_Perceptor(perceptor_vs_nvar, perceptor_min_gradient, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, perceptor_delta, ws_varmin, ws_varmax, perceptor_weight, stack):
    # New_Line_Search_Workspace
    lsw_nvar = perceptor_vs_nvar
    lsw_alpha = 1.0
    lsw_ro = 0.5
    lsw_c1 = 0.01
    lsw_c2 = 0.3
    lsw_grad = np.zeros(perceptor_vs_nvar)
    lsw_start_grad = np.zeros(perceptor_vs_nvar)
    lsw_score = float("-inf")

    # Set_Line_Search_Workspace
    lsw_alpha = 0.2
    lsw_ro = 0.8
    lsw_c1 = 0.01
    lsw_c2 = 0.1
    lsw_min_direction = perceptor_min_gradient

    fs_n = 2
    fs_options = [0, 1]
    fs_scores = [0, 0]
    ws_mask = []

    update_direction = np.zeros(perceptor_vs_nvar)
    perceptor_vs_var[11] = 1.0

    # Perceptor_Gradient(perceptor, stack, lsw->start_grad)
    score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)
    for i in range(perceptor_vs_nvar):
        var_index = perceptor_vs_var_index[i]
        # perceptor_gradient_partial(perceptor->vs, i, stack, perceptor->delta[var_index],score, perceptor->arg, perceptor->s->f);
        lsw_start_grad[i] = perceptor_gradient_partial(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, stack, perceptor_delta[var_index], score, fs_n, fs_options, fs_scores)

    lsw_score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)

    for i in range(perceptor_vs_nvar):
        update_direction[i] = lsw_start_grad[i]

    stop = False
    iter = 0
    succ = True


    while stop == False:
        direction_length = darray_norm(update_direction, perceptor_vs_nvar)

        # print direction_length

        if direction_length < lsw_min_direction:
            succ = False
        else:
            lsw_score, succ = LINE_SEARCH(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, ws_varmin, ws_varmax, stack, fs_n, fs_options, fs_scores, perceptor_weight, update_direction, lsw_min_direction, lsw_score, lsw_alpha, lsw_start_grad, lsw_c1, lsw_ro)

        if succ == False:
            direction_length = darray_norm(lsw_start_grad, perceptor_vs_nvar)
            if direction_length > perceptor_min_gradient:
                for i in range(perceptor_vs_nvar):
                    update_direction[i] = lsw_start_grad[i]
                lsw_score, succ = LINE_SEARCH(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, ws_varmin, ws_varmax, stack, fs_n, fs_options, fs_scores, perceptor_weight, update_direction, lsw_min_direction, lsw_score, lsw_alpha, lsw_start_grad, lsw_c1, lsw_ro)

        if succ == True:
            iter += 1

            # print iter, lsw_score

            if iter >= 500:
                stop = True
            else:
                # Perceptor_Gradient(perceptor, stack, lsw->grad);
                score = Local_Neuroseg_Score_R(perceptor_vs_var, stack, fs_n, fs_options, fs_scores)
                for i in range(perceptor_vs_nvar):
                    var_index = perceptor_vs_var_index[i]
                    # perceptor_gradient_partial(perceptor->vs, i, stack, perceptor->delta[var_index],score, perceptor->arg, perceptor->s->f);
                    lsw_grad[i] = perceptor_gradient_partial(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, stack, perceptor_delta[var_index], score, fs_n, fs_options, fs_scores)

                # Conjugate_Update_Direction_W(perceptor->vs->nvar, lsw->grad, lsw->start_grad, perceptor->weight, update_direction);
                dg = np.zeros(perceptor_vs_nvar)
                for i in range(perceptor_vs_nvar):
                    dg[i] = lsw_grad[i] - lsw_start_grad[i]
                sum1 = 0.0
                sum2 = 0.0
                for i in range(perceptor_vs_nvar):
                    sum1 += dg[i] * lsw_grad[i]
                for i in range(perceptor_vs_nvar):
                    sum2 += lsw_start_grad[i] * lsw_start_grad[i]
                beta = sum1 / sum2

                # if iter == 6:
                #     print lsw_grad

                if beta < 0:
                    beta = 0

                for i in range(perceptor_vs_nvar):
                    update_direction[i] *= beta
                    update_direction[i] += lsw_grad[i]

                # darraycpy(lsw->start_grad, lsw->grad, 0, perceptor->vs->nvar);
                for i in range(perceptor_vs_nvar):
                    lsw_start_grad = lsw_grad

        else:
            stop = True

    score = lsw_score
    return score

def Fit_Local_Neuroseg_W(locseg, stack, z_scale, fws_n, fws_options, fws_scores, fws_pos_adjust):
    NEUROSEG_DEFAULT_H = 11.0
    NEUROSEG_SLICE_FIELD_LENGTH = 277
    LOCAL_NEUROSEG_NPARAM = 11

    weight = np.zeros(LOCAL_NEUROSEG_NPARAM + 1)
    var = np.zeros(LOCAL_NEUROSEG_NPARAM + 1)

    Local_Neuroseg_Param_Array(locseg, z_scale, var)
    # fws_options = np.array([0, 1])

    ws_nvar = 4
    ws_var_index = [0, 2, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ws_varmin = [0.500000, -4.000000, float("-inf"), float("-inf"), 2.000000, 0.000000, float("-inf"), 0.200000, float("-inf"), float("-inf"), float("-inf"), 0.500000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    ws_varmax = [50.000000, 4.000000, float("inf"), float("inf"), 30.000000, 3.141593, float("inf"), 20.000000, float("inf"), float("inf"), float("inf"), 6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    ws_var_link = []
    delta = [0.100000, 0.100000, 0.015000, 0.015000, 1.000000, 0.050000, 0.015000, 0.050000, 0.500000, 0.500000, 0.500000, 0.100000]

    perceptor_vs_nvar = ws_nvar
    perceptor_vs_var_index = ws_var_index
    perceptor_vs_var = var
    perceptor_vs_link = ws_var_link
    perceptor_min_gradient = 1e-3
    perceptor_delta = delta
    # perceptor_arg = ws_sws

    for i in range(perceptor_vs_nvar):
        weight[i] = delta[perceptor_vs_var_index[i]]

    wl = darray_norm(weight, perceptor_vs_nvar)

    for i in range(perceptor_vs_nvar):
        weight[i] /= wl
    perceptor_weight = weight

    # print perceptor_vs_nvar
    # print perceptor_weight

    # print perceptor_vs_var

    Fit_Perceptor(perceptor_vs_nvar, perceptor_min_gradient, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, perceptor_delta, ws_varmin, ws_varmax, perceptor_weight, stack)

    # print perceptor_vs_var

    for i in range(LOCAL_NEUROSEG_NPARAM):
        Local_Neuroseg_Set_Var(locseg, i, perceptor_vs_var[i])

    locseg[2] = Normalize_Radian(locseg[2])
    locseg[3] = Normalize_Radian(locseg[3])

    return Local_Neuroseg_Score_W(locseg, stack, z_scale, fws_n, fws_options, fws_scores)

def Geo3d_Scalar_Field_Center(field_size, field_points, field_values, center):
    center[0] = 0.0
    center[1] = 0.0
    center[2] = 0.0

    for i in range(field_size):
        center[0] += field_points[i][0]
        center[1] += field_points[i][1]
        center[2] += field_points[i][2]

    center[0] /= field_size
    center[1] /= field_size
    center[2] /= field_size

def Geo3d_Scalar_Field_Centroid(field_size, field_points, field_values, centroid):
    weight = 0.0
    centroid[0] = 0.0
    centroid[1] = 0.0
    centroid[2] = 0.0

    for i in range(field_size):
        if field_values[i] == field_values[i]:
            weight += field_values[i]
            centroid[0] += field_points[i][0] * field_values[i]
            centroid[1] += field_points[i][1] * field_values[i]
            centroid[2] += field_points[i][2] * field_values[i]

    if weight == 0.0:
        Geo3d_Scalar_Field_Center(field_size, field_points, field_values, centroid)
    else:
        centroid[0] /= weight
        centroid[1] /= weight
        centroid[2] /= weight

def Local_Neuroseg_Position_Adjust(locseg, stack, z_scale):
    field_size = 0
    field_points = np.zeros(0)
    field_values = np.zeros(0)
    field_size, field_points, field_values = Local_Neuroseg_Field_S(locseg, field_size, field_points, field_values)

    # print field_size

    # Geo3d_Scalar_Field_Stack_Sampling(field, stack, z_scale, field->values);
    # Stack_Points_Sampling(stack, Coordinate_3d_Double_Array(field->points), field->size, signal);
    # Stack_Points_Sampling(tmp_points, field_size, field_values)
    if z_scale == 1.0:
        for i in range(field_size):
            field_values[i] = Stack_Point_Sampling(stack, field_points[i][0], field_points[i][1], field_points[i][2])
    center = np.zeros(3)
    Geo3d_Scalar_Field_Centroid(field_size, field_points, field_values, center)
    # center[2] -= 5

    Set_Neuroseg_Position(locseg, center, 2)

def field_func(x, y):
    t = x * x + y * y
    return (1 - t) * math.exp(-t)

def NEUROSEG_PLANE_FIELD2(points, values, length, x, y, offset):
    points[length + offset][0] = x
    points[length + offset][1] = y
    values[length + offset] = field_func(x, y)
    length += 1
    if x != 0.0:
        points[length + offset][0] = -x
        points[length + offset][1] = y
        values[length + offset] = values[length - 1 + offset]
        length += 1
    if y != 0.0:
        points[length + offset][0] = x
        points[length + offset][1] = -y
        values[length + offset] = values[length - 1 + offset]
        length += 1
        if x != 0.0:
            points[length + offset][0] = -x
            points[length + offset][1] = -y
            values[length + offset] = values[length - 1 + offset]
            length += 1
    return points, values, length

def Neuroseg_Field_Sp(seg, field_size, field_points, field_values):
    NEUROSEG_DEFAULT_H = 11.0
    NEUROSEG_SLICE_FIELD_LENGTH = 277
    NEUROSEG_MIN_CURVATURE = 0.2
    NEUROSEG_MAX_CURVATURE = 3.14159265358979323846264338328

    # locseg[0] = locseg_seg_r1 = r + r
    # locseg[1] = locseg_seg_c = 0.0
    # locseg[2] = locseg_seg_theta = 0.0
    # locseg[3] = locseg_seg_psi = 0.0
    # locseg[4] = locseg_seg_h = 11.0
    # locseg[5] = locseg_seg_curvature = 0.0
    # locseg[6] = locseg_seg_alpha = 0.0
    # locseg[7] = locseg_seg_scale = 1.0
    # locseg_pos = np.zeros(3)
    # locseg[8] = locseg_pos[0] = x
    # locseg[9] = locseg_pos[1] = y
    # locseg[10] = locseg_pos[2] = z

    if (seg[0] == 0) | (seg[7] == 0):
        return field_size, field_points, field_values

    nslice = 11
    if field_size == 0:
        field_size = NEUROSEG_SLICE_FIELD_LENGTH * nslice
        field_points = np.zeros((field_size, 3))
        field_values = np.zeros(field_size)

    z_start = 0.0
    z_step = (seg[4] - 1.0) / (nslice - 1)
    points = field_points
    values = field_values
    # coef = NEUROSEG_COEF(seg) * z_step
    # if seg[4] == 1.0:
    #     coef = seg[1]
    # else:
    #     if seg[1] > (0.5 - seg[0]) / (seg[4] - 1.0):
    #         coef = seg[1]
    #     else:
    #         coef = (0.5 - seg[0]) / (seg[4] - 1.0)
    # coef *= z_step

    coef = NEUROSEG_COEF(seg) * z_step

    r = seg[0]
    z = z_start

    field_size = 0
    offset = 0

    # print seg

    for j in range(nslice):
        y = 0.0
        x = 0.0
        step = 0.2
        start = 0.2
        end = 0.85

        length = 0
        values[length + offset] = field_func(x, y)
        points[length + offset][0] = 0.0
        points[length + offset][1] = 0.0
        length += 1

        for x in np.arange(start, end, step):
            points[length + offset][0] = x
            points[length + offset][1] = y
            values[length + offset] = field_func(x, y)
            length += 1
            points[length + offset][0] = -x
            points[length + offset][1] = y
            values[length + offset] = values[length - 1 + offset]
            length += 1

        x = 0.0
        for y in np.arange(start, end, step):
            points[length + offset][0] = x
            points[length + offset][1] = y
            values[length + offset] = field_func(x, y)
            length += 1
            points[length + offset][0] = x
            points[length + offset][1] = -y
            values[length + offset] = values[length - 1 + offset]
            length += 1

        for y in np.arange(start, 0.45, step):
            for x in np.arange(start, end, step):
                points, values, length = NEUROSEG_PLANE_FIELD2(points, values, length, x, y, offset)

        if y < 0.65:
            y = 0.6
            for x in np.arange(start, 0.65, step):
                points, values, length = NEUROSEG_PLANE_FIELD2(points, values, length, x, y, offset)

        y = 0.8
        for x in np.arange(start, 0.45, step):
            points, values, length = NEUROSEG_PLANE_FIELD2(points, values, length, x, y, offset)

        # print field_size
        # print values[5+offset:10+offset]
        # print points[5+offset:10+offset]

        weight = 0.0
        for i in range(length):
            points[i + offset][0] *= r * seg[7]
            points[i + offset][1] *= r
            points[i + offset][2] = z
            weight += math.fabs(values[i + offset])
        for i in range(length):
            values[i + offset] /= weight
        z += z_step
        r += coef

        # print field_size
        # print values[5+offset:10+offset]
        # print points[5+offset:10+offset]

        # points += length
        # values += length
        offset += length
        field_size += length

    # print field_size
    # print values[500:510]
    # print points[500:510]

    if seg[6] != 0.0:
        tmp_points = field_points.reshape(field_points.size)
        tmp_points = Rotate_Z(points, field_size, seg[6], 0)
        field_points = tmp_points.reshape((field_points.size / 3, 3))

    if seg[5] >= NEUROSEG_MIN_CURVATURE:
        curvature = seg[5]
        if curvature > NEUROSEG_MAX_CURVATURE:
            curvature = NEUROSEG_MAX_CURVATURE

        Geo3d_Point_Array_Bend(points, field_size, seg[4] / curvature)
        # n = field_size
        # c = seg[4] / curvature
        # for i in range(n):
        #     d = c - points[i][1]
        #     if d == 0.0:
        #         points[i][1] = 0.0
        #         points[i][2] = 0.0
        #     else:
        #         points[i][2] /= d
        #         points[i][1] += d - d * math.cos(points[i][2])
        #         points[i][2] = math.sin(points[i][2])
        #     points[i][2] *= d

    if (seg[2] != 0.0) | (seg[3] != 0.0):
        tmp_points = field_points.reshape(field_points.size)
        tmp_points = Rotate_XZ(tmp_points, field_size, seg[2], seg[3], 0)
        field_points = tmp_points.reshape((field_points.size / 3, 3))

    return field_size, points, values

def Local_Neuroseg_Field_Sp(locseg, field_size, field_points, field_values):
    field_size, field_points, field_values = Neuroseg_Field_Sp(locseg, field_size, field_points, field_values)
    #
    # print field_size
    # print field_values[500:505]
    # print field_points[500:505]

    pos = np.zeros(3)
    local_neuroseg_field_shift(locseg, pos)
    Geo3d_Point_Array_Translate(field_points, field_size, pos[0], pos[1], pos[2])
    return field_size, field_points, field_values

def Vector_Angle(x, y):
    TZ_PI = 3.14159265358979323846264338328
    TZ_PI_2 = 1.57079632679489661923132169164
    TZ_2PI = 6.2831853071795862319959269

    if x == 0.0:
        angle = TZ_PI_2
        if y < 0.0:
            angle += TZ_PI
    else:
        angle = math.atan(y / x)
        if x < 0.0:
            angle += TZ_PI
    if angle < 0.0:
        angle += TZ_2PI
    return angle

def Geo3d_Rotate_Orientation(rtheta, rpsi, theta, psi):
    TZ_PI_2 = 1.57079632679489661923132169164
    GEOANGLE_COMPARE_EPS = 0.00001

    sin_theta = math.sin(theta)
    x = sin_theta * math.sin(psi)
    y = -sin_theta * math.cos(psi)
    z = math.sqrt(1.0 - sin_theta * sin_theta)
    coord = np.zeros(3)
    coord = Rotate_XZ(coord, 1, rtheta, rpsi, 0)
    theta = math.acos(z)
    if theta < GEOANGLE_COMPARE_EPS:
        psi = 0.0
    else:
        if y >= 0:
            theta = -theta
            psi = Vector_Angle(x, y) - TZ_PI_2
        else:
            psi = Vector_Angle(x, y) - TZ_PI_2 * 3.0
    return theta, psi

def Local_Neuroseg_Orientation_Search_C(locseg, stack, z_scale, fs_n, fs_options, fs_scores):
    TZ_PI = 3.14159265358979323846264338328
    TZ_PI_2 = 1.57079632679489661923132169164
    TZ_2PI = 6.2831853071795862319959269

    center = np.zeros(3)
    Local_Neuroseg_Center(locseg, center)

    field_size = 0
    field_points = np.zeros(0)
    field_values = np.zeros(0)
    field_size, field_points, field_values = Local_Neuroseg_Field_Sp(locseg, field_size, field_points, field_values)

    # print field_size
    # print field_values[500:505]
    # print field_points[500:505]

    fs_options = np.array([0, 1])
    best_score = Geo3d_Scalar_Field_Stack_Score(field_size, field_points, field_values, stack, z_scale, fs_n, fs_options, fs_scores)
    best_theta = locseg[2]
    best_psi = locseg[3]

    # print best_score, best_theta, best_psi

    tmp_locseg = locseg
    theta_range = TZ_PI * 0.75

    for theta in np.arange(0.1, theta_range+0.0001, 0.2):

        # print theta

        step = 2.0 / locseg[4] / math.sin(theta)

        psi = 0.0
        for psi in np.arange(0.0, TZ_2PI, step):
            tmp_locseg[2] = theta
            tmp_locseg[3] = psi

            # if (theta == 0.1) & (psi == 0.0):
            #     print tmp_locseg[2:4]

            tmp_locseg[2], tmp_locseg[3] = Geo3d_Rotate_Orientation(locseg[2], locseg[3], tmp_locseg[2], tmp_locseg[3])

            # if (theta == 0.1) & (psi == 0.0):
            #     print tmp0_locseg[2:4]

            Set_Neuroseg_Position(tmp_locseg, center, 2)

            # if (theta == 0.1) & (psi == 0.0):
            #     print tmp_locseg[2:4]
            #     print center

            Local_Neuroseg_Field_Sp(tmp_locseg, field_size, field_points, field_values)

            score = Geo3d_Scalar_Field_Stack_Score(field_size, field_points, field_values, stack, z_scale, fs_n, fs_options, fs_scores)

            if score > best_score:
                best_theta = tmp_locseg[2]
                best_psi = tmp_locseg[3]
                best_score = score

            # print theta, psi, score

    locseg[2] = best_theta
    locseg[3] = best_psi

    # print best_theta, best_psi
    # print locseg

    # print center
    # print locseg

    Set_Neuroseg_Position(locseg, center, 2)

    # print locseg

    # print best_score

    return best_score

def Local_Neuroseg_Optimize_W(locseg, stack, z_scale, option):
    fs_n = 2
    fs_options = np.array([0, 1])
    fs_pos_adjust = 1
    fs_scores = np.zeros(2)

    # print locseg
    # stack = stack.reshape(int(width), int(height), int(depth))

    for i in range(fs_pos_adjust):
        Local_Neuroseg_Position_Adjust(locseg, stack, z_scale)

    Local_Neuroseg_Orientation_Search_C(locseg, stack, z_scale, fs_n, fs_options, fs_scores)

    if option <= 1:
        for i in range(1):
            Local_Neuroseg_Position_Adjust(locseg, stack, z_scale)

    score = Fit_Local_Neuroseg_W(locseg, stack, z_scale, fs_n, fs_options, fs_scores, fs_pos_adjust)

    return score

def Spark_Optimize(ball):
    fs_n = 2
    fs_options = np.array([0, 1])
    fs_pos_adjust = 1
    fs_scores = np.zeros(2)

    x = ball[0]
    y = ball[1]
    z = ball[2]
    r = ball[3]

    locseg = New_Local_Neuroseg()
    Set_Local_Neuroseg(locseg, x, y, z, r)

    ball = np.zeros(4)
    Local_Neuroseg_Ball_Bound(locseg, ball)
    stack = Dvid_Access.readStack(ball[0], ball[1], ball[2], ball[3])

    box = np.zeros(6)
    box[0] = math.floor(ball[0] - ball[3])
    box[1] = math.floor(ball[1] - ball[3])
    box[2] = math.floor(ball[2] - ball[3])
    box[3] = math.ceil(ball[0] + ball[3])
    box[4] = math.ceil(ball[1] + ball[3])
    box[5] = math.ceil(ball[2] + ball[3])
    width = box[3] - box[0]
    height = box[4] - box[1]
    depth = box[5] - box[2]
    stack = stack.reshape(int(width), int(height), int(depth))

    score = -1.0

    if stack.size > 0:
        stack_offset = np.zeros(3)
        stack_offset[0] = math.floor(x - r)
        stack_offset[1] = math.floor(y - r)
        stack_offset[2] = math.floor(z - r)

        Dvid_Access.registerToRawStack(stack_offset, locseg)

        Local_Neuroseg_Optimize_W(locseg, stack, 1.0, 0)

        Flip_Local_Neuroseg(locseg)
        Fit_Local_Neuroseg_W(locseg, stack, 1.0, fs_n, fs_options, fs_scores, fs_pos_adjust)
        Flip_Local_Neuroseg(locseg)
        Fit_Local_Neuroseg_W(locseg, stack, 1.0, fs_n, fs_options, fs_scores, fs_pos_adjust)
        Dvid_Access.registerToStack(stack_offset, locseg)

        # print fs_scores
        score = fs_scores[1]

    segOptimize = np.zeros(12)
    segOptimize[0:11] = locseg
    segOptimize[11] = score

    return segOptimize




