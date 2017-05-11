__author__ = 'root'

import numpy as np
import math

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
    apos[0] += pos[8]
    apos[1] += pos[9]
    apos[2] += pos[10]

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
        Neuroseg_Axis_Offset(locseg, axis_offset, apos)
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

    points = field_points
    values = field_values

    if (coef != 0.0) & (nslice > 1):
    # memcpy(field->values + length, field->values, sizeof(double) * length);
        for i in range(length):
            field_values[length + i] = field_values[i]

    for i in range(length):
        points[i][2] = z

        pt = points[i]
        if values[i] >= 0:
            pt[0] *= r_scale
            pt[1] *= r
            values[i] *= sqrt_r_scale
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
        weight += math.fabs(values[i])
        points[i] = pt

    for i in range(length):
        values[i] /= weight

    if (coef != 0.0) & (nslice > 1):
    # memcpy(field->values + length, field->values, sizeof(double) * length);
        for i in range(length):
            field_values[length + i] = field_values[i]

    for j in range(1,nslice):
        z += z_step
        if coef != 0.0:
            r += coef
            r_scale = r * seg[7]
            sqrt_r = math.sqrt(r)
            sqrt_r_scale = sqrt_r * sqrt_sqrt_scale
        pointsOffset = length
        valuesOffset = length
        field_size += length

        for i in range(length):
            points[i + pointsOffset] = field_points[i]

        for i in range(length):
            points[i + pointsOffset][2] = z

        if j > 1:
            for i in range(length):
                values[i + valuesOffset] = field_values[i]
        if coef != 0.0:
            weight = 0.0
            for i in range(length):
                points[i + pointsOffset][2] = z

                pt = points[i]
                if values[i] >= 0:
                    pt[0] *= r_scale
                    pt[1] *= r
                    values[i] *= sqrt_r_scale
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
                weight += math.fabs(values[i])
                points[i] = pt

            for i in range(length):
                values[i] /= weight

    if seg[6] != 0.0:
        field_points[0] = Rotate_Z(field_points[0], field_size, seg[7], 0)

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

    return field_size,field_points,field_values

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
    width = 100
    height = 100
    depth = 100
    stack_array = stack

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

        # stack_array = stack.reshape(width * height * depth)

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
            signal[i] = Stack_Point_Sampling(stack, points[i][0], points[i][1], points[i][2])

    score = 0.0


    if (fs_n != 0):
        for j in range(fs_n):
            # STACK_FIT_DOT
            if fs_options[j] == 0:
                d = 0.0
                for i in range(field_size):
                    p = values[i] * signal[i]
                    if p!= 0:
                        d += p
                fs_scores[j] = d
            # STACK_FIT_CORRCOEF:
            elif fs_options[j] == 1:
                sum1 = 0
                sum2 = 0
                for i in range(field_size):
                    sum1 += values[i]
                for i in range(field_size):
                    sum2 += signal[i]
                mu1 = sum1 / field_size
                mu2 = sum2 / field_size

                r = v1 = v2 = 0.0
                for i in range(field_size):
                    if (values[i] != 0) | (signal[i] != 0):
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

    # print field_size
    # print field_values[0:5]
    # print field_points[0:5]

    score = 0.0

    # if (ws->mask == NULL)
    score = Geo3d_Scalar_Field_Stack_Score(field_size, field_points, field_values, stack, z_scale, fs_n, fs_options, fs_scores)

    return score

def Fit_Local_Neuroseg_W(locseg, stack, z_scale, fws_n, fws_options, fws_scores, fws_pos_adjust):
    NEUROSEG_DEFAULT_H = 11.0
    NEUROSEG_SLICE_FIELD_LENGTH = 277
    LOCAL_NEUROSEG_NPARAM = 11

    weight = np.zeros(LOCAL_NEUROSEG_NPARAM + 1)
    var = np.zeros(LOCAL_NEUROSEG_NPARAM + 1)

    Local_Neuroseg_Param_Array(locseg, z_scale, var)

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
        weight /= wl
    perceptor_weight = weight

    # Fit_Perceptor(perceptor, stack)

    for i in range(LOCAL_NEUROSEG_NPARAM):
        Local_Neuroseg_Set_Var(locseg, i, perceptor_vs_var[i])

    locseg[2] = Normalize_Radian(locseg[2])
    locseg[3] = Normalize_Radian(locseg[3])

    locseg = [11.008722, -0.000000, 4.867409, 4.645343, 11.000000, 0.000000, 0.000000, 1.034885, 7.055716, 47.775794, 61.976532]
    fws_n = 3
    fws_options = np.array([0, 1, 1])
    fws_scores = np.zeros(3)

    return Local_Neuroseg_Score_W(locseg, stack, z_scale, fws_n, fws_options, fws_scores)
