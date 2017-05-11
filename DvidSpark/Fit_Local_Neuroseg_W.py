import numpy as np
import math
import os
import sys

__author__ = 'gbw'

def Local_Neuroseg_Param_Array(z_scale, seg_r1, seg_c, seg_theta, seg_psi, seg_h, seg_curvature, seg_alpha, seg_scale, seg_pos):
    var = np.zeros(12)
    var[0] = seg_r1
    var[1] = seg_c
    var[2] = seg_theta
    var[3] = seg_psi
    var[4] = seg_h
    var[5] = seg_curvature
    var[6] = seg_alpha
    var[7] = seg_scale
    var[8] = seg_pos[0]
    var[9] = seg_pos[1]
    var[10] = seg_pos[2]
    var[11] = z_scale
    return var


def Local_Neuroseg_Var(seg_r1, seg_c, seg_theta, seg_psi, seg_h, seg_curvature, seg_alpha, seg_scale, seg_pos):
    var = np.zeros(11)
    var[0] = seg_r1
    var[1] = seg_c
    var[2] = seg_theta
    var[3] = seg_psi
    var[4] = seg_h
    var[5] = seg_curvature
    var[6] = seg_alpha
    var[7] = seg_scale
    var[8] = seg_pos[0]
    var[9] = seg_pos[1]
    var[10] = seg_pos[2]
    return var


def Normalize_Radian(r):
    TZ_PI_2 = 1.57079632679489661923132169164
    TZ_2PI = 6.2831853071795862319959269
    TZ_PI = 3.14159265358979323846264338328
    norm_r = r
    if (r < 0.0) | (r >= TZ_2PI):
        norm_r = r - int((r / TZ_2PI) * 2.0 * TZ_PI) + 0.0
    return norm_r


def field_func(x, y):
    t = x * x + y * y
    return (1 - t) * math.exp(-t)


def NEUROSEG_PLANE_FIELD(points, values, length, x, y):
    points[length][0] = x
    points[length][1] = y
    values[length] = field_func(x, y)
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


def local_neuroseg_field_shift(locseg_pos, offset):
    pos = np.zeros((2, 3))
    pos[0][0] = 0.0
    pos[0][1] = 0.0
    pos[0][2] = 0.0
    # Local_Neuroseg_Bottom(const Local_Neuroseg *locseg, double pos[])
    # Neuropos_Coordinate(locseg->pos, pos, pos + 1, pos + 2);
    pos[1][0] = locseg_pos[0]
    pos[1][1] = locseg_pos[1]
    pos[1][2] = locseg_pos[2]

    offset[0] = pos[1][0] - pos[0][0]
    offset[1] = pos[1][1] - pos[0][1]
    offset[2] = pos[1][2] - pos[0][2]

    return offset


def Geo3d_Point_Array_Translate(points, n, pos):
    for i in range(int(n)):
        points[i][0] += pos[0]
        points[i][1] += pos[1]
        points[i][2] += pos[2]
    return points


def Stack_Point_Sampling(stack, x, y, z):
    width = stack_width.value
    height = stack_height.value
    depth = stack_depth.value
    stack_array = stacks.value

    if (x >= width - 1) | (x <= 0) | (y >= height - 1) | (y <= 0) | (z >= depth - 1) | (z <= 0):
        return 0
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
        offset = area *  z_low + width * y_low + x_low

        # stack_array = stack.reshape(width * height * depth)
        sum = wx_low * float(stack_array[offset + 1])
        sum += wx_high * float(stack_array[offset])
        sum *= wy_low * wz_low
        offset += width
        tmp_sum = wx_high * float(stack_array[offset - 1])
        tmp_sum += wx_low * float(stack_array[offset])
        sum += tmp_sum * wy_high * wz_low
        offset += area
        tmp_sum = wx_low * float(stack_array[offset + 1])
        tmp_sum += wx_high * float(stack_array[offset])
        sum += tmp_sum * wy_high * wz_high
        offset -= width
        tmp_sum = wx_high * float(stack_array[offset - 1])
        tmp_sum += wx_low * float(stack_array[offset])
        sum += tmp_sum * wy_low * wz_high

        return sum


def Geo3d_Scalar_Field_Stack_Score(field_size, points, values, z_scale, stack, fs_n, fs_options, fs_scores):
    field_size = int(field_size)
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

def Stack_Point_Hit_Mask(mask, arr):
    width = stack_width.value
    height = stack_height.value
    depth = stack_depth.value
    array = stacks.value

    x = arr[0]
    y = arr[1]
    z = arr[2]

    x_low = int(x + 0.5)
    y_low = int(y + 0.5)
    z_low = int(z + 0.5)

    if (x_low >= width) | (x_low < 0) | (y_low >= height) | (y_low < 0) | (z_low >= depth) | (z_low < 0):
        return False

    area = width * height
    offset = area * z_low + width * y_low + x_low

    # array = stack.reshape(area * stack_depth)
    if array[offset]:
        return True

    return False


def Geo3d_Scalar_Field_Stack_Score_M(field_size, points, values, z_scale, stack, fs_n, fs_options, fs_scores, mask):
    field_size = int(field_size)
    # print field_size
    signal = np.zeros(field_size)
    if z_scale == 1.0:
        for i in range(field_size):
            # Stack_Points_Sampling_M(stack, mask, field->points[0], field->size, signal);
            if Stack_Point_Hit_Mask(mask, points[0]):
                signal[i] = np.nan
            else:
                signal[i] = Stack_Point_Sampling(stack, points[i][0], points[i][1], points[i][2])

    if fs_n != 0:
        for j in range(fs_n):
            # STACK_FIT_DOT
            if fs_options[j] == 0:
                d = 0.0
                for i in range(field_size):
                    p = values[i] * signal[i]
                    if p!= np.nan:
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
                    if (values[i] != np.nan) | (signal[i] != np.nan):
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

    else:
        # score = darray_dot_nw(field->values, signal, field->size);
        w1 = 0.0
        w2 = 0.0
        nw1 = 0.0
        nw2 = 0.0

        for i in range(field_size):
            if values[i] > 0.0:
                w1 += values[i]
            else:
                w1 -= values[i]
            if signal[i] != np.nan:
                if values[i] > 0.0:
                    nw1 += values[i]
                else:
                    nw1 -= values[i]
        if (nw1 > 0.0) & (nw2 > 0.0):
            w1 /= nw1
            w2 /= nw2

        d = 0.0
        for i in range(field_size):
            if (values[i] != np.nan) | (signal[i] != np.nan):
                if values[i] > 0.0:
                    d += values[i] * signal[i] * w1
                else:
                    d += values[i] * signal[i] * w2
        score = d

    return score

def Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, locseg_pos, z_scale, fs_n, fs_options, fs_scores, ws_mask):
    stack = stacks.value

    NEUROSEG_DEFAULT_H = 11.0
    NEUROSEG_SLICE_FIELD_LENGTH = 277
    # field = Local_Neuroseg_Field_S(locseg, ws->field_func, NULL);
    # field = Neuroseg_Field_S_Fast(&(locseg->seg), field_func, field);


    if (seg_r1 == 0) | (seg_scale == 0):
        field_size = 0
        nslice = NEUROSEG_DEFAULT_H
        if field_size == 0:
            field_size = int(NEUROSEG_SLICE_FIELD_LENGTH * nslice)
            field_points = np.zeros((field_size, 3))
            field_values = np.zeros(field_size)
    else:
        nslice = NEUROSEG_DEFAULT_H
        field_size = int(NEUROSEG_SLICE_FIELD_LENGTH * nslice)
        field_points = np.zeros((field_size, 3))
        field_values = np.zeros(field_size)
        offset = np.zeros(3)
        offset = local_neuroseg_field_shift(locseg_pos, offset)


    z_start = 0.0
    z_step = (seg_h - 1.0) / (nslice - 1)
    points = field_points
    values = field_values

    # double coef = NEUROSEG_COEF(seg) * z_step;
    if seg_h == 1.0:
        coef = seg_c
    else:
        if seg_c > (0.5 - seg_r1) / (seg_h - 1.0):
            coef = seg_c
        else:
            coef = (0.5 - seg_r1) / (seg_h - 1.0)

    r = seg_r1
    z = z_start

    # Neuroseg_Slice_Field(points, values, &length, field_func);
    start = 0.2
    end = 1.65
    y = 0.0
    x = 0.0
    step = 0.2
    offset = 0
    range = (end - 0.05) * (end - 0.05) + 0.1
    length = 0

    values[length] = field_func(x, y)
    points[length][0] = 0.0
    points[length][1] = 0.0
    length += 1

    for x in np.arange(start, end, step):
        points[length][0] = x
        points[length][1] = y
        values[length] = field_func(x, y)
        length += 1
        points[length][0] = -x
        points[length][1] = y
        values[length] = values[length - 1]
        length += 1


    x = 0.0
    for y in np.arange(start, end, step):
        points[length][0] = x
        points[length][1] = y
        values[length] = field_func(x, y)
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

    if end >= 1.8:
        y = 1.8
        for x in np.arange(start, 0.85, step):
            if x * x + y * y < range:
                points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y)

    field_values = values
    field_points = points

    offset = np.zeros(3)
    offset = local_neuroseg_field_shift(locseg_pos, offset)

    field_points = Geo3d_Point_Array_Translate(field_points, field_size, offset)

    score = 0.0
    if len(ws_mask) == 0:
        score = Geo3d_Scalar_Field_Stack_Score(field_size, field_points, field_values, z_scale, stack, fs_n, fs_options, fs_scores)
    else:
        score = Geo3d_Scalar_Field_Stack_Score_M(field_size, field_points, field_values, z_scale, stack, fs_n, fs_options, fs_scores, ws_mask)

    return score


def line_search(vs_nvar, vs_var, vs_varindex, vs_link, param, cf_varmin, cf_varmax, delta, weigh, direction, lsw_min_direction, lsw_score, lsw_alpha, lsw_start_grad, lsw_c1, lsw_ro):
    STOP_GRADIENT = 1e-1
    for i in range(vs_nvar):
        if weigh!= 0:
            direction[i] *= weigh[i]

    sum = 0.0
    for i in range(vs_nvar):
        sum += direction[i]
    direction_length = math.sqrt(sum)

    improved = True
    if direction_length > lsw_min_direction:
        alpha = 0.0
        org_var = np.zeros(vs_nvar)
        for i in range(vs_nvar):
            org_var[i] = vs_var[vs_varindex[i]]
        start_score = lsw_score
        alpha = lsw_alpha / direction_length

        gd_dot = 0.0
        for i in range(vs_nvar):
            gd_dot += lsw_start_grad[i] * direction[i]
        gd_dot_c1 = gd_dot * lsw_c1

        wolfe1 = 0.0

        while True:
            for i in range(vs_nvar):
                vs_var[vs_varindex[i]] = alpha * direction[i]
                vs_var[vs_varindex[i]] += org_var[i]

            # cf->v(vs->var, cf->var_min, cf->var_max, NULL);
            for i in range(11):
                if vs_var[i] < cf_varmin[i]:
                    vs_nvar = cf_varmin[i]
                elif vs_var[i] > cf_varmax[i]:
                    vs_var[i] = cf_varmax[i]
                if vs_var[1] < 0.0:
                    if (0.5 - vs_var[0]) / vs_var[4] > vs_var[1]:
                        vs_var[1] = (0.5 - vs_var[0]) / vs_var[4]

            # Variable_Set_Update_Link(Variable_Set *vs)
            if vs_link[0] != 0:
                for i in range(vs_nvar):
                    remain = vs_link[i]
                    while remain > 0:
                        vs_var[remain % 100 - 1] = vs_var[vs_varindex[i]]
                        remain /= 100

            # lsw->score = cf->f(vs->var, param);
            # Local_Neuroseg_Score_R(const double *var, const void *param)
            param_array = param
            stack = param_array[0]
            ws = param[1]

            seg_pos = np.zeros(3)
            seg_r1 = vs_var[0]
            seg_c = vs_var[1]
            seg_theta = vs_var[2]
            seg_psi = vs_var[3]
            seg_h = vs_var[4]
            seg_curvature = vs_var[5]
            seg_alpha = vs_var[6]
            seg_scale = vs_var[7]
            seg_pos[0] = vs_var[8]
            seg_pos[1] = vs_var[9]
            seg_pos[2] = vs_var[10]
            z_scale = vs_var[11]

            lsw_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, fs_n, fs_options, fs_scores, ws_mask)

            alpha *= lsw_ro
            if alpha * direction_length < STOP_GRADIENT:
                for i in range(vs_nvar):
                     vs_var[vs_varindex[i]] = org_var[i]
                # Variable_Set_Update_Link(vs);
                if vs_link.size == 0:
                    for j in range(vs_nvar):
                        remain = vs_link[i]
                        while remain > 0:
                            vs_var[remain % 100 - 1] = vs_var[vs_varindex[j]]
                            remain /= 100

                lsw_score = start_score
                improved = False
                break

            wolfe1 = 0.0
            if alpha / lsw_ro * gd_dot_c1 > wolfe1:
                wolfe1 = alpha / lsw_ro * gd_dot_c1

            if lsw_score < start_score + wolfe1:
                break

    else:
        improved = False

    return improved


def update_variable(vs_nvar, vs_var, vs_var_index, vs_link, index, delta):
    vs_var[vs_var_index[index]] += delta
    if len(vs_link) != 0:
        for i in range(vs_nvar):
            remain = vs_link[i]
            while remain > 0:
                vs_var[remain % 100 - 1] = vs_var[vs_var_index[i]]
                remain /= 100


def Fit_Perceptor(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index,  perceptor_vs_link, perceptor_s_varmin, perceptor_s_varmax, perceptor_delta, perceptor_weight, perceptor_min_gradient):
    lsw_nvar = perceptor_vs_nvar
    lsw_alpha = 1.0
    lsw_ro = 0.5
    lsw_c1 = 0.01
    lsw_c2 = 0.3
    lsw_grad = np.zeros(perceptor_vs_nvar)
    lsw_start_grad = np.zeros(perceptor_vs_nvar)
    lsw_score = float("-inf")

    lsw_alpha = 0.2
    lsw_ro = 0.8
    lsw_c1 = 0.01
    lsw_c2 = 0.1
    lsw_min_direction = perceptor_min_gradient

    fs_n = 2
    fs_options = [0, 1]
    fs_scores = [0, 0]
    ws_mask = []

    vs_var = perceptor_vs_var
    seg_pos = np.zeros(3)
    seg_r1 = vs_var[0]
    seg_c = vs_var[1]
    seg_theta = vs_var[2]
    seg_psi = vs_var[3]
    seg_h = vs_var[4]
    seg_curvature = vs_var[5]
    seg_alpha = vs_var[6]
    seg_scale = vs_var[7]
    seg_pos[0] = vs_var[8]
    seg_pos[1] = vs_var[9]
    seg_pos[2] = vs_var[10]
    z_scale = vs_var[11]
    score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, \
                                   fs_n, fs_options, fs_scores, ws_mask)

    for i in range(perceptor_vs_nvar):
        var_index = perceptor_vs_var_index[i]

        # lsw_grad[i] = perceptor_gradient_partial(perceptor->vs, i, stack, perceptor->delta[var_index], score, perceptor->arg, perceptor->s->f);
        delta = perceptor_delta[var_index]

        # update_variable(vs, index, delta);
        # print perceptor_vs_var
        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, delta)
        # print perceptor_vs_var

        vs_var = perceptor_vs_var
        seg_pos = np.zeros(3)
        seg_r1 = vs_var[0]
        seg_c = vs_var[1]
        seg_theta = vs_var[2]
        seg_psi = vs_var[3]
        seg_h = vs_var[4]
        seg_curvature = vs_var[5]
        seg_alpha = vs_var[6]
        seg_scale = vs_var[7]
        seg_pos[0] = vs_var[8]
        seg_pos[1] = vs_var[9]
        seg_pos[2] = vs_var[10]
        z_scale = vs_var[11]
        right_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale,\
                                             fs_n, fs_options, fs_scores, ws_mask)

        update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, -delta)

        grad = (right_score - score) / delta

        if grad < 0.0:
            vs_var = perceptor_vs_var
            seg_pos = np.zeros(3)
            seg_r1 = vs_var[0]
            seg_c = vs_var[1]
            seg_theta = vs_var[2]
            seg_psi = vs_var[3]
            seg_h = vs_var[4]
            seg_curvature = vs_var[5]
            seg_alpha = vs_var[6]
            seg_scale = vs_var[7]
            seg_pos[0] = vs_var[8]
            seg_pos[1] = vs_var[9]
            seg_pos[2] = vs_var[10]
            z_scale = vs_var[11]
            left_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, \
                                                fs_n, fs_options, fs_scores, ws_mask)

            if left_score < score:
                grad = 0.0
            else:
                grad = (score - left_score) / delta
            update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, delta)
        elif grad > 0.0:
            update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, -delta)
            vs_var = perceptor_vs_var
            seg_pos = np.zeros(3)
            seg_r1 = vs_var[0]
            seg_c = vs_var[1]
            seg_theta = vs_var[2]
            seg_psi = vs_var[3]
            seg_h = vs_var[4]
            seg_curvature = vs_var[5]
            seg_alpha = vs_var[6]
            seg_scale = vs_var[7]
            seg_pos[0] = vs_var[8]
            seg_pos[1] = vs_var[9]
            seg_pos[2] = vs_var[10]
            z_scale = vs_var[11]
            left_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, \
                                                fs_n, fs_options, fs_scores, ws_mask)

            if left_score > score:
                grad = (score - left_score) / delta
                update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, delta)
        else:
            update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, -delta)
            vs_var = perceptor_vs_var
            seg_pos = np.zeros(3)
            seg_r1 = vs_var[0]
            seg_c = vs_var[1]
            seg_theta = vs_var[2]
            seg_psi = vs_var[3]
            seg_h = vs_var[4]
            seg_curvature = vs_var[5]
            seg_alpha = vs_var[6]
            seg_scale = vs_var[7]
            seg_pos[0] = vs_var[8]
            seg_pos[1] = vs_var[9]
            seg_pos[2] = vs_var[10]
            z_scale = vs_var[11]
            left_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, \
                                                fs_n, fs_options, fs_scores, ws_mask)
            grad = (score - left_score) / delta
            update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, delta)
        lsw_start_grad[i] = grad

    print lsw_start_grad

    # update_direction = np.zeros(perceptor_vs_nvar)
    # stop = False
    # iter = 0
    # succ = True
    #
    # while stop == False:
    #     sum = 0.0
    #     for i in range(perceptor_vs_nvar):
    #         sum += update_direction[i]
    #     direction_length = math.sqrt(sum)
    #
    #     if direction_length < lsw_min_direction:
    #         succ = False
    #     else:
    #         # succ = LINE_SEARCH(perceptor->vs, param_array,
    #         # perceptor->s, perceptor->delta,
    #         # perceptor->weight,
    #         # update_direction, lsw);
    #
    #         succ = line_search(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index,  perceptor_vs_link,param_array, \
    #                            perceptor_s_varmin, perceptor_s_varmax, perceptor_delta, perceptor_weigh, update_direction, lsw)
    #
    #         if succ == True:
    #             iter += 1
    #             if iter >= 500:
    #                 stop = True
    #             else:
    #                 # Perceptor_Gradient(perceptor, stack, lsw->grad);
    #                 vs_var = perceptor_vs_var
    #                 seg_pos = np.zeros(3)
    #                 seg_r1 = vs_var[0]
    #                 seg_c = vs_var[1]
    #                 seg_theta = vs_var[2]
    #                 seg_psi = vs_var[3]
    #                 seg_h = vs_var[4]
    #                 seg_curvature = vs_var[5]
    #                 seg_alpha = vs_var[6]
    #                 seg_scale = vs_var[7]
    #                 seg_pos[0] = vs_var[8]
    #                 seg_pos[1] = vs_var[9]
    #                 seg_pos[2] = vs_var[10]
    #                 z_scale = vs_var[11]
    #                 score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, \ fs_n, fs_options, fs_scores, ws_mask)
    #
    #                 for i in range(perceptor_vs_var):
    #                     var_index = perceptor_vs_var_index[i]
    #                     # lsw_grad[i] = perceptor_gradient_partial(perceptor->vs, i, stack, perceptor->delta[var_index], score, perceptor->arg, perceptor->s->f);
    #                     delta = perceptor_delta[var_index]
    #                     # update_variable(vs, index, delta);
    #                     update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_varindex, perceptor_vs_link, i, delta)
    #
    #                     vs_var = perceptor_vs_var
    #                     seg_pos = np.zeros(3)
    #                     seg_r1 = vs_var[0]
    #                     seg_c = vs_var[1]
    #                     seg_theta = vs_var[2]
    #                     seg_psi = vs_var[3]
    #                     seg_h = vs_var[4]
    #                     seg_curvature = vs_var[5]
    #                     seg_alpha = vs_var[6]
    #                     seg_scale = vs_var[7]
    #                     seg_pos[0] = vs_var[8]
    #                     seg_pos[1] = vs_var[9]
    #                     seg_pos[2] = vs_var[10]
    #                     z_scale = vs_var[11]
    #                     right_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, fs_n, fs_options, fs_scores, ws_mask)
    #
    #                     update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, -delta)
    #
    #                     grad = (right_score - score) / delta
    #                     if grad < 0.0:
    #                         vs_var = perceptor_vs_var
    #                         seg_pos = np.zeros(3)
    #                         seg_r1 = vs_var[0]
    #                         seg_c = vs_var[1]
    #                         seg_theta = vs_var[2]
    #                         seg_psi = vs_var[3]
    #                         seg_h = vs_var[4]
    #                         seg_curvature = vs_var[5]
    #                         seg_alpha = vs_var[6]
    #                         seg_scale = vs_var[7]
    #                         seg_pos[0] = vs_var[8]
    #                         seg_pos[1] = vs_var[9]
    #                         seg_pos[2] = vs_var[10]
    #                         z_scale = vs_var[11]
    #                         left_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, fs_n, fs_options, fs_scores, ws_mask)
    #
    #                         if left_score < score:
    #                             grad = 0.0
    #                         else:
    #                              grad = (score - left_score) / delta
    #                         update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, delta)
    #                     elif grad > 0.0:
    #                         update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, -delta)
    #                         vs_var = perceptor_vs_var
    #                         seg_pos = np.zeros(3)
    #                         seg_r1 = vs_var[0]
    #                         seg_c = vs_var[1]
    #                         seg_theta = vs_var[2]
    #                         seg_psi = vs_var[3]
    #                         seg_h = vs_var[4]
    #                         seg_curvature = vs_var[5]
    #                         seg_alpha = vs_var[6]
    #                         seg_scale = vs_var[7]
    #                         seg_pos[0] = vs_var[8]
    #                         seg_pos[1] = vs_var[9]
    #                         seg_pos[2] = vs_var[10]
    #                         z_scale = vs_var[11]
    #                         left_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, fs_n, fs_options, fs_scores, ws_mask)
    #
    #                         if left_score > score:
    #                             grad = (score - left_score) / delta
    #                         update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, delta)
    #                     else:
    #                         update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index, perceptor_vs_link, i, -delta)
    #                         vs_var = perceptor_vs_var
    #                         seg_pos = np.zeros(3)
    #                         seg_r1 = vs_var[0]
    #                         seg_c = vs_var[1]
    #                         seg_theta = vs_var[2]
    #                         seg_psi = vs_var[3]
    #                         seg_h = vs_var[4]
    #                         seg_curvature = vs_var[5]
    #                         seg_alpha = vs_var[6]
    #                         seg_scale = vs_var[7]
    #                         seg_pos[0] = vs_var[8]
    #                         seg_pos[1] = vs_var[9]
    #                         seg_pos[2] = vs_var[10]
    #                         z_scale = vs_var[11]
    #                         left_score = Local_Neuroseg_Score_W(seg_r1, seg_scale, seg_h, seg_c, seg_pos, 0, z_scale, fs_n, fs_options, fs_scores, ws_mask)
    #                         grad = (score - left_score) / delta
    #                         update_variable(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_varindex, perceptor_vs_link, i, delta)
    #                     lsw_grad[i] = grad
    #
    #                 # Conjugate_Update_Direction(perceptor->vs->nvar, lsw->grad, lsw->start_grad, update_direction);
    #                 dg = np.zeros(perceptor_vs_nvar)
    #                 for i in range(perceptor_vs_nvar):
    #                     dg[i] = lsw_grad[i] - lsw_start_grad[i]
    #                 sum1 = 0.0
    #                 sum2 = 0.0
    #                 for i in range(perceptor_vs_nvar):
    #                     sum1 += dg[i] * lsw_grad[i]
    #                 for i in range(perceptor_vs_nvar):
    #                     sum2 += lsw_start_grad[i] * lsw_start_grad[i]
    #                 beta = sum1 / sum2
    #
    #                 if beta < 0:
    #                     beta = 0
    #
    #                 for i in range(perceptor_vs_nvar):
    #                     update_direction[i] *= beta
    #                     update_direction[i] += lsw_grad[i]
    #
    #                 # darraycpy(lsw->start_grad, lsw->grad, 0, perceptor->vs->nvar);
    #                 for i in range(perceptor_vs_nvar):
    #                     lsw_start_grad = lsw_grad
    #         else:
    #             stop = True

    score = lsw_score


def Fit_Local_Neuroseg_W(x):
    locseg_seg_r1 = x[0]
    locseg_seg_c = x[1]
    locseg_seg_theta = x[2]
    locseg_seg_psi = x[3]
    locseg_seg_h = x[4]
    locseg_seg_curvature = x[5]
    locseg_seg_alpha = x[6]
    locseg_seg_scale = x[7]
    locseg_pos = np.zeros(3)
    locseg_pos[0] = x[8]
    locseg_pos[1] = x[9]
    locseg_pos[2] = x[10]
    z_scale = x[11]

    ws_nvar = 4
    ws_var_index = [0, 2, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ws_varmin = [0.500000, -4.000000, float("-inf"), float("-inf"), 2.000000, 0.000000, float("-inf"), 0.200000, float("-inf"), float("-inf"), float("-inf"), 0.500000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    ws_varmax = [50.000000, 4.000000, float("inf"), float("inf"), 30.000000, 3.141593, float("inf"), 20.000000, float("inf"), float("inf"), float("inf"), 6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    ws_var_link = []
    ws_fs_n = 2
    ws_fs_options = [0, 1]
    # ws->pos_adjust = 1;

    LOCAL_NEUROSEG_NPARAM = 11
    DELTA_R = 0.1
    DELTA_THETA = 0.015
    DELTA_H = 1.0
    DELTA_X = 0.5
    DELTA_Y = 0.5
    DELTA_Z = 0.5
    Delta = [0.100000, 0.100000, 0.015000, 0.015000, 1.000000, 0.050000, 0.015000, 0.050000, 0.500000, 0.500000, 0.500000, 0.100000]

    weight = np.zeros(LOCAL_NEUROSEG_NPARAM + 1)
    var = np.zeros(LOCAL_NEUROSEG_NPARAM + 1)

    var = Local_Neuroseg_Param_Array(z_scale, locseg_seg_r1, locseg_seg_c, locseg_seg_theta, locseg_seg_psi, locseg_seg_h, locseg_seg_curvature, locseg_seg_alpha, locseg_seg_scale, locseg_pos)

    perceptor_vs_nvar = ws_nvar
    perceptor_vs_var_index = ws_var_index
    perceptor_vs_var = var
    perceptor_vs_link = ws_var_link
    perceptor_min_gradient = 1e-3
    # perceptor_arg = ws_sws
    perceptor_delta = Delta

    for i in range(perceptor_vs_nvar):
        weight[i] = Delta[perceptor_vs_var_index[i]]

    result = 0
    for i in range(perceptor_vs_nvar):
        result += weight[i] * weight[i]
    wl = math.sqrt(result)

    for i in range(perceptor_vs_nvar):
        weight[i] /= wl
    perceptor_weight = weight

    # Make_Continuous_Function(Local_Neuroseg_Score_R, Local_Neuroseg_Validate,ws->var_min, ws->var_max);

    perceptor_s_varmin = ws_varmin
    perceptor_s_varmax = ws_varmax

    # Fit_Perceptor(&perceptor, stack);

    Fit_Perceptor(perceptor_vs_nvar, perceptor_vs_var, perceptor_vs_var_index,  perceptor_vs_link, perceptor_s_varmin, perceptor_s_varmax, perceptor_delta, perceptor_weight, perceptor_min_gradient)

    # #for i in range(LOCAL_NEUROSEG_NPARAM):
    #     # Local_Neuroseg_Set_Var(locseg, i, perceptor.vs->var[i]);
    # locseg_seg_r1 = perceptor_vs_var[0]
    # locseg_seg_c = perceptor_vs_var[1]
    # locseg_seg_theta = perceptor_vs_var[2]
    # locseg_seg_psi = perceptor_vs_var[3]
    # locseg_seg_h = perceptor_vs_var[4]
    # locseg_seg_curvature = perceptor_vs_var[5]
    # locseg_seg_alpha = perceptor_vs_var[6]
    # locseg_seg_scale = perceptor_vs_var[7]
    # locseg_seg_pos[0] = perceptor_vs_var[8]
    # locseg_seg_pos[1] = perceptor_vs_var[9]
    # locseg_seg_pos[2] = perceptor_vs_var[10]
    # z_scale = perceptor_vs_var[11]
    #
    # locseg_seg_theta = Normalize_Radian(locseg_seg_theta)
    # locseg_seg_psi = Normalize_Radian(locseg_seg_psi)
    #
    # # Local_Neuroseg_Score_P(locseg, stack, z_scale, fs);
    # return Local_Neuroseg_Score_W()

if __name__ == '__main__':
    os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
    sys.path.append("/home/gbw/spark-1.6.2/python")
    try:
        from pyspark import SparkContext
        sc = SparkContext("local", "Stack Local Max")

    except ImportError as e:
        print("Can not import Spark Modules", e)
        sys.exit(1)

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

    seg = [(locseg_seg_r1, locseg_seg_c, locseg_seg_theta, locseg_seg_psi,\
                    locseg_seg_h, locseg_seg_curvature, locseg_seg_alpha, locseg_seg_scale,\
                    locseg_pos[0], locseg_pos[1], locseg_pos[2], z_scale)]

    stacks = sc.broadcast(stack)
    stack_height = sc.broadcast(stack_height)
    stack_width = sc.broadcast(stack_width)
    stack_depth = sc.broadcast(stack_depth)

    # seg = sc.parallelize(seg)
    # seg = seg.map(Fit_Local_Neuroseg_W)
    # seg.collect()

    pos = [7.055716, 47.775794, 61.976532]
    fs_n = 2
    fs_options = [0, 1]
    fs_scores = [0, 0]
    ws_mask = []
    print Local_Neuroseg_Score_W(11.008722, 1.034885, 11.000000, 0.000000, pos, 1.000000, fs_n, fs_options, fs_scores, ws_mask)

