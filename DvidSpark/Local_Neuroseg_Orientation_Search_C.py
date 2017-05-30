import numpy as np
import math
import os
import sys

__author__ = 'gbw'


def Rotate_XZ(input, n, theta, psi, inverse):
    Ar0 = math.cos(theta)
    Ar1 = math.sin(theta)
    Ar2 = math.cos(psi)
    Ar3 = math.sin(psi)
    offset = 0
    result = np.zeros(3)
    iny = np.zeros(3)
    inz = np.zeros(3)
    iny[0] = input[1]
    iny[1] = input[2]
    iny[2] = input[0]
    inz[0] = input[2]
    inz[1] = input[0]
    inz[2] = input[1]
    output = np.zeros(3)
    if inverse == 0:
        for i in range(n):
            result[2] = Ar1 * iny[offset] + Ar0 * inz[offset]
            result[0] = inz[offset] * Ar1 - iny[offset] * Ar0
            result[1] = input[offset] * Ar3 - result[0] * Ar2
            result[0] = input[offset] * Ar2 + result[0] * Ar3
            for j in range(3):
                output[j + offset] = result[j]
            offset += 3
    else:
        for i in range(n):
            result[0] = Ar2 * input[offset] + Ar3 * iny[offset]
            result[1] = iny[offset] * Ar2 - input[offset] * Ar3
            result[2] = inz[offset] * Ar0 - result[1] * Ar1
            result[1] = inz[offset] * Ar1 + result[1] * Ar0
            for j in range(3):
                output[j + offset] = result[j]
            offset += 3
    return output


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


def Set_Neuroseg_Position(center,  locseg_seg_h, seg_theta, seg_psi, ref, Neuropos_Reference):
    NEUROSEG_BOTTOM = 0
    NEUROSEG_TOP = 1
    NEUROSEG_CENTER = 2
    locseg_pos = center
    axis_offset = 0
    if ref != Neuropos_Reference:
        if Neuropos_Reference == NEUROSEG_BOTTOM :
            if ref == NEUROSEG_TOP:
                axis_offset = -locseg_seg_h + 1.0
            if ref == NEUROSEG_CENTER:
                axis_offset = -(locseg_seg_h - 1.0) / 2.0
        elif Neuropos_Reference == NEUROSEG_TOP:
            if ref == NEUROSEG_BOTTOM:
                axis_offset = locseg_seg_h - 1.0
            if ref == NEUROSEG_CENTER:
                axis_offset = (locseg_seg_h - 1.0) / 2.0
        elif Neuropos_Reference == NEUROSEG_CENTER:
            if ref == NEUROSEG_BOTTOM:
                axis_offset = (locseg_seg_h - 1.0) / 2.0
            if ref == NEUROSEG_TOP:
                axis_offset = -(locseg_seg_h - 1.0) / 2.0

    apos = np.zeros(3)
    apos[0] = 0.0
    apos[1] = 0.0
    apos[2] = axis_offset
    apos = Rotate_XZ(apos, 1, seg_theta, seg_psi, 0)

    locseg_pos[0] += apos[0]
    locseg_pos[1] += apos[1]
    locseg_pos[2] += apos[2]

    return locseg_pos


# def Local_Neuroseg_Field_Sp():

def TZ_PIunc(x, y):
    t = x * x + y * y
    return (1 - t) * math.exp(-t)


def NEUROSEG_PLANE_FIELD(points, values, length, x, y, offset):
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


def Rotate_Z(input, n, alpha, inverse):
    output = input
    if alpha != 0.0:
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        if inverse == 0:
            for i in range(n):
                tmp = input[0] * cos_a - input[1] * sin_a
                output[1] = input[0] *sin_a + input[1] * cos_a
                output[0] = tmp
                output[2] = input[2]
                input += 3
                output += 3
        else:
            for i in range(n):
                tmp = input[0] * cos_a + input[1] * sin_a
                output[1] = -input[0] * sin_a + input[1] * cos_a
                output[0] = tmp
                output[2] = input[2]
                input += 3
                output += 3
    else:
        output[0] = input[0]
        output[1] = input[1]
        output[2] = input[2]
    return output

def Rotate_XZ2(input, n, theta, psi, inverse):
    Ar0 = math.cos(theta)
    Ar1 = math.sin(theta)
    Ar2 = math.cos(psi)
    Ar3 = math.sin(psi)

    offset = 0
    result = np.zeros(3)
    iny = input
    inz = input
    size = len(input)

    output = np.zeros(3)
    if inverse == 0:
        for i in range(n):
            result[2] = Ar1 * iny[(offset + 1) % size] + Ar0 * inz[(offset + 2) % size]
            result[0] = inz[(offset + 2) % size] * Ar1 - iny[(offset + 1) % size] * Ar0
            result[1] = input[offset] * Ar3 - result[0] * Ar2
            result[0] = input[offset] * Ar2 + result[0] * Ar3
            for j in range(3):
                output[j + offset] = result[j]
            offset += 3
    else:
        for i in range(n):
            result[0] = Ar2 * input[offset] + Ar3 * iny[(offset + 1) % size]
            result[1] = iny[(offset + 1) % size] * Ar2 - input[offset] * Ar3
            result[2] = inz[(offset + 2) % size] * Ar0 - result[1] * Ar1
            result[1] = inz[(offset + 2) % size] * Ar1 + result[1] * Ar0
            for j in range(3):
                output[j + offset] = result[j]
            offset += 3
    return output

def field_func(x, y):
    t = x * x + y * y
    return (1 - t) * math.exp(-t)

def Neuroseg_Field_Sp(seg_r1, seg_scale, seg_h, seg_c, seg_alpha, seg_curvature, seg_theta, seg_psi, field_size, field_points, field_values):
    NEUROSEG_DEFAULT_H = 11.0
    NEUROSEG_SLICE_FIELD_LENGTH = 277
    NEUROSEG_MIN_CURVATURE = 0.2
    NEUROSEG_MAX_CURVATURE = 3.14159265358979323846264338328

    if (seg_r1 == 0) | (seg_scale == 0):
        return field_size, field_points, field_values

    nslice = 11
    if field_size == 0:
        field_size = NEUROSEG_SLICE_FIELD_LENGTH * nslice
        field_points = np.zeros((field_size, 3))
        field_values = np.zeros(field_size)

    z_start = 0.0
    z_step = (seg_h - 1.0) / (nslice - 1)
    points = field_points
    values = field_values
    # coef = NEUROSEG_COEF(seg) * z_step
    if seg_h == 1.0:
        coef = seg_c
    else:
        if seg_c > (0.5 - seg_r1) / (seg_h - 1.0):
            coef = seg_c
        else:
            coef = (0.5 - seg_r1) / (seg_h - 1.0)

    coef *= z_step
    r = seg_r1
    z = z_start

    field_size = 0
    offset = 0

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
                points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y, offset)


        if y < 0.65:
            y = 0.6
            for x in np.arange(start, 0.65, step):
                points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y, offset)

        y = 0.8
        for x in np.arange(start, 0.45, step):
            points, values, length = NEUROSEG_PLANE_FIELD(points, values, length, x, y, offset)

        weight = 0.0
        for i in range(length):
            points[i + offset][0] *= r * seg_scale
            points[i + offset][1] *= r
            points[i + offset][2] = z
            weight += math.fabs(values[i])
        for i in range(length):
            values[i] /= weight
        z += z_step
        r += coef
        # points += length
        # values += length
        offset += length
        field_size += length

    if seg_alpha != 0.0:
        points[0] = Rotate_Z(points[0], field_size, seg_alpha, 0)

    if seg_curvature >= NEUROSEG_MIN_CURVATURE:
        curvature = seg_curvature
        if curvature > NEUROSEG_MAX_CURVATURE:
            curvature = NEUROSEG_MAX_CURVATURE

        n = field_size
        c = seg_h / curvature
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

    if (seg_theta != 0.0) | (seg_psi != 0.0):
        points = Rotate_XZ(points, field_size, seg_theta, seg_psi, 0)

    return field_size, points, values


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


def Geo3d_Point_Array_Translate(points, n ,pos):
    for i in range(n):
        points[i][0] += pos[0]
        points[i][1] += pos[1]
        points[i][2] += pos[2]
    return points


def Stack_Point_Sampling(x, y, z):
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


def Geo3d_Scalar_Field_Stack_Score(field_size, points, values, z_scale):
    fs_n = 1
    fs_options = np.array([1])
    fs_scores = np.zeros(1)

    signal = np.zeros(field_size)
    if z_scale == 1.0:
        for i in range(field_size):
            signal[i] = Stack_Point_Sampling(points[i][0], points[i][1], points[i][2])

    # print signal[10:15]

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


def Local_Neuroseg_Orientation_Search_C(x):
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

    # tmp_locseg = New_Local_Neuroseg()
    tmp_locseg_seg_r1 = 1.0
    tmp_locseg_seg_c = 0.0
    tmp_locseg_seg_theta = 0.0
    tmp_locseg_seg_psi = 0.0
    tmp_locseg_seg_h = 11.0
    tmp_locseg_seg_curvature = 0.0
    tmp_locseg_seg_alpha = 0.0
    tmp_locseg_seg_scale = 1.0
    tmp_locseg_pos = np.zeros(3)
    tmp_locseg_pos[0] = 0.0
    tmp_locseg_pos[1] = 0.0
    tmp_locseg_pos[2] = 0.0

    center = np.zeros(3)
    # Local_Neuroseg_Center(locseg, center)
    # Local_Neuroseg_Axis_Position(locseg, pos, (locseg->seg.h - 1.0) / 2.0);



    # field_size = 0
    # field_points = np.zeros((field_size, 3))
    # field_values = np.zeros(field_size)
    #
    # field_size, points, values = Neuroseg_Field_Sp(locseg_seg_r1, locseg_seg_scale, locseg_seg_h, locseg_seg_c, locseg_seg_alpha, locseg_seg_curvature, locseg_seg_theta, locseg_seg_psi, field_size, field_points, field_values)
    # pos = np.zeros(3)
    # pos = local_neuroseg_field_shift(locseg_pos, pos)
    # points = Geo3d_Point_Array_Translate(points, field_size, pos)
    #
    # best_score = Geo3d_Scalar_Field_Stack_Score(field_size, points, values, z_scale)
    # best_theta = locseg_seg_theta
    # best_psi = locseg_seg_psi

    # print best_score

    # center = np.zeros(3)
    # Neuropos_Reference = 0
    #
    # NEUROSEG_BOTTOM = 0
    # NEUROSEG_TOP = 1
    # NEUROSEG_CENTER = 2
    # TZ_PI = 3.14159265358979323846264338328
    # TZ_PI_2 = 1.57079632679489661923132169164
    # TZ_2PI = 6.2831853071795862319959269
    # NEUROSEG_DEFAULT_H = 11.0
    #
    # theta_range = TZ_PI * 0.75
    # theta = 0.1

    # for theta in range(theta_range, 0.2):
    #     step = 2.0 / locseg_seg_h / math.sin(theta)
    #     psi = 0.0
    #     for psi in range(TZ_2PI, step):
    #         tmp_locseg_seg_theta = theta
    #         tmp_locseg_seg_psi = psi
    #
    #         tmp_locseg_seg_theta, tmp_locseg_seg_psi = Geo3d_Rotate_Orientation(locseg_seg_theta, locseg_seg_psi, tmp_locseg_seg_theta, tmp_locseg_seg_psi)
    #         tmp_locseg_pos = Set_Neuroseg_Position(center, locseg_seg_h, locseg_seg_theta, locseg_seg_psi, 2, 0)
    #
    #         # Local_Neuroseg_Field_Sp
    #         field_size, points, values = Neuroseg_Field_Sp(locseg_seg_r1, locseg_seg_scale, locseg_seg_h, locseg_seg_c, locseg_seg_alpha, locseg_seg_curvature, locseg_seg_theta, locseg_seg_psi, \
    #                           field_size, field_points, field_values)
    #         pos = np.zeros(3)
    #         pos = local_neuroseg_field_shift(tmp_locseg_pos, pos)
    #         points = Geo3d_Point_Array_Translate(points, field_size, pos)
    #
    #         # Geo3d_Scalar_Field_Stack_Score(field, stack, z_scale, fs);
    #         Geo3d_Scalar_Field_Stack_Score()

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

    seg = [(locseg_seg_r1, locseg_seg_c, locseg_seg_theta, locseg_seg_psi,\
                    locseg_seg_h, locseg_seg_curvature, locseg_seg_alpha, locseg_seg_scale,\
                    locseg_pos[0], locseg_pos[1], locseg_pos[2], z_scale)]

    stacks = sc.broadcast(stack)
    stack_height = sc.broadcast(stack_height)
    stack_width = sc.broadcast(stack_width)
    stack_depth = sc.broadcast(stack_depth)

    print seg
    seg = sc.parallelize(seg)
    seg = seg.map(Local_Neuroseg_Orientation_Search_C)
    seg.collect()

    # Test

    # for i in range(100):
    #     print Geo3d_Rotate_Orientation(0.0, 0.0, 1.9, 1.9)

    # center =  np.zeros(3)
    # center[0] = 28.007241
    # center[1] = 46.140601
    # center[2] = 65.006341
    # print Set_Neuroseg_Position(center, 11.000000, 1.041593, 0.631891, 2, 0)

    # Geo3d_Scalar_Field *field = Local_Neuroseg_Field_Sp(locseg, NULL, NULL)
    # field = Neuroseg_Field_Sp(&(locseg->seg), field_func, field);
    # double pos[3];
    # local_neuroseg_field_shift(locseg, pos);
    # Geo3d_Point_Array_Translate(field->points, field->size,pos[0], pos[1], pos[2])

    # seg_h = 11.0
    # seg_theta = 0.0
    # seg_psi = 0.0
    # seg_r1 = 1.5
    # seg_scale = 1.0
    # seg_c = 0.0
    # seg_alpha = 0.0
    # seg_curvature = 0.0
    # field_size = 0
    # field_points = np.zeros((field_size, 3))
    # field_values = np.zeros(field_size)
    # field_size, field_points, field_values = Neuroseg_Field_Sp(seg_r1, seg_scale, seg_h, seg_c, seg_alpha, seg_curvature, seg_theta, seg_psi, field_size, field_points, field_values)
    # pos = np.zeros(3)
    # tmp_locseg_pos = np.array([81.713979, 85.058690, 11.670575])
    # print local_neuroseg_field_shift(tmp_locseg_pos, pos)

    # print Geo3d_Scalar_Field_Stack_Score
    # print "666"

