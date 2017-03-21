__author__ = 'gbw'


import os
import sys
import numpy as np

os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
sys.path.append("/home/gbw/spark-1.6.2/python")

try:
    from pyspark import SparkContext
    sc = SparkContext("local", "Stack Local Max")

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

file = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/input.txt')
try:
    sizestr = file.readline()
    sizearr = sizestr.split(' ')
    size = []
    for i in range(3):
        size.append(int(sizearr[i]))
    area = int(sizearr[3])
    pad = int(sizearr[4])
    row = file.readline()
    words = row.split(' ')
    input = []
    for i in range(area):
        input.append(int(words[i]))
finally:
     file.close()


def image2block(data_in):
    startx = 0
    starty = 0
    startz = 0
    data_out =[]
    while True:
        if startx + block_size[0] >= size[0]:
            if starty + block_size[1] >= size[1]:
                if startz + block_size[2] >= size[2]:
                    list= []
                    for z in range(startz, size[2]):
                        for y in range(starty, size[1]):
                            for x in range(startx, size[0]):
                                list.append(data_in[z, y, x])
                    data_out.append(np.array(list))
                    break
                else:
                    list= []
                    for z in range(startz, startz + block_size[2]):
                        for y in range(starty, size[1]):
                            for x in range(startx, size[0]):
                                list.append(data_in[z, y, x])
                    data_out.append(np.array(list))
                    startx = 0
                    starty = 0
                    startz += block_size[2]
            else:
                list= []
                for z in range(startz, startz + block_size[2]):
                     for y in range(starty, starty + block_size[1]):
                        for x in range(startx, size[0]):
                            list.append(data_in[z, y, x])
                data_out.append(np.array(list))
                startx = 0
                starty += block_size[1]
        else:
            list= []
            for z in range(startz, startz + block_size[2]):
                for y in range(starty, starty + block_size[1]):
                    for x in range(startx, startx + block_size[0]):
                        list.append(data_in[z, y, x])
            data_out.append(np.array(list))
            startx += block_size[0]
    return data_out


def block2image(data_in):
    data_out = np.zeros((size[2], size[1], size[0]))
    startx = 0
    starty = 0
    startz = 0
    index1 = 0
    while True:
        if startx + block_size[0] >= size[0]:
            if starty + block_size[1] >= size[1]:
                if startz + block_size[2] >= size[2]:
                    index2 = 0
                    for z in range(startz, size[2]):
                        for y in range(starty, size[1]):
                            for x in range(startx, size[0]):
                                data_out[z, y, x] = int(data_in[index1][index2])
                                index2 += 1
                    break
                else:
                    index2 = 0
                    for z in range(startz, startz + block_size[2]):
                        for y in range(starty, size[1]):
                            for x in range(startx, size[0]):
                                data_out[z, y, x] = int(data_in[index1][index2])
                                index2 += 1
                    startx = 0
                    starty = 0
                    startz += block_size[2]
            else:
                index2 = 0
                for z in range(startz, startz + block_size[2]):
                     for y in range(starty, starty + block_size[1]):
                        for x in range(startx, size[0]):
                            data_out[z, y, x] = int(data_in[index1][index2])
                            index2 += 1
                startx = 0
                starty += block_size[1]
        else:
            index2 = 0
            for z in range(startz, startz + block_size[2]):
                for y in range(starty, starty + block_size[1]):
                    for x in range(startx, startx + block_size[0]):
                        data_out[z, y, x] = int(data_in[index1][index2])
                        index2 += 1
            startx += block_size[0]
        index1 += 1
    return data_out


def dt1d_first_m_mu16(d, n, f, v, z):
    q = 1
    for q in range(1, n):
        if f[q-1] != f[q]:
            break

    if q == n:
        return d

    if f[0] == 0:
        v[0] = 0
    else:
        v[0] = q

    k = 0
    z[0] = -FLOAT_INF
    z[1] = +FLOAT_INF

    s = 0.0

    for q in range(int(v[0]+1), n):
        if f[q] == 0:
            s = float(q + v[k]) * 0.5
            # pop seeds
            while s <= z[k]:
                k -= 1
                s = float(q + v[k]) * 0.5
            k += 1
            v[k] = q
            z[k] = s
            z[k+1] = +FLOAT_INF

    k = 0
    for q in range(n):
        if d[q] > 0:
            while z[k+1] < q:
                k += 1
            if f[q] > 0:
                d[q] = abs(q - v[k])
    return d


def dt1d_second_m_mu16(f, n, d, v, z, m, sqr_field):
    q = 0
    for q in range(n):
        if m[q] == 0:
            break

    if q == n - 1:
        return d

    v[0] = q

    k = 0
    z[0] = -FLOAT_INF
    z[1] = +FLOAT_INF

    s = 0.0

    for q in range(int(v[0]) + 1, n):
        if m[q] == 0:
            s = f[q] - f[v[k]]
            if sqr_field == 0:
                s *= f[q] + f[v[k]]
            s = (s / (q - v[k]) + float(q + v[k])) / 2.0
            while s <= z[k]:
                k -= 1
                s = f[q] - f[v[k]]
                if sqr_field == 0:
                    s *= f[q] + f[v[k]]
                s = (s / (q - v[k]) + float(q + v[k])) / 2.0
            k += 1
            v[k] = q
            z[k] = s
            z[k+1] = +FLOAT_INF

    k = 0
    for q in range(n):
        if f[q] > 0:
            while z[k] < q:
                k += 1
            k -= 1
            df = q - v[k]
            if sqr_field == 0:
                df = df * df + f[v[k]] * f[v[k]]
            else:
                df = df * df + f[v[k]]
            if df > UINT16_INF:
                d[q] = UINT16_INF - 1
            else:
                d[q] = df
        else:
            d[q] = 0
    return d


def dt3d_mu16(data):
    sz10 = sz[1] * sz[0]
    if sz[0] > sz[1]:
        len = sz[0]
    else:
        len = sz[1]
    if sz[2] > len:
        len = sz[2]
    len += pad * 2

    f = np.zeros(len)
    d = np.zeros(len)
    m = np.zeros(len)
    v = np.zeros(len)
    z = np.zeros(len + 1)

    for k in range(sz[2]):
        tmp_k = k * sz10
        for j in range(sz[1]):
            tmp_j = j * sz[0]
            f[pad: pad + sz[0]] = data[tmp_k + tmp_j: tmp_k + tmp_j + sz[0]]
            d[pad: pad + sz[0]] = data[tmp_k + tmp_j: tmp_k + tmp_j + sz[0]]
            d = dt1d_first_m_mu16(d, sz[0] + pad * 2, f, v, z)
            data[tmp_k + tmp_j: tmp_k + tmp_j + sz[0]] = d[pad: pad + sz[0]]

    if pad == 1:
        f[0] = 0
        f[sz[1] + 1] = 0
        m[0] = 0
        m[sz[1] + 1] = 0
        d[0] = 0
        d[sz[1] + 1] = 0

    for k in range(sz[2]):
        tmp_k = k * sz10
        for i in range(sz[0]):
            for j in range(pad, sz[1] + pad):
                f[j] = data[tmp_k + (j - pad) * sz[0] + i]
                m[j] = (f[j] == UINT16_INF)
            d = dt1d_second_m_mu16(f, sz[1] + pad * 2, d, v, z, m, 0)
            for j in range(pad, sz[1] + pad):
                data[tmp_k + (j - pad)*sz[0] + i] = d[j]

    if pad == 1:
        f[0] = 0
        f[sz[2] + 1] = 0
        d[0] = 0
        d[sz[2] + 1] = 0

    for j in range(sz[1]):
        tmp_j = j * sz[0]
        for i in range(sz[0]):
            for k in range(pad, sz[2] + pad):
                f[k] = data[(k - pad) * sz10 + tmp_j + i]
                m[k] = (f[k] == UINT16_INF)
            d = dt1d_second_m_mu16(f, sz[2] + pad * 2, d, v, z, m, 1)
            for k in range(pad, sz[2] + pad):
                data[(k - pad) * sz10 + tmp_j + i] = d[k]
    return data

block_size = [50, 50, 50]
array = np.array(input)
data = array.reshape((size[2], size[1], size[0]))
data = image2block(data)
sz = block_size

UINT16_INF = 0xFFFF
FLOAT_INF = 1E20
data = sc.parallelize(data)
data = data.map(dt3d_mu16).collect()
data_out = block2image(data)
output = data_out.reshape(area)

file_output = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/mydist.txt', 'w')
for i in range(area):
    file_output.write(str(int(output[i])))
    file_output.write(" ")
file_output.close()

