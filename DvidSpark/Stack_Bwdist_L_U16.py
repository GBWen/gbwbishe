__author__ = 'gbw'

import os
import sys
import numpy as np
import time

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
    sz = []
    for i in range(3):
        sz.append(int(sizearr[i]))
    area = int(sizearr[3])
    pad = int(sizearr[4])
    row = file.readline()
    words = row.split(' ')
    input = []
    for i in range(area):
        input.append(int(words[i]))
finally:
     file.close()

# dt3d_binary_mu16(out_array, sz, !pad)
# dt3d_binary_mu16(data, sz, pad)
# --> dt3d_mu16(data, sz, pad);

sz10 = sz[1] * sz[0]
if sz[0] > sz[1]:
    len = sz[0]
else:
    len = sz[1]
if sz[2] > len:
    length = sz[2]
len += pad * 2

# f = np.zeros(len)
# d = np.zeros(len)
# m = np.zeros(len)
# v = np.zeros(len)
# z = np.zeros(len + 1)

if pad == 1:
    data = []
    line = []
    for i in range(area):
        if i % sz[0] == 0:
            line.append(0)
            line.append(input[i])
        elif i % sz[0] == (sz[0] - 1):
            line.append(0)
            line.append(input[i])
            npline = np.array(line)
            data.append(npline)
            line = []
        else:
            line.append(input[i])
else:
    data = []
    line = []
    for i in range(area):
        if i % sz[0] == (sz[0] - 1):
            line.append(input[i])
            npline = np.array(line)
            data.append(npline)
            line = []
        else:
            line.append(input[i])

# dt1d_first_m_mu16(d, sz[0] + pad * 2, f, v, z);

def dt1d_first_m_mu16(x):
    d = x
    n = sz[0] + pad * 2
    f = x
    v = np.zeros(n)
    z = np.zeros(n + 1)

    for q in range(1, n):
        if f[q-1] != f[q]:
            break

    if q == n - 1:
        return x

    if f[0] == 0:
        v[0] = 0
    else:
        v[0] = q

    k = 0
    z[0] = -1e20
    z[1] = 1e20
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
            z[k+1] = 1e20

    k = 0
    for q in range(0, n):
        if d[q] > 0:
            while z[k+1] < q:
                k += 1
            if f[q] > 0:
                d[q] = abs(q-v[k])
    return d

data = sc.parallelize(data)
data_first = data.map(dt1d_first_m_mu16).collect()
line_size = data_first[0].size
data_first = np.array(data_first)
size = data_first.size
data_first = data_first.reshape(data_first.size)

data_1 = []
for i in range(data_first.size):
    if (i % line_size > 0) & ((i % line_size) < (line_size - 1)):
        data_1.append(data_first[i])

if pad == 1:
    data = []
    line = []
    for k in range(sz[2]):
        tmp_k = k * sz10
        for i in range(sz[0]):
            for j in range(pad, sz[1] + pad):
                if j == pad:
                    line.append(0)
                    line.append(data_1[tmp_k + (j - pad) * sz[0] + i])
                elif j == sz[1] + pad - 1:
                    line.append(0)
                    line.append(data_1[tmp_k + (j - pad) * sz[0] + i])
                    npline = np.array(line)
                    data.append(npline)
                    line = []
                else:
                    line.append(data_1[tmp_k + (j - pad) * sz[0] + i])
else:
    data = []
    line = []
    for k in range(sz[2]):
        tmp_k = k * sz10
        for i in range(sz[0]):
            for j in range(pad, sz[1] + pad):
                if j == sz[1]:
                    line.append(data_1[tmp_k + (j - pad) * sz[0] + i])
                    npline = np.array(line)
                    data.append(npline)
                    line = []
                else:
                    line.append(data_1[tmp_k + (j - pad) * sz[0] + i])

# dt1d_second_m_mu16(f, sz[1] + pad * 2, d, v, z, m, 0);

def dt1d_second_m_mu16(x):
    f = x
    n = sz[1] + pad * 2
    m = np.zeros(n)
    d = np.zeros(n)
    v = np.zeros(n)
    z = np.zeros(n+1)

    for i in range(x.size):
        m[j] = (f[j] == 0xffff)

    for q in range(n):
        if m[q] == 0:
            break

    if q == n:
        return

    v[0] = q

    k = 0
    z[0] = -1e20
    z[1] = 1e20

    s = 0.0

    for q in range(int(v[0]) + 1, n):
        if m[q] == 0:
            s = f[q] - f[v[k]]
            if sqr_field == 0:
                s *= f[q] + f[v[k]]
            s = (s / (q - v[k]) + (float)(q + v[k])) / 2.0
            while s <= z[k]:
                k -= 1
                s = f[q] - f[v[k]]
                if sqr_field == 0:
                    s *= f[q] + f[v[k]]
                s = (s / (q - v[k]) + float(q + v[k])) / 2.0
            k += 1
            v[k] = q
            z[k] = s
            z[k+1] = 1e20

    k = 0
    df = 0
    for q in range(n):
        if f[q] > 0:
            while z[k] < q:
                k += 1
            k -= 1
            df = q-v[k]
            if sqr_field == 0:
                df = df * df + f[v[k]] * f[v[k]]
            else:
                df = df * df + f[v[k]]
            if df > 0xffff:
                d[q] = int(0xffff - 1)
            else:
                d[q] = int(df)
        else:
            d[q] = int(0)
    return d

sqr_field = 0
data = sc.parallelize(data)
data_second = data.map(dt1d_second_m_mu16).collect()
line_size = data_second[0].size
data_second = np.array(data_second)
size = data_second.size
data_second = data_second.reshape(data_second.size)

data_2 = np.zeros(area)
for i in range(area):
    x = i % sz[1]
    y = i / sz[1] % sz[0]
    z = i / (sz[1] * sz[0])
    data_2[y + x * sz[0] + z * sz[0] * sz[1]] = data_second[i]

if pad == 1:
    data = []
    line = []
    for j in range(sz[1]):
        tmp_j = j * sz[0]
        for i in range(sz[0]):
            for k in range(pad, sz[2] + pad):
                if k == pad:
                    line.append(0)
                    line.append(data_2[(k - pad)*sz10 + tmp_j + i])
                elif k == sz[1] + pad - 1:
                    line.append(0)
                    line.append(data_2[(k - pad)*sz10 + tmp_j + i])
                    npline = np.array(line)
                    data.append(npline)
                    line = []
                else:
                    line.append(data_2[(k - pad)*sz10 + tmp_j + i])
else:
    data = []
    line = []
    for j in range(sz[1]):
        tmp_j = j * sz[0]
        for i in range(sz[0]):
            for k in range(pad, sz[2] + pad):
                if k == sz[1] + pad - 1:
                    line.append(data_2[(k - pad)*sz10 + tmp_j + i])
                    npline = np.array(line)
                    data.append(npline)
                    line = []
                else:
                    line.append(data_2[(k - pad)*sz10 + tmp_j + i])



sqr_field = 1
data = sc.parallelize(data)
data_second2 = data.map(dt1d_second_m_mu16).collect()
line_size = data_second2[0].size
data_second2 = np.array(data_second2)
size = data_second2.size
data_second2 = data_second2.reshape(data_second2.size)

output = np.zeros(area)
for i in range(area):
    x = i % sz[2]
    y = i / sz[2] % sz[0]
    z = i / (sz[2] * sz[0])
    output[x * sz[0] * sz[1] + y + z * sz[0]] = data_second2[i]

file_output = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/mydist.txt', 'w')
for i in range(area):
    file_output.write(str(output[i]))
    file_output.write(" ")
file_output.close()





