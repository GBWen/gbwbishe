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

# file = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/dist.txt')
# try:
#     sizestr = file.readline()
#     sizearr = sizestr.split(' ')
#     size = []
#     for i in range(3):
#         size.append(int(sizearr[i]))
#     area = size[0] * size[1] * size[2]
#     row = file.readline()
#     words = row.split(' ')
#     stack = []
#     for i in range(area):
#         stack.append(int(words[i]))
# finally:
#      file.close()

file = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/array.txt')
try:
    sizestr = file.readline()
    sizearr = sizestr.split(' ')
    size = []
    for i in range(3):
        size.append(int(sizearr[i]))
    area = size[0] * size[1] * size[2]
    row = file.readline()
    words = row.split(' ')
    array = []
    for i in range(area):
        array.append(int(words[i]))
finally:
     file.close()

file = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/array_out.txt')
try:
    sizestr = file.readline()
    sizearr = sizestr.split(' ')
    size = []
    for i in range(3):
        size.append(int(sizearr[i]))
    area = size[0] * size[1] * size[2]
    row = file.readline()
    words = row.split(' ')
    array_out = []
    for i in range(area):
        array_out.append(int(words[i]))
finally:
     file.close()

width = size[0]
height = size[1]
depth = size[2]
stack_out = np.ones((width, height, depth))
boundary_size = area
if width < depth | height < depth:
    if width < height:
        boundary_size = height * depth
    elif width > height:
        boundary_size = width * depth

ncon = 13
neighbor = np.zeros(ncon)
mask = np.zeros(ncon)
boundary = np.zeros(boundary_size)

# print stack[0:1000]

# array_out = stack_out.reshape([area])
array = np.array(array)
array_out = np.array(array_out)

# init_scan_array(stack->width, stack->height, neighbor);

planeOffset = width * height
neighbor[0] = 1                          # x-directon
neighbor[1] = width                      # y-direction
neighbor[2] = planeOffset                # z-direction
neighbor[3] = width + 1                  # x-y diagonal
neighbor[4] = width - 1                  # x-y counterdiagonal
neighbor[5] = planeOffset + 1            # x-z diagonal
neighbor[6] = planeOffset - 1            # x-z counter diaagonal
neighbor[7] = planeOffset + width        # y-z diagonal
neighbor[8] = planeOffset - width        # y-z counter diaagonal
neighbor[9] = planeOffset + width + 1    # x-y-z diagonal
neighbor[10] = planeOffset + width - 1   # x-y-z diagonal -x
neighbor[11] = planeOffset - width + 1   # x-y-z diagonal -y
neighbor[12] = planeOffset - width - 1   # x-y-z diagonal -x -y

scan_mask = np.array([
  # 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5
  [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],   # 0
  [1,  1,  1,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0],   # 1
  [0,  1,  1,  0,  1,  0,  1,  1,  0,  0,  1,  0,  0],   # 2
  [1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],   # 3
  [0,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1],   # 4
  [1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # 5
  [0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],   # 6
  [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # 7
  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # 8
  [1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0],   # 9  (1|2)
  [1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  1],   # 10 (3|4)
  [1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],   # 11 (5|6)
  [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # 12 (7|8)
  [1,  1,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0],   # 13 (1|3)
  [0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1],   # 14 (2|4)
  [1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # 15 (5|7)
  [0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],   # 16 (6|8)
  [1,  1,  1,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0],   # 17 (1|5)
  [0,  1,  1,  0,  1,  0,  1,  1,  0,  0,  1,  0,  0],   # 18 (2|6)
  [1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],   # 19 (3|7)
  [0,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1],   # 20 (4|8)
  [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],   # 21 (1|2|3|4)
  [1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],   # 22 (5|6|7|8)
  [1,  1,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0],   # 23 (1|3|5|7)
  [0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1],   # 24 (2|4|6|8)
  [1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0],   # 25 (1|2|5|6)
  [1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  1]    # 26 (3|4|7|8)
  # 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5
])


# for depth 1
# 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5
scan_mask_depth = np.array([1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0])

# for height 1
# 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5
scan_mask_height = np.array([1,  0,  1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0])

# for width 1 */
# 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5
scan_mask_width = np.array([0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0])

# nbr = boundary_offset(stack->width, stack->height, stack->depth,	id, boundary);

def ARRAY_CMP(array, array_out, offset, c, nboffset, neighbor, option):
    nboffset = int(offset + neighbor[c])
    # print nboffset
    if nboffset < array.size:
        if array[nboffset] > 0:
            if array[offset] < array[nboffset]:
                array_out[offset] = 0
            elif array[offset] > array[nboffset]:
                array_out[nboffset] = 0
            else:
                array_out[nboffset] = 0

for id in range(1, 27):
    mask = scan_mask[id]
    if depth == 1:
        for i in range(13):
            mask[i] = mask[i] & scan_mask_depth[i]
    if width == 1:
        for i in range(13):
            mask[i] = mask[i] & scan_mask_width[i]
    if height == 1:
        for i in range(13):
            mask[i] = mask[i] & scan_mask_height[i]

    area = width * height
    volume = area * depth
    cwidth = width - 1
    carea = area - width
    cvolume = volume - area
    start = 0
    n = 0

    if id == 1:
        boundary[0] = 0
        n = 1
    elif id == 2:
        boundary[0] = cwidth
        n = 1
    elif id == 3:
        boundary[0] = carea
        n = 1
    elif id == 4:
        boundary[0] = carea + cwidth
        n = 1
    elif id == 5:
        boundary[0] = cvolume
        n = 1
    elif id == 6:
        boundary[0] = cvolume + cwidth
        n = 1
    elif id == 7:
        boundary[0] = cvolume + carea
        n = 1
    elif id == 8:
        boundary[0] = volume - 1
    elif id == 9:
        for j in range(1, cwidth):
            boundary[n] = j
            n = n + 1
    elif id == 10:
        start = carea
        for i in range(1, cwidth):
            boundary[n] = start + i
            n = n + 1
    elif id == 11:
        start = cvolume
        for i in range(1, cwidth):
            boundary[n] = start + i
            n = n + 1
    elif id == 12:
        start = cvolume + carea
        for i in range(1, cwidth):
            boundary[n] = start + i
            n = n + 1
    elif id == 13:
        for i in range(width, carea, width):
            boundary[n] = i
            n = n + 1
    elif id == 14:
        start = cwidth
        for i in range(width, carea, width):
            boundary[n] = start + i
            n = n + 1
    elif id == 15:
        start = cvolume
        for i in range(width, carea, width):
            boundary[n] = start + i
            n = n + 1
    elif id == 16:
        start = cvolume + cwidth
        for i in range(width, carea, width):
            boundary[n] = start + i
            n = n + 1
    elif id == 17:
        for i in range(area, cvolume, area):
            boundary[n] = i
            n = n + 1
    elif id == 18:
        start = cwidth
        for i in range(area, cvolume, area) :
            boundary[n] = start + i
            n = n + 1
    elif id == 19:
        start = carea
        for i in range(area, cvolume, area):
            boundary[n] = start + i
            n = n + 1
    elif id == 20:
        start = carea + cwidth
        for i in range(area, cvolume, area):
            boundary[n] = start + i
            n = n + 1
    elif id == 21:
        for j in range(width, carea, width):
            for i in range(1,cwidth):
                boundary[n] = i + j
                n = n + 1
    elif id == 22:
        start = cvolume
        for j in range(width, carea, width):
            for i in range(1,cwidth):
                boundary[n] = start + i + j
                n = n + 1
    elif id == 23:
        for j in range(area, cvolume, area):
            for i in range(width, carea, width):
                boundary[n] = i + j
                n = n + 1
    elif id == 24:
        start = cwidth
        for j in range(area, cvolume, area):
            for i in range(width, carea, width):
                boundary[n] = start + i + j
                n = n + 1
    elif id == 25:
        for j in range(area, cvolume, area):
            for i in range(1, cwidth):
                boundary[n] = i + j
                n = n + 1
    elif id == 26:
        start = carea
        for j in range(area, cvolume, area):
            for i in range(1, cwidth):
                boundary[n] = start + i + j
                n = n + 1
    nbr = n
    nboffset = 0

    # print array[0:1000]

    for i in range(nbr):
        offset = int(boundary[i])
        if array[offset] == 0:
            array_out[offset] = 0
        for c in range(ncon):
            if mask[c] == 1:
                ARRAY_CMP(array, array_out, offset, c, nboffset, neighbor, 0)

# x: (offset, (array(offset), array_out(offset)))
def array_cmp1(x):
    if x[1][0] == 0:
        return x[0], (x[1][0], 0)
    else:
        return x

def array_cmp2(x):
    keys = []
    for c in range(13):
        nboffset = int(x[0] + neighbor[c])
        if nboffset < area:
            if x[1][0] < array[nboffset]:
                keys.append(x[0])
            else:
                keys.append(nboffset)
    return keys

area = size[0] * size[1] * size[2]
arr = []
for i in range(area):
    arr.append([i, int(array[i])])
arr_out = []
for i in range(area):
    arr_out.append([i, int(array_out[i])])

arr = sc.parallelize(arr).map(lambda x: (x[0], x[1]))
arr_out = sc.parallelize(arr_out).map(lambda x: (x[0], x[1]))
arr_cmp = arr.join(arr_out).map(array_cmp1).flatMap(array_cmp2)
output = arr_cmp.map(lambda x: (x, 1)).distinct()
output = output.collect()

out_zero = np.array(output)
for i in out_zero:
    array_out[i] = 0

cnt = 0
for i in range(area):
    if array_out[i] == 1:
        cnt = cnt + 1

print cnt

file_output = open('/home/gbw/PycharmProjects/DvidSpark/smalldata/seeds.txt', 'w')
file_output.write(str(cnt))
file_output.write(" ")
for i in range(area):
    if array_out[i] == 1:
        file_output.write(str(i))
        file_output.write(" ")
file_output.close()
