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
                                data_out[z, y, x] = data_in[index1][index2]
                                index2 += 1
                    break
                else:
                    index2 = 0
                    for z in range(startz, startz + block_size[2]):
                        for y in range(starty, size[1]):
                            for x in range(startx, size[0]):
                                data_out[z, y, x] = data_in[index1][index2]
                                index2 += 1
                    startx = 0
                    starty = 0
                    startz += block_size[2]
            else:
                index2 = 0
                for z in range(startz, startz + block_size[2]):
                     for y in range(starty, starty + block_size[1]):
                        for x in range(startx, size[0]):
                            data_out[z, y, x] = data_in[index1][index2]
                            index2 += 1
                startx = 0
                starty += block_size[1]
        else:
            index2 = 0
            for z in range(startz, startz + block_size[2]):
                for y in range(starty, starty + block_size[1]):
                    for x in range(startx, startx + block_size[0]):
                        data_out[z, y, x] = data_in[index1][index2]
                        index2 += 1
            startx += block_size[0]
        index1 += 1
    return data_out

block_size = [50, 50, 50]
array = np.array(array)
stack = array.reshape((size[2], size[1], size[0]))
array_out = np.array(array_out)
stack_out = array_out.reshape((size[2], size[1], size[0]))
stack = image2block(stack)
stack_out = image2block(stack_out)

block_area = block_size[0] * block_size[1] * block_size[2]
block_num = area / block_area

ncon = 13
width = block_size[0]
height = block_size[1]
depth = block_size[2]
neighbor = np.zeros(ncon)
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

for i in range(block_num):
    for j in range(block_area):
        if stack[i][j] == 0:
            stack_out[i][j] = 0

def ARRAY_CMP(arr):
    offset = block_size[0] * block_size[1] + block_size[0] + 1
    keys = []
    for k in range(1, block_size[2] - 1):
        for j in range(1, block_size[1] - 1):
            for i in range(1, block_size[0] - 1):
                for c in range(ncon):
                    nboffset = int(offset + neighbor[c])
                    if nboffset < block_area:
                        if arr[offset] < arr[nboffset]:
                            keys.append(offset)
                        else:
                            keys.append(nboffset)
                offset += 1
            offset += 2
        offset += size[0] * 2
    return np.array(keys)

array = sc.parallelize(stack)
array_out = array.map(ARRAY_CMP).collect()

for i in range(block_num):
    for j in range(array_out[i].size):
        stack_out[i][array_out[i][j]] = 0

output = block2image(stack_out).reshape(area)

for i in range(area):
    if output[i] > 0:
        print i