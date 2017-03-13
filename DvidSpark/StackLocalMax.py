__author__ = 'gbw'

import os
import sys
from libtiff import TIFF
import numpy as np

os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
sys.path.append("/home/gbw/spark-1.6.2/python")

try:
    from pyspark import SparkContext
    sc = SparkContext("local","Simple App")

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

option = 0
tif = TIFF.open("/home/gbw/PycharmProjects/DvidSpark/smalldata/stack.tif", mode='r')
stack = tif.read_image()
i = 0
for image in tif.iter_images():
    if i > 0 :
        image = tif.read_image()
        stack = np.dstack((stack, image))
    i = i + 1

if stack.any() == False :
    print "Dist is NULL"

size = stack.shape
area = size[0] * size[1] * size[2]
width = size[0]
height = size[1]
depth = size[2]

stack_out = np.ones((width, height, depth))

boundary_size = area
if width < depth | height < depth :
    if width < height :
        boundary_size = height * depth
    elif width > height :
        boundary_size = width * depth

ncon = 13
neighbor = np.zeros(ncon)
mask = np.zeros(ncon)
boundary = np.zeros(boundary_size)

array_out = stack_out.reshape([area])

array = stack.reshape([area])

# init_scan_array(stack->width, stack->height, neighbor);

planeOffset = width * size[1]
neighbor[0] = 1                          #/* x-directon */
neighbor[1] = width                      #/* y-direction */
neighbor[2] = planeOffset                #/* z-direction */
neighbor[3] = width + 1                  #/* x-y diagonal */
neighbor[4] = width - 1                  #/* x-y counterdiagonal */
neighbor[5] = planeOffset + 1            #/* x-z diagonal */
neighbor[6] = planeOffset - 1            #/* x-z counter diaagonal */
neighbor[7] = planeOffset + width        #/* y-z diagonal */
neighbor[8] = planeOffset - width        #/* y-z counter diaagonal */
neighbor[9] = planeOffset + width + 1    #/* x-y-z diagonal */
neighbor[10] = planeOffset + width - 1   #/* x-y-z diagonal -x*/
neighbor[11] = planeOffset - width + 1   #/* x-y-z diagonal -y*/
neighbor[12] = planeOffset - width - 1   #/* x-y-z diagonal -x -y*/

scan_mask = np.array([
  #/* 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5 */
  [   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],   #/* 0 */
  [   1,  1,  1,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0 ],   #/* 1 */
  [   0,  1,  1,  0,  1,  0,  1,  1,  0,  0,  1,  0,  0 ],   #/* 2 */
  [   1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0 ],   #/* 3 */
  [   0,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1 ],   #/* 4 */
  [   1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 5 */
  [   0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 6 */
  [   1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 7 */
  [   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 8 */
  [   1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0 ],   #/* 9  (1|2) */
  [   1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  1 ],   #/* 10 (3|4) */
  [   1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 11 (5|6) */
  [   1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 12 (7|8) */
  [   1,  1,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0 ],   #/* 13 (1|3) */
  [   0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1 ],   #/* 14 (2|4) */
  [   1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 15 (5|7) */
  [   0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 16 (6|8) */
  [   1,  1,  1,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0 ],   #/* 17 (1|5) */
  [   0,  1,  1,  0,  1,  0,  1,  1,  0,  0,  1,  0,  0 ],   #/* 18 (2|6) */
  [   1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0 ],   #/* 19 (3|7) */
  [   0,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1 ],   #/* 20 (4|8) */
  [   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],   #/* 21 (1|2|3|4) */
  [   1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0 ],   #/* 22 (5|6|7|8) */
  [   1,  1,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0 ],   #/* 23 (1|3|5|7) */
  [   0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1 ],   #/* 24 (2|4|6|8) */
  [   1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0 ],   #/* 25 (1|2|5|6) */
  [   1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  1 ]    #/* 26 (3|4|7|8) */
  #/* 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5 */
])

# /* for depth 1 */
#/* 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5 */
scan_mask_depth = np.array([   1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0 ])

#/* for height 1 */
# #/* 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5 */
scan_mask_height = np.array([   1,  0,  1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0 ])

#/* for width 1 */
#/* 1-2 1-3 1-5 1-4 2-3 1-6 2-5 1-7 3-5 1-8 2-7 3-6 4-5 */
scan_mask_width = np.array([   0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0 ])

for id in range(26) :
    mask = scan_mask[id+1]
    if depth == 1 :
        for i in range(13) :
            mask[i] = mask[i] & scan_mask_depth[i]
    if width == 1 :
        for i in range(13) :
            mask[i] = mask[i] & scan_mask_width[i]
    if height == 1 :
        for i in range(13) :
            mask[i] = mask[i] & scan_mask_height[i]

    # nbr = boundary_offset(stack->width, stack->height, stack->depth, id, boundary);
    area2 = width * height
    volume = area2 * depth
    cwidth = width - 1
    carea = area - width
    cvolume = volume - area
    start = 0
    n = 0
    if id == 1 :
        boundary[0] = 0
        n = 1
    elif id == 2 :
        boundary[0] = cwidth
        n = 1
    elif id == 3 :
        boundary[0] = carea
        n = 1
    elif id == 4 :
        boundary[0] = carea + cwidth
        n = 1
    elif id == 5 :
        boundary[0] = cvolume
        n = 1
    elif id == 6 :
        boundary[0] = cvolume + cwidth
        n = 1
    elif id == 7 :
        boundary[0] = cvolume + carea
        n = 1
    elif id == 8 :
        boundary[0] = volume - 1
    elif id == 9 :
        for j in range(1, cwidth) :
            boundary[n] = j
            n = n + 1
    elif id == 10 :
        start = carea
        for i in range(1, cwidth) :
            boundary[n] = start + i
            n = n + 1
    elif id == 11 :
        start = cvolume
        for i in range(1, cwidth) :
            boundary[n] = start + i
            n = n + 1
    elif id == 12 :
        start = cvolume + carea
        for i in range(1, cwidth) :
            boundary[n] = start + i
            n = n + 1
    elif id == 13 :
        for i in range(width, carea, width) :
            boundary[n] = i
            n = n + 1
    elif id == 14 :
        start = cwidth
        for i in range(width, carea, width) :
            boundary[n] = start + i
            n = n + 1
    elif id == 15 :
        start = cvolume
        for i in range(width, carea, width) :
            boundary[n] = start + i
            n = n + 1
    elif id == 16 :
        start = cvolume + cwidth
        for i in range(width, carea, width) :
            boundary[n] = start + i
            n = n + 1
    elif id == 17 :
        for i in range(area, volume, area) :
            boundary[n] = i
            n = n + 1
    elif id == 18 :
        start = cwidth
        for i in range(area, volume, area) :
            boundary[n] = start + i
            n = n + 1
    elif id == 19 :
        start = carea
        for i in range(area, volume, area) :
            boundary[n] = start + i
            n = n + 1
    elif id == 20 :
        start = carea + cwidth
        for i in range(area, volume, area) :
            boundary[n] = start + i
            n = n + 1
    elif id == 21 :
        for j in range(width, carea, width) :
            for i in range(1,cwidth) :
                boundary[n] = i + j
                n = n + 1
    elif id == 22 :
        start = cvolume
        for j in range(width, carea, width) :
            for i in range(1,cwidth) :
                boundary[n] = start + i + j
                n = n + 1
    elif id == 23 :
        for j in range(area, cvolume, area) :
            for i in range(width, carea, width) :
                boundary[n] = i + j
                n = n + 1
    elif id == 24 :
        start = cwidth
        for j in range(area, cvolume, area) :
            for i in range(width, carea, width) :
                boundary[n] = start + i + j
                n = n + 1
    elif id == 25 :
        for j in range(area, cvolume, area) :
            for i in range(1, cwidth) :
                boundary[n] = i + j
                n = n + 1
    elif id == 25 :
        start = carea
        for j in range(area, cvolume, area) :
            for i in range(1, cwidth) :
                boundary[n] = start + i + j
                n = n + 1
    nbr = n

    for i in range(0, nbr) :
        offset = int(boundary[i])
        if array[offset] == 0 :
            array_out[offset] = 0
        for c in (0,ncon-1) :
            if mask[c] == 1 :
                # ARRAY_CMP(array, array_out, offset, c, nboffset, neighbor, option);
                nboffset = int(offset + neighbor[c])
                if array[nboffset] > 0 :
                    if array[offset] < array[nboffset] :
                        array_out[offset] = 0
                    elif array[offset] > array[nboffset] :
                        array_out[nboffset] = 0
                    else :
                        array_out[nboffset] = 0
                else :
                    array_out[nboffset] = 0


# finding local maxima for internal voxels
offset = area + width + 1
for k in range(1, depth - 1) :
    for j in range(1, height - 1) :
        for i in range(1, width - 1) :
            if array[offset] == 0 :
                array_out[offset] = 0
            for c in range(0, ncon) :
                nboffset = offset + neighbor[c]
                if array[nboffset] > 0 :
                    if array[offset] < array[nboffset] :
                        array_out[offset] = 0
                    elif array[offset] > array[nboffset] :
                        array_out[nboffset] = 0
                    else :
                        array_out[nboffset] = 0
                else :
                    array_out[nboffset] = 0
            offset = offset + 1
        offset = offset + 2
    offset = offset + width * 2

# paralle
arr_out = sc.parallelize(array_out)
arr = sc.parallelize(array)




# tif = TIFF.open('/home/gbw/PycharmProjects/DvidSpark/stack_out.tif', mode='w')
# tif.write_image(dist)
