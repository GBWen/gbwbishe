__author__ = 'gbw'

import os
import sys
from libtiff import TIFF
import numpy as np

# os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
# sys.path.append("/home/gbw/spark-1.6.2/python")
#
# try:
#     from pyspark import SparkContext
#     sc = SparkContext("local","Simple App")
#
# except ImportError as e:
#     print("Can not import Spark Modules", e)
#     sys.exit(1)

tif = TIFF.open("/home/gbw/PycharmProjects/DvidSpark/stack.tif", mode='r')
stack = tif.read_image()
# print stack
i = 0
for image in tif.iter_images():
    if i > 0 :
        image = tif.read_image()
        stack = np.dstack((stack, image))
    i = i + 1

if stack.any() == False :
    print "Dist is NULL"

size = stack.shape
area = size[0]*size[1]*size[2]

stack_out = np.ones((size[0],size[1],size[2]))

boundary_size = area
if size[0] < size[2] | size[1] < size[2] :
    if size[0] < size[1] :
        boundary_size = size[1] * size[2]
    elif size[0] > size[1] :
        boundary_size = size[0] * size[2]

ncon = 13

# tif = TIFF.open('/home/gbw/PycharmProjects/DvidSpark/stack_out.tif', mode='w')
# tif.write_image(dist)
