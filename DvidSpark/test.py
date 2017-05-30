import os
import sys

__author__ = 'gbw'

# os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
# sys.path.append("/home/gbw/spark-1.6.2/python")
#
# try:
#     from pyspark import SparkContext
#     sc = SparkContext("local", "Stack Local Max")
#
# except ImportError as e:
#     print("Can not import Spark Modules", e)
#     sys.exit(1)
#
# x = sc.parallelize([(0,4),(1,3),(2,2),(4,1)])
# y = sc.parallelize([(0,3),(1,2),(2,5),(4,6)])
# z = x.join(y)
#
# print x.collect()
# print y.collect()
# print z.collect()

# def writeNeuroseg(file_url, Neurosegs, time):
#     file_output = open(file_url + "/scores.txt", 'w')
#     num = len(Neurosegs)
#     for j in
#         for i in range(num):
#             for j in range(11):
#                 file_output.write(str(Neurosegs[i][1][j]))
#                 file_output.write(" ")
#             file_output.write(str(Neurosegs[i][0]))
#             file_output.write("\n")

from PIL import Image

im = Image.open("/home/gbw/neutu_flyem_release/test_0p9.tif")


