__author__ = 'root'

import Local_Neuroseg
import os
import sys

if __name__ == '__main__':
    os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
    sys.path.append("/home/gbw/spark-1.6.2/python")
    try:
        from pyspark import SparkContext
        sc = SparkContext("local", "Stack Local Max")

    except ImportError as e:
        print("Can not import Spark Modules", e)
        sys.exit(1)

    x = 110
    y = 112
    z = 2635
    r = 1.0
    # seg = sc.parallelize(seg)
    # seg = seg.map(Local_Neuroseg_Orientation_Search_C)
    # seg.collect()

    ball = [(x, y, z, r), (x, y, z, r)]
    ball = sc.parallelize(ball)
    score = ball.map(Local_Neuroseg.Spark_Optimize)
    score = score.collect()

    print score