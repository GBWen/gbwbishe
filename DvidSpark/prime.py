# -*- coding:UTF-8 -*-

import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from graphframe import GraphFrame

__author__ = 'gbw'

if __name__ == '__main__':
    os.environ['SPARK_HOME'] = "/home/gbw/spark-1.6.2"
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.3.0-spark1.6-s_2.10 pyspark-shell"
    sys.path.append("/home/gbw/spark-1.6.2/python")

    CONF = (SparkConf()
            .setMaster("local")
            .setAppName("Prime")
            .set("spark.executor.memory", "4g")
            .set("spark.executor.instances", "4"))

    sc = SparkContext(conf=CONF)
    sqlContext = SQLContext(sc)

    localVertices = [(0, 0, False, 0), (1, 1000, False, -1), (2, 1000, False, -1), (3, 1000, False, -1), (4, 1000, False, -1)]
    localEdges = [(0, 1, 10), (2, 0, 20), (1, 2, 22), (1, 0, 10), (0, 2, 20), (2, 1, 22), (1, 4, 14), (4, 1, 14), (2, 3, 32), (3, 2, 32), (3, 1, 13), (1, 3, 13)]
    v = sqlContext.createDataFrame(localVertices, ["id", "cost", "flag", "parent"])
    e = sqlContext.createDataFrame(localEdges, ["src", "dst", "dist"])
    g = GraphFrame(v, e)

    def getMinCost(x, y):
        if x[1] <= y[1]:
            return x
        else:
            return y

    def update(x):
        if x[0] == currentVertexId:
            x[2] = True
        if (x[2] == False) & (x[0] in currentEdgesDst):
            tmpDist = currentEdgesDist[currentEdgesDst.index(x[0])]
            if x[1] > tmpDist:
                x[1] = tmpDist
                x[3] = currentVertexId
        return x

    verticesRDD = g.vertices.rdd.map(list)
    edgesRDD = g.edges.rdd.map(list)
    for i in range(g.vertices.count()):
        # 找到cost最小的点
        currentVertexId = verticesRDD.filter(lambda x: x[2] == False).reduce(getMinCost)[0]
        # 如果新加入的点相邻点flag == False && cost < 其原本cost, 更新其cost, parent
        currentEdges = edgesRDD.filter(lambda x: x[0] == currentVertexId)
        currentEdgesDst = currentEdges.map(lambda x: x[1]).collect()
        currentEdgesDist = currentEdges.map(lambda x: x[2]).collect()
        verticesRDD = verticesRDD.map(update).cache()
    # print verticesRDD.collect()

    # 根据parent, cost连接整个图,得到最小生成树
    Tree = verticesRDD.map(lambda x: (x[3], x[0], x[1])).collect()
    print Tree
