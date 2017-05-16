__author__ = 'gbw'

import Local_Neuroseg
import os
import sys
import datetime

def getSeeds(file_url):
    file = open(file_url + "/" + uuid + "_seeds.swc")
    seeds = []
    line = file.readline()
    while len(line) > 0:
        line = file.readline()
        words = line.split(' ')
        if len(words) > 6:
            x = int(words[2])
            y = int(words[3])
            z = int(words[4])
            r = float(words[5])
            seed = (x, y, z, r)
            seeds.append(seed)
    return seeds

def writeOptimizeAns(file_url, scores, time):
    file_output = open(file_url + "/seg_scores.txt", 'w')
    num = len(scores)
    file_output.write("cost time: ")
    file_output.write(str(time))
    file_output.write("\n")
    for i in range(num):
        for j in range(11):
            file_output.write(str(scores[i][1][j]))
            file_output.write(" ")
        file_output.write(str(scores[i][0]))
        file_output.write("\n")

if __name__ == '__main__':
    os.environ['SPARK_HOME']="/home/gbw/spark-1.6.2"
    sys.path.append("/root/gbw/spark-1.6.2/python")
    try:
        from pyspark import SparkContext
        sc = SparkContext("local", "Stack Local Max")

    except ImportError as e:
        print("Can not import Spark Modules", e)
        sys.exit(1)

    begin = datetime.datetime.now()
    # print "begin:", begin

    file_url = "/root/gbw/DvidSpark/seeds"
    uuid = "hatu"
    seeds = getSeeds(file_url)

    # seeds = seeds[0:3]

    seeds = sc.parallelize(seeds)
    scores = seeds.map(Local_Neuroseg.Spark_Optimize)
    scores = scores.map(lambda x: (x[11], x[0:11]))
    # print scores.collect()
    scores = scores.filter(lambda x: x[0] > 0.3)
    scores = scores.sortByKey(ascending=False)
    scores = scores.collect()
    # print scores

    end = datetime.datetime.now()
    # print "begin:", end
    # print "cost time:", end - begin

    writeOptimizeAns(file_url, scores, end - begin)
