__author__ = 'gbw'


import os
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

if __name__ == '__main__':
    begin = datetime.datetime.now()
    print "begin:", begin

    file_url = "/home/gbw/PycharmProjects/DvidSpark/seeds"
    uuid = "253e"
    seeds = getSeeds(file_url)

    # seeds = seeds[0:1000]
    print len(seeds)

    for i in range(len(seeds)):
        x = seeds[i][0]
        y = seeds[i][1]
        z = seeds[i][2]
        r = seeds[i][3]

        # x = 1058
        # y = 329
        # z = 2660
        # r = 1.0
        print i, x, y, z, r

        # os.system('cd /home/gbw/miniconda2/envs/neutu-env2/bin')
        # os.system('pwd')
        command = './../../miniconda2/envs/neutu-env2/bin/neutu --command --trace http:127.0.0.1:8000:253e:grayscale --position '
        command += str(x)
        command += ' '
        command += str(y)
        command += ' '
        command += str(z)
        command += ' --scale '
        command += str(r)
        command += ' -o /home/gbw/neutu_flyem_release/253e/trace_out'
        command += str(i)
        command += '.swc'
        os.system(command)

    end = datetime.datetime.now()
    print "end:", end
    print "cost time:", end - begin