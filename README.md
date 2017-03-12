# DvidSpark
Neutu并行实现

##3.1

###具体实现方法
neutu的工程下文件很多,全部实现怕来不及.
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/2017-03-12%2019:28:23%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)
想法是保留一些neutu原来的程序,我将其中耗时较多的算法函数使用spark并行实现,然后在执行到这些函数的时候调用spark­submit命令运行调用对应的算法函数,在把结果返回给neutu程序.

##3.7

###种子点提取
runComputeSeed (zcommandine.cpp)

####三个主要函数
从DVID读取图片: readDvidStack 
种子点计算: computeSeedPosition 
建swc树,存到文件中: CreateSwc,save

####其中耗时较多的函数,也就是需要主要实现的函数
computeSeedPosition (zneurontracer.cpp) 
extractSeed 提取种子点,返回种子点个数,坐标,权重
->extractSeedOriginal
--->Stack *seeds = Stack_Local_Max(dist, NULL, STACK_LOCMAX_CENTER);
重点先实现Stack_Local_Max(gui工程下被编译成lib,只能看到tz_stack_lib.h,tz_stack_lib.c文件位于位于c文件夹下)

##3.10

###环境搭建

####DVID

#####dvid.sh

``` bash
cd /home/gbw/dvid-v1.1
./launch config-simple.toml
```

####Spark

##### SparkHadoop.sh

``` bash
cd /home/gbw/hadoop-2.7.2/sbin
./start-all.sh
cd /home/gbw/spark-1.6.2/sbin
./start-all.sh
```

####pycharm

#####pycharm.sh

``` bash
cd ~/pycharm-community-4.5.5/bin/
sudo ./pycharm.sh 
```

####编译脚本

#####build.sh

``` bash
cd ~/neutu_flyem_release/NeuTu-flyem_release/neurolabi/shell/
./setup_neutu_j ~/neutu_flyem_release/build2/
```

####测试脚本

#####test.sh

``` bash
cd ~/neutu_flyem_release/build2/Download/neutube/neurolabi/build
./neutu --command --compute_seed ~/neutu_flyem_release/build2/Download/neutube/neurolabi/data/dvid.json --position 100 100 2600 --size 1024 1024 100 -o ~/neutu_flyem_release/build2/Download/neutube/neurolabi/data/outt.swc
```
提取种子点运行结果
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/2017-03-12%2020:05:22%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)



##3.12

###Spark demo

Spark,以及pycharm配置成功

``` python
from pyspark import SparkContext

logFile = "/home/gbw/spark-1.6.2/README.md"
sc = SparkContext("local","Simple App")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print("Lines with a: %i, lines with b: %i"%(numAs, numBs))
```

###输出stack到文件

可以调用c_stack.cpp 中函数write

``` cpp
C_Stack::write("/home/gbw/neutu_flyem_release/dist.tif", dist);
```

###tif图片读写

``` python
tif = TIFF.open("/home/gbw/PycharmProjects/DvidSpark/dist.tif", mode='r')
stack = tif.read_image()
i = 0
for image in tif.iter_images():
    if i > 0 :
        image = tif.read_image()
        stack = np.dstack((stack, image))
    i = i + 1
    
tif = TIFF.open('/home/gbw/PycharmProjects/DvidSpark/dist2.tif', mode='w')
tif.write_image(dist)
```

####存在问题
gbw@gbw-pc:~/PycharmProjects/DvidSpark$ diff stack.tif stack_out.tif 
二进制文件 stack.tif 和 stack_out.tif 不同



