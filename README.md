# DvidSpark
Neutu并行实现

## 3.1

### 具体实现方法
neutu的工程下文件很多,全部实现怕来不及.
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/2017-03-12%2019:28:23%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)
想法是保留一些neutu原来的程序,我将其中耗时较多的算法函数使用spark并行实现,然后在执行到这些函数的时候调用spark­submit命令运行调用对应的算法函数,在把结果返回给neutu程序.

## 3.7

### 种子点提取
runComputeSeed (zcommandine.cpp)

#### 三个主要函数
从DVID读取图片: readDvidStack 
种子点计算: computeSeedPosition 
建swc树,存到文件中: CreateSwc,save

#### 其中耗时较多的函数,也就是需要主要实现的函数
computeSeedPosition (zneurontracer.cpp) 
extractSeed 提取种子点,返回种子点个数,坐标,权重
->extractSeedOriginal
--->Stack *seeds = Stack_Local_Max(dist, NULL, STACK_LOCMAX_CENTER);
重点先实现Stack_Local_Max(gui工程下被编译成lib,只能看到tz_stack_lib.h,tz_stack_lib.c文件位于位于c文件夹下)

## 3.10

### 环境搭建

#### DVID

##### dvid.sh

``` bash
cd /home/gbw/dvid-v1.1
./launch config-simple.toml
```

#### Spark

##### SparkHadoop.sh

``` bash
cd /home/gbw/hadoop-2.7.2/sbin
./start-all.sh
cd /home/gbw/spark-1.6.2/sbin
./start-all.sh
```

#### pycharm

##### pycharm.sh

``` bash
cd ~/pycharm-community-4.5.5/bin/
sudo ./pycharm.sh 
```

#### 编译脚本

##### build.sh

``` bash
cd ~/neutu_flyem_release/NeuTu-flyem_release/neurolabi/shell/
./setup_neutu_j ~/neutu_flyem_release/build2/
```

#### 测试脚本

##### test.sh

``` bash
cd ~/neutu_flyem_release/build2/Download/neutube/neurolabi/build
./neutu --command --compute_seed ~/neutu_flyem_release/build2/Download/neutube/neurolabi/data/dvid.json --position 100 100 2600 --size 1024 1024 100 -o ~/neutu_flyem_release/build2/Download/neutube/neurolabi/data/outt.swc
```
提取种子点运行结果
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/2017-03-12%2020:05:22%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)



## 3.12

### Spark demo

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

### 输出stack到文件

可以调用c_stack.cpp 中函数write

``` cpp
C_Stack::write("/home/gbw/neutu_flyem_release/dist.tif", dist);
```

### tif图片读写

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

#### 存在问题
gbw@gbw-pc:~/PycharmProjects/DvidSpark$ diff stack.tif stack_out.tif 
二进制文件 stack.tif 和 stack_out.tif 不同


## 3.13

### 准备数据
找了一个100*100*100的数据存在repo 09c0 中,便于测试

### StackLocalMax.py
完成StackLocalMax.py部分,感觉大部分不需要做并行啊,就是O(width/height/depth) 时间解决的
耗时最多的部分,也就是说最需要并行化的应该在:
``` cpp
    /* finding local maxima for internal voxels*/			\
    offset = area + stack->width + 1;					\
									\
    for (k=1; k<stack->depth - 1; k++) {				\
      for (j=1; j<stack->height - 1; j++) {				\
		for (i=1; i<stack->width - 1; i++) {				\
		  if (array[offset] == 0) {					\
		    array_out[offset] = 0;					\
		  }								\
		  for (c=0; c<ncon; c++) {					\
		    ARRAY_CMP(array, array_out, offset, c,			\
			      nboffset, neighbor, option);			\
		  }								\
		  offset++;							\
		}								\
		offset += 2;							\
	  }									\
	  offset += stack->width * 2;					\
	  /*PROGRESS_REFRESH;*/						\
	}	
```
回顾了下pyspark的api,明天把这部分并行化了

##3.18

### Stack_Local_Max 基本完成 

和单机程序的结果一模一样,很开心,但是耗时很长,还有待优化
输入array,array_out的初始化结果
输出stack_out应该为1的点,结果保存到seeds.txt中


