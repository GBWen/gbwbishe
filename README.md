#### Table of contents
- [DvidSpark](#dvidspark)
    - [3.1](#31)
    - [3.7](#37)
    - [3.10](#310)
        - [环境搭建](#)
            - [DVID](#dvid)
            - [Spark](#spark)
            - [pycharm](#pycharm)
            - [编译脚本](#)
            - [测试脚本](#)
    - [3.12](#312)
    - [3.13](#313)
    - [3.18](#318)
    - [3.19](#319)
    - [3.21](#321)
    - [3.22](#322)
    - [3.23](#323)
    - [4.10](#410)
    - [4.13](#413)
        - [总体设计](#)
            - [3.1 架构设计](#31-)
            - [3.2 处理逻辑](#32-)
            - [3.3 系统模块](#33-)
            - [3.4 算法的Spark并行化分析](#34-spark)
                - [3.4.1 数据并行化](#341-)
                - [3.4.2 任务并行化](#342-)
# DvidSpark
Neutu并行实现

## 3.1

neutu的工程下文件很多,全部实现怕来不及.
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/2017-03-12%2019:28:23%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)
想法是保留一些neutu原来的程序,我将其中耗时较多的算法函数使用spark并行实现,然后在执行到这些函数的时候调用spark­submit命令运行调用对应的算法函数,在把结果返回给neutu程序.

## 3.7

种子点提取
runComputeSeed (zcommandine.cpp)

三个主要函数
从DVID读取图片: readDvidStack 
种子点计算: computeSeedPosition 
建swc树,存到文件中: CreateSwc,save

其中耗时较多的函数,也就是需要主要实现的函数
computeSeedPosition (zneurontracer.cpp) 
extractSeed 提取种子点,返回种子点个数,坐标,权重
->extractSeedOriginal
--->Stack *seeds = Stack_Local_Max(dist, NULL, STACK_LOCMAX_CENTER);
重点先实现Stack_Local_Max(gui工程下被编译成lib,只能看到tz_stack_lib.h,tz_stack_lib.c文件位于位于c文件夹下)

## 3.10

### 环境搭建

#### DVID

dvid.sh

``` bash
cd /home/gbw/dvid-v1.1
./launch config-simple.toml
```

#### Spark

SparkHadoop.sh

``` bash
cd /home/gbw/hadoop-2.7.2/sbin
./start-all.sh
cd /home/gbw/spark-1.6.2/sbin
./start-all.sh
```

#### pycharm

pycharm.sh

``` bash
cd ~/pycharm-community-4.5.5/bin/
sudo ./pycharm.sh 
```

#### 编译脚本

build.sh

``` bash
cd ~/neutu_flyem_release/NeuTu-flyem_release/neurolabi/shell/
./setup_neutu_j ~/neutu_flyem_release/build2/
```

#### 测试脚本

test.sh

``` bash
cd ~/neutu_flyem_release/build2/Download/neutube/neurolabi/build
./neutu --command --compute_seed ~/neutu_flyem_release/build2/Download/neutube/neurolabi/data/dvid.json --position 100 100 2600 --size 1024 1024 100 -o ~/neutu_flyem_release/build2/Download/neutube/neurolabi/data/outt.swc
```
提取种子点运行结果
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/2017-03-12%2020:05:22%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)



## 3.12

Spark demo

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

输出stack到文件

可以调用c_stack.cpp 中函数write

``` cpp
C_Stack::write("/home/gbw/neutu_flyem_release/dist.tif", dist);
```

tif图片读写

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

存在问题
gbw@gbw-pc:~/PycharmProjects/DvidSpark$ diff stack.tif stack_out.tif 
二进制文件 stack.tif 和 stack_out.tif 不同


## 3.13

准备数据
找了一个100*100*100的数据存在repo 09c0 中,便于测试

StackLocalMax.py
完成StackLocalMax.py部分,感觉大部分不需要做并行啊,就是O(width/height/depth) 时间解决的
耗时最多的部分,也就是说最需要并行化的应该在:
``` cpp
    /* finding local maxima for internal voxels*/     \
    offset = area + stack->width + 1;         \
                  \
    for (k=1; k<stack->depth - 1; k++) {        \
      for (j=1; j<stack->height - 1; j++) {       \
    for (i=1; i<stack->width - 1; i++) {        \
      if (array[offset] == 0) {         \
        array_out[offset] = 0;          \
      }               \
      for (c=0; c<ncon; c++) {          \
        ARRAY_CMP(array, array_out, offset, c,      \
            nboffset, neighbor, option);      \
      }               \
      offset++;             \
    }               \
    offset += 2;              \
    }                 \
    offset += stack->width * 2;         \
    /*PROGRESS_REFRESH;*/           \
  } 
```
回顾了下pyspark的api,明天把这部分并行化了

## 3.18

Stack_Local_Max 基本完成
和单机程序的结果一模一样,很开心,但是耗时很长,还有待优化
输入array,array_out的初始化结果
输出stack_out应该为1的点,结果保存到seeds.txt中


## 3.19
这个函数主要是做三次计算
dt1d_first_m_mu16(d, sz[0] + pad * 2, f, v, z);  
dt1d_second_m_mu16(f, sz[1] + pad * 2, d, v, z, m, 0);
dt1d_second_m_mu16(f, sz[2] + pad * 2, d, v, z, m, 1);
基本上大同小异
比如说第一个函数:
我的做法是:
1. 首先把一个个计算单元f 取出来成ndarray,存到一个list中:
2. 转化成RDD
    data = sc.parallelize(data)
3. 对每个list中的每个f 分别执行函数dt1d_first_m_mu16, 返回计算结果d
  data_first = data.map(dt1d_first_m_mu16).collect()
  dt1d_first_m_mu16基本上是照抄原函数
这一步输出结果和单机程序一样,没什么问题

但是第二个函数就和单击程序有差别了,

1. 第二个函数看起来是以每个图像切片的行为计算单元改为以每个图形切片的列为计算单元,我就用相似的方法,把每一列的数据存成ndarray,然后组合成list
2. 然后转化成RDD, 分别执行函数dt1d_second_m_mu16
sqr_field = 0
data = sc.parallelize(data)
data_second = data.map(dt1d_second_m_mu16).collect()
这个dt1d_second_m_mu16基本上也是原函数改写成python

不知道老师之前指的会存在一些差异是什么样的差异,这边dist的结果也不太好描述,就找了一个小数据,把他和Stack_local_max一起执行,输出

单机程序找到41个种子点:
41 
30359 100853 118489 141150 148285 158183 178080 217774 227672 257468 262040 277365 287263 302337 317059 326957 346854 386548 406445 413128 426242 446139 456037 463524 485833 495731 525527 535425 544118 555322 634714 644720 654629 664634 674543 684549 694458 704464 714373 724379 734287 

我做的并向程序找到17个
17
503821 505730 525527 535425 544118 555322 634714 644720 654629 664634
674543 684549 694458 704464 714373 724379 734287

可以看到一部分是重合的,不知道这样的结果可不可以接收
好累,问问赵老师,会去休息啦


## 3.21
问题发给赵老师,他是这样回复的:
我看了一下，程序看着没有问题，最后并行程序输出的种子点只包括了图像后面一部分，应该是哪里漏了一些数据。你可以把每一步处理后的图像存下来看看有什么不同。
另外将这些c函数转换成并行程序其实不会有太多性能的优势。这些函数每次处理的数据很少（如只有一行或一列），这样光分布这些数据都会占用很多时间。我原来的意思是将图像分块，然后调用neutube命令行对每块图像计算种子点。这样RDD是图像分块的集合，map函数是将图像分块输入转换成种子点输出。我对Spark也不熟，但其对应的python API直观易用，对于图像分块处理这样的简单逻辑应该是很容易写的。
分块大小
可以以512x512x256为一块。

确实是这样的,函数每次处理的数据很少（如只有一行或一列）光分布这些数据都会占用很多时间,这次分块做一下,其实比原来还容易了.
做了一下Stack_Bwdist_L_U16_2.py
1. 图像分块data = image2block(data)
2. 处理data = sc.parallelize(data)
data = data.map(dt3d_mu16).collect()
3. 结果转化为array: data_out = block2image(data)

结果是这样子的: 和原来的结果基本一致
data: 100 * 100 * 100

单机程序:
41 30359 100853 118489 141150 148285 158183 178080 217774 227672 257468 262040 277365 287263 302337 317059 326957 346854 386548 406445 413128 426242 446139 456037 463524 485833 495731 525527 535425 544118 555322 634714 644720 654629 664634 674543 684549 694458 704464 714373 724379 734287 

block: 25 * 25 * 25
30359 100853 118489 148285 158183 161347 178080 181447 187470 207772 217770 227268 237063 267664 272139 277768 317059 326957 346854 352733 406445 413128 426242 446139 456037 473622 495223 524019 535528 544118 555320 594417 595310 604410 605308 625151 634314 644318 644331 645328 664242 665252 674257 684381 685176 763861 763957 764053 764151 764246 764344 764442 765157 765176 773970 774067 774163 774260 774358 774456 783886 783982 784079 784371 794288 65 seeds found.

block: 50 * 50 * 50
30359 100853 118489 148285 158183 161347 178080 181447 217774 227672 257468 262040 277365 287263 302337 317059 326957 346854 406445 413128 426242 446139 456037 463524 495223 524019 544118 545424 594417 595310 604410 605308 625151 634314 644320 664242 665252 684257 714173 724180 734185 41 seeds found.

block: 100 * 100 * 100
30359 100853 118489 141150 148285 158183 178080 217774 227672 257468 262040 277365 287263 302337 317059 326957 346854 386548 406445 413128 426242 446139 456037 463524 485833 495731 525527 535425 544118 555322 634714 644720 654629 664634 674543 684549 694458 704464 714373 724379 734287 41 seeds found.

但是发现一个问题:

``` cpp
Stack *Stack_Bwdist_L_U16(const Stack *in, Stack *out, int pad)
{
  ASSERT(in->kind == GREY, "GREY stack only");

  if (out == NULL) {
    out = Make_Stack(GREY16, in->width, in->height, in->depth);
  }
  int i;

  int nvoxel = Stack_Voxel_Number(in);

  uint16 *out_array = (uint16 *) out->array;

  for (i = 0; i < nvoxel; i++) {
    out_array[i] = in->array[i];
  }

  long int sz[3];
  sz[0] = in->width;
  sz[1] = in->height;
  sz[2] = in->depth;
  //dt3d_u16(out_array, sz, pad);
  /* The meaning of pad is different in the private function */

   // FILE *stream;
   // stream = fopen( "/home/gbw/PycharmProjects/DvidSpark/smalldata/input.txt", "w+" );
   // fprintf(stream, "%d %d %d %d %d\n", sz[0], sz[1], sz[2], nvoxel, !pad);
   // for (i=0;i<nvoxel;i++)
   //      fprintf( stream, "%d ", out_array[i]);
   // fclose(stream);

   dt3d_binary_mu16(out_array, sz, !pad);
 
   // FILE *fp;
   // fp=fopen("/home/gbw/PycharmProjects/DvidSpark/smalldata/mydist.txt","r");
   // int area = sz[0] * sz[1] * sz[2];
   // int a;
   // for(i=0;i<area;++i)
   // {
   //     fscanf(fp,"%d",&a);
   //    out_array[i] =a;
   // }
   // fclose(fp);

  return out;
}
```

这里 函数dt3d_binary_mu16(out_array, sz, !pad)返回的out_array 和 最终结果返回的out->array 其实是不一样的
原因在:
uint16 *out_array = (uint16 *) out->array;
但是stack的定义:
``` cpp
typedef struct
  { int      kind;
    int      width;
    int      height;
    int      depth;
    char    *text;
    uint8   *array;   // array of pixel values lexicographically ordered on (z,y,x,c)
  } Stack;
```
这样强制类型转换真的没关系吗?

## 3.22
老师说那个可以自动转换. 虽然感觉还是不太对,但是和我做的部分没多大关系,就不想管了.
把Stack_Local_Max部分得分快处理做了,还是在那个100*100*100的数据上测试的结果并不理想,不过也无所谓了,这部分也不是重点

Stack_Local_Max2.py测试:

单机程序: 41 seeds
分块100 * 100 * 100: 41 seeds
分块50* 50 * 50 : 1002 seeds

接下来:
1. 找一份大一点的数据测试一下
2. TraceNeoron 命令行程序
3. TraceNeuron Neutu源码阅读
4. 开题报告!!
5. 毕设报告的系统架构和种子点提取部分

## 3.23
整合了一下tz_stack_bwmorph.c 和 Stack_Bwdist_L_U16的代码
100 * 100 * 100 小数据测试正确
找了份1024 * 1024 * 100 的数据测试:
一开始内存不足:
: java.lang.OutOfMemoryError: Java heap space
17/03/23 08:53:25 INFO MemoryStore: MemoryStore started with capacity 511.1 MB
spark-submit 的时候 --driver-memory 4g 解决
然后莫名其妙:
17/03/23 10:18:30 INFO SparkContext: Successfully stopped SparkContext
17/03/23 10:18:30 INFO ShutdownHookManager: Shutdown hook called
17/03/23 10:18:30 INFO Executor: Finished task 0.0 in stage 0.0 (TID 0). 963 bytes result sent to driver
17/03/23 10:18:30 INFO ShutdownHookManager: Deleting directory /tmp/spark-e70a8c41-ebad-4da4-b103-2ca758faa5c7
17/03/23 10:18:30 INFO ShutdownHookManager: Deleting directory /tmp/spark-e70a8c41-ebad-4da4-b103-2ca758faa5c7/pyspark-ed90e801-dcf3-42de-b104-93687242b553
17/03/23 10:18:30 ERROR Executor: Exception in task 0.0 in stage 0.0 (TID 0)
java.lang.IllegalStateException: RpcEnv already stopped.
  at org.apache.spark.rpc.netty.Dispatcher.postMessage(Dispatcher.scala:159)
  at org.apache.spark.rpc.netty.Dispatcher.postOneWayMessage(Dispatcher.scala:131)
  at org.apache.spark.rpc.netty.NettyRpcEnv.send(NettyRpcEnv.scala:192)
  at org.apache.spark.rpc.netty.NettyRpcEndpointRef.send(NettyRpcEnv.scala:516)
  at org.apache.spark.scheduler.local.LocalBackend.statusUpdate(LocalBackend.scala:151)
  at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:304)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)

没有很好的解决方法,明天再弄吧

## 4.10
配置graphframe 和 pycharm 环境,参照:
http://stackoverflow.com/questions/39572603/using-graphframes-with-pycharm

原来还以为可以兼容graphx的函数,可是graphx并没有python的版本,准备好的scala版本prim也就没法参考实现了

```
  //最小生成树
  def prime[VD: ClassTag](g : Graph[VD, Double], origin: VertexId) = {
    //初始化，其中属性为（boolean, double，Long）类型，boolean用于标记是否访问过，double为加入当前顶点的代价，Long是上一个顶点的id
    var g2 = g.mapVertices((vid, _) => (false, if(vid == origin) 0 else Double.MaxValue, -1L))

    for(i <- 1L to g.vertices.count()) {
      //从没有访问过的顶点中找出 代价最小 的点
      val currentVertexId = g2.vertices.filter(! _._2._1).reduce((a,b) => if (a._2._2 < b._2._2) a else b)._1
      //更新currentVertexId邻接顶点的‘double’值
      val newDistances = g2.aggregateMessages[(Double, Long)](
        triplet => if(triplet.srcId == currentVertexId && !triplet.dstAttr._1) {    //只给未确定的顶点发送消息
          triplet.sendToDst((triplet.attr, triplet.srcId))
        },
        (x, y) => if(x._1 < y._1) x else y ,
        TripletFields.All
      )
      //newDistances.foreach(x => println("currentVertexId\t"+currentVertexId+"\t->\t"+x))
      //更新图形
      g2 = g2.outerJoinVertices(newDistances) {
        case (vid, vd, Some(newSum)) => (vd._1 || vid == currentVertexId, math.min(vd._2, newSum._1), if(vd._2 <= newSum._1) vd._3 else newSum._2 )
        case (vid, vd, None) => (vd._1|| vid == currentVertexId, vd._2, vd._3)
      }
      //g2.vertices.foreach(x => println("currentVertexId\t"+currentVertexId+"\t->\t"+x))
    }

    //g2
    g.outerJoinVertices(g2.vertices)( (vid, srcAttr, dist) => (srcAttr, dist.getOrElse(false, Double.MaxValue, -1)._2, dist.getOrElse(false, Double.MaxValue, -1)._3) )
  }
```

没办法,只好参照graphframe的api:
http://graphframes.github.io/api/scala/index.html#org.graphframes.GraphFrame
自己完成最小生成树算法,有待优化

## 4.13

### 总体设计
目标是设计一个基于Spark的神经元并行重建系统.该系统以大规模光学显微镜图像为研究对象,以Spark的并行计算框架为计算平台,帮助神经科学家进行神经数据的分析,完成复杂的神经元结构自动重建工作.

#### 3.1 架构设计
系统的结构视图如图3-1所示:
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/architecture%20%283%29.png)
图3-1: 系统结构视图

#### 3.2 处理逻辑
根据图3-1, 系统处理逻辑如下:
1) 用户调用系统的用户访问接口,提交神经元重建任务.
2) 根据用户调用的接口和输入的参数, 从DVID中加载图像数据信息.
3) 对图像数据进行预处理, 也就是分块图像分块.
4) 图像经过分块后,开始神经元追踪工作,神经元追踪的主要工作包括:
   a) 图像块加载.
   b) 提交图像块追踪子任务.
   c) 监控子任务运行状态.
5) Spark 平台将具体子任务调度至各个计算节点运行,所有子任务运行完成后得到各个图像块的神经元分支.
6) 将各个神经元分支部分抽象为一张图,使用最小生成树算法，得到重建后的神经元树型结构.
7) 输出最后结果为swc文件,并返回给用户

#### 3.3 系统模块
根据系统的结构视图,以下将对其中模块进行详细的阐述,如下图所示:
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/architecture2%20%282%29.png)
图3-2: 系统的模块视图

#### 3.4 算法的Spark并行化分析
Spark的并行化主要通过建立逻辑执行图,即数据流的流向过程,然后根据逻辑执行图划分为物理执行图,来实现任务并行化,最后生成任务分配到节点执行

##### 3.4.1 数据并行化 
逻辑执行图描述的是 job 的数据流：job 会经过哪些 transformation()，中间生成哪些 RDD 及 RDD 之间的依赖关系。
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/%E9%80%BB%E8%BE%91%E5%9B%BE_Last.png)
图3-3: job逻辑执行图

1) 从数据源(DVID)读取数据创建最初的 RDD。
2) 对 RDD,执行map()操作,种子点检测,得到种子点集合的RDD
3) 对种子点集合的RDD,执行map()操作,种子点拟合,得到拟合神经元分段集合的RDD
4) 拟合神经元分段集合的RDD,执行sortBy操作,根据拟合得分对这些神经元分段进行排序,得到排序后的神经元分段集合RDD
5) 对每个神经元分段子集进行追踪,执行map操作,得到追踪后的神经元分支的子集合RDD
6) 对最后的神经元分支 RDD 进行 collect() 操作，每个 partition 计算后产生结果 result
7) 将 result 回送到 driver 端，把多个result的array合并成一个array

##### 3.4.2 任务并行化
逻辑执行图表示的是数据上的依赖关系，不是 task 的执行图。在 Hadoop 中，用户直接面对 task，mapper 和 reducer 的职责分明：一个进行分块处理，一个进行 aggregate。Hadoop 中 整个数据流是固定的，只需要填充 map() 和 reduce() 函数即可。Spark 面对的是更复杂的数据处理流程，数据依赖更加灵活，很难将数据流和物理 task 简单地统一在一起。因此 Spark 将数据流和具体 task 的执行流程分开，并设计算法将逻辑执行图转换成 task 物理执行图.
![enter image description here](http://oa4pac8sx.bkt.clouddn.com/%E7%89%A9%E7%90%86%E5%9B%BE_Last.png)
图3-4: job物理执行图

整个job被划分为两个stage:
1) Stage 1 种子点提取,该 stage 里面的 task 都是  ShuffleMapTask.计算结果需要 shuffle 到下一个 stage，本质上相当于 MapReduce 中的 mapper。
2) Stage 0 种子点追踪,该 stage 里面的 task 都是 ResultTask,ResultTask 相当于 MapReduce 中的 reducer（如果需要从 parent stage 那里 shuffle 数据）
stage 里面 task 的数目由该 stage 最后一个 RDD 中的 partition 个数决定。

参考 https://www.kancloud.cn/kancloud/spark-internals/45240


