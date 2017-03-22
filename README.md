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

#####test.sh

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

###StackLocalMax.py
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