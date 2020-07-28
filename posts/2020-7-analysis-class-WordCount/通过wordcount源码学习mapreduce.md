# mapreduce学习（之一）——分析经典WordCount类

## WordCount代码分析：

```java

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;	// hadoop配置文件
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;		// hadoop的一种数据类型
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

  public static class TokenizerMapper
       extends Mapper<Object, Text, Text, IntWritable>{

private final static IntWritable one = new IntWritable(1);
private Text word = new Text();

public void map(Object key, Text value, Context context
                ) throws IOException, InterruptedException {
  // 通过StringTokenizer 以空格为分隔符将一行切分为若干tokens
  // 将每一行拆分成一个个的单词，以<word，1>作为map方法的结果输出。
  StringTokenizer itr = new StringTokenizer(value.toString());
  // 遍历所有的tokens
  while (itr.hasMoreTokens()) {
    word.set(itr.nextToken());
    // 调用RecordWriter类的write方法
    context.write(word, one);
  }
}

  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();
	// values的形式类似[v1, v2, v3, ...]
    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      // 将int类型转换为IntWritable类型
      result.set(sum);
      // 调用RecordWriter类的write方法，写入新的键值对
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    // 初始化配置类，该类包含系统所需的hdfs等信息
    Configuration conf = new Configuration();
    // 创建一个Job，该Job有自己的配置信息，对该conf修改不会影响整个系统的
    // 该Job名字为"word count"
    Job job = Job.getInstance(conf, "word count");
    // 装载已编写的程序。也就是上面的类
    job.setJarByClass(WordCount.class);
    // 实现map函数，根据输入的<key, value>生成中间结果
    job.setMapperClass(TokenizerMapper.class);
    // 实现conbine函数，合并map函数产生的键值对
    // 如果不设置，则不合并中间结果
    job.setCombinerClass(IntSumReducer.class);
    // 实现reduce函数，合并中间结果，产生最终结果
    job.setReducerClass(IntSumReducer.class);
    // 定义存储在hdfs的结果文件的key/value类型
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    // 构建输入的数据文件
    FileInputFormat.addInputPath(job, new Path(args[0]));
    // 构建输出的数据文件
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    // Job运行成功，程序就会正常退出。反之，报错。
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}

```

## WordCount代码输出分析：

（每个操作的意义和详细情况将在下篇介绍）

假设WordCount有两个输入文件，文件内容分别为：

​	""Hello World Bye World"

​	"Hello Hadoop Goodbye Hadoop"

1. Job在执行时，map操作，即完成TokenizerMapper类的方法会得到两组输出（分别是两个文件的）：

```
< Hello, 1>
< World, 1>
< Bye, 1>
< World, 1>
```

```
< Hello, 1>
< Hadoop, 1>
< Goodbye, 1>
< Hadoop, 1>
```

2. combine操作，即第一次完成IntSumReducer类的方法会得到新的输出：

```
< Bye, 1>
< Hello, 1>
< World, 2>
```

```
< Goodbye, 1>
< Hadoop, 2>
< Hello, 1>
```

3. reduce操作，即第二次IntSumReducer类的方法会得到最终的输出：

```
< Bye, 1>
< Goodbye, 1>
< Hadoop, 2>
< Hello, 2>
< World, 2>
```

## WordCount代码部分细节：

### 1. 加载配置文件（代码第56行）

​	运行mapreduce程序前都要初始化Configuration，该类主要是读取mapreduce系统配置信息，这些信息包括hdfs还有mapreduce，也就是安装hadoop时候的配置文件例如：core-site.xml、hdfs-site.xml和mapred-site.xml等等文件里的信息。

​	程序员开发mapreduce时候只是在填空，在map函数和reduce函数里编写实际进行的业务逻辑，其它的工作都是交给mapreduce框架自己操作的。conf包下的配置文件会告诉框架完成任务时需要的资源和信息，比如hdfs的位置，jobstracker的位置。

### 2. 装载编写的程序（代码第61行）

​	虽然编写mapreduce程序时，我们只需要实现map函数和reduce函数，但是实际开发过程中要实现三个类，第三个类是为了配置mapreduce如何运行map和reduce函数，准确的说就是构建一个mapreduce能执行的job了，例如WordCount类。

### 3. combine操作（代码第66行）

​	在mapreduce操作中，combine操作是可选的。但是加上combine操作会让代码效率更高。

### 4. 为什么使用IntWritable而不是Int？为什么使用LongWritable而不是Long？（代码第19、38行）

​	LongWritable是针对Long类型的WritableComparable接口，IntWritable是针对Int类型的WritableComparable接口。

​	<b>Comparable</b>接口的抽象方法能快速比较两个对象的大小

​	<b>Writable</b>接口能以序列化的形式写数据到本地磁盘。因为JAVA的序列化笨重并且缓慢，所有Hadoop用Writable实现序列化和反序列化。

​	<b>WritableComparable</b>是上面两种接口的结合。

​	<b>int</b>作为原始类型不能用在键值对中。Integer是它的包装器类。

​	<b>IntWritable</b>是Hadoop环境中能更快实现序列化Integer变体。比JAVA的序列化表现得好。



-----------
参考资料：<br>
[案例单词计数-WordCount](https://my.oschina.net/gently/blog/669168) <br>
[Why we use IntWritable instead of Int?](https://community.cloudera.com/t5/Support-Questions/Why-we-use-IntWritable-instead-of-Int-Why-we-use/td-p/228098)<br>
[Hadoop中文文档](https://hadoop.apache.org/docs/r1.0.4/cn/mapred_tutorial.html)<br>
[hadoop 学习笔记：mapreduce框架详解](https://www.cnblogs.com/sharpxiajun/p/3151395.html)<br>

