spark-submit \
  --master local[2] \
  --num-executors 2 \
  --executor-memory 2G \
  --executor-cores 1 \
  --total-executor-cores 2 \
  --driver-memory 1G \
  --conf spark.default.parallelism=10 \
  --conf spark.storage.memoryFraction=0.5 \
  --conf spark.shuffle.memoryFraction=0.3 \
Main.py

