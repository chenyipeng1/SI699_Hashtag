PORT="$(shuf -i 10000-60000 -n 1)"
echo $PORT
pyspark --master yarn --conf spark.ui.port="${PORT}" --num-executors 32  --executor-memory 2g
