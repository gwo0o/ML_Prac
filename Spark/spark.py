from pyspark.sql import SparkSession
from pyspark import SparkContext


spark =SparkSession.builder.appName(str(name))\
      .master("local[*]").config("spark.driver.memory","5g")\
      .config("spark.driver.host","127.0.0.1")
      .config("spark.driver.bindAddress","127.0.0.1")
      .getOrCreate()

sc = SparkContext(master="local", appName="first app")