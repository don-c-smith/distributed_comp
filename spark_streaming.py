# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Library imports
from pyspark.sql import functions as f
from pyspark.sql.types import *  # It's just easier to import all the types
from pyspark.sql.window import Window
import time

# COMMAND ----------

# Read in full .csv file and create multiple files

# Define schema as per prompt
fifaSchema = StructType([
    StructField('ID', LongType(), True),
    StructField('lang', StringType(), True),
    StructField('Date', TimestampType(), True),
    StructField('Source', StringType(), True),
    StructField('len', LongType(), True),
    StructField('Orig_Tweet', StringType(), True),
    StructField('Tweet', StringType(), True),
    StructField('Likes', LongType(), True),
    StructField('RTs', LongType(), True),
    StructField('Hashtags', StringType(), True),
    StructField('UserMentionNames', StringType(), True),
    StructField('UserMentionID', StringType(), True),
    StructField('Name', StringType(), True),
    StructField('Place', StringType(), True),
    StructField('Followers', LongType(), True),
    StructField('Friends', LongType(), True),
])

# Read in full .csv file, dropping any non-conforming rows
tweetsDF = spark.read.csv('/FileStore/tables/fifa.csv', schema=fifaSchema, header=True, mode='DROPMALFORMED')

# Create 20 partitions for testing
tweetsDF.repartition(20).write.csv('/FileStore/tables/tweet_files/', mode='overwrite')

# COMMAND ----------

# Static Window Creation

# Reduce columns using PySpark functions, filter out rows with no hashtags
staticDF = tweetsDF.select(f.col('ID'), f.col('Date'), f.col('Hashtags')).filter(f.col('Hashtags').isNotNull())

# Per prompt - 'explode' Hashtags column to create one row per hashtag
staticDF = staticDF.withColumn('Hashtags', f.explode(f.split('Hashtags', ',')))

# Create and apply window function
windowDF = (staticDF.groupBy(
    f.window(
        'Date',  # Organize window first at date level 
        '60 minutes',  # Set window length
        '30 minutes'),  # Set window interval
    'Hashtags')  # Then at the hashtag level
            .agg(f.count('*').alias('Count')))  # Count rows per group and rename field to 'Count'

# Filter to include hashtags with count gt 100, order the results
# More uses of PySpark functions
trendDF = windowDF.filter(f.col('Count') > 100).orderBy('window', f.desc('Count'))

# Display final DDF
trendDF.show()

# COMMAND ----------

# Build Streaming Setup

# Buld the DDF used for streaming, read from created directory
streamDF = spark.readStream.schema(fifaSchema).csv('/FileStore/tables/tweet_files/', header=False)

# Fetch and "window-ize" (can't think of a better word) the data as it streams in
# More method-chaining here, because it make more sense to me in this form
finalDF = (streamDF.filter(f.col('Hashtags').isNotNull())  # Filter out rows with no hashtags
           .withColumn('Hashtags', f.explode(f.split('Hashtags', ',')))  # Explode hashtag column as above
           .withWatermark('Date', '24 hours')  # Set "watermark"
           .groupBy(
    # Same setup as above
    f.window('Date',  # Group by date first
             '60 minutes',  # Window length
             '30 minutes'),  # Window interval
    'Hashtags')  # Then by tag
           .agg(f.count('*').alias('Count'))  # Aggregate and rename column
           .filter(f.col('Count') > 100))  # Filter for tag count gt 100

# Build and start streaming query with memory sink per the prompt
query = (finalDF.writeStream
         .outputMode('complete')
         .format('memory')
         .queryName('trendingTags')
         .start())

# COMMAND ----------

# Send query and watch results come in
while True:
    time.sleep(10)  # Wait 10 seconds between queries
    print('Tags returned from stream:')
    spark.sql('SELECT * FROM trendingTags ORDER BY window, Count DESC').show(truncate=False)
