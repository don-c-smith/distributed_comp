# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Import, Cleaning, and Dataframe/Schema Building

# COMMAND ----------

# Data Import and Header Handling
full_rdd = sc.textFile('/FileStore/tables/Master.csv')  # Import full datafile from DataBricks filestore

# Guidance for handling header row from:
# https://stackoverflow.com/questions/32877326/how-to-skip-more-then-one-lines-of-header-in-rdd-in-spark
header_row = full_rdd.first()  # Isolate header row in its own RDD
rdd = full_rdd.filter(lambda row: row != header_row)  # Filter full RDD by removing header row
# print(rdd.take(10))  # Print debugging - looks good

# Filtering Rows and Columns
# We need to parse the file to see at which index positions the features we need occur
# We can use the isolated header RDD to do this
features = header_row.split(',')  # Split the fields in the header
# Create objects containing the column index values of the needed fields using Python's .index() method
# Exact column names were provided - we store the idx values of those features
playerID_idx = features.index('playerID')
birthCountry_idx = features.index('birthCountry')
birthState_idx = features.index('birthState')
height_idx = features.index('height')

# Now that we have the feature index positions stored, we can parse/clean the RDD
# I'll avoid the repeated creation of RDDs by chaining method calls
clean_rdd = (rdd.map(lambda row: row.split(','))  # Split fields on commas with .map()
             .filter(
    lambda features: features[height_idx] != '')  # Filter out rows with empty height fields per prompt using .filter()
             .map(lambda cols: (cols[playerID_idx], cols[birthCountry_idx], cols[birthState_idx], int(
    cols[height_idx]))))  # Map only desired columns to the clean rdd - cast height as integer-type

# print(clean_rdd.take(10))  # Print debugging - looks good

# Build Schema - all columns are string-type except height, which is integer-type
my_schema = StructType([StructField('playerID', StringType()),
                        StructField('birthCountry', StringType()),
                        StructField('birthState', StringType()),
                        StructField('height', IntegerType())])

# Now that we have an explicit schema and a clean RDD, we can create our dataframe
df = spark.createDataFrame(clean_rdd, my_schema)

# Print debugging - looks good
print(df.show(10))
print(my_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Query 1
# MAGIC
# MAGIC Find the number of players born in Colorado both using SparkSQL syntax and with direct,
# programmatic DataFrame functions.

# COMMAND ----------

# Using SparkSQL syntax
df.createOrReplaceTempView('player_info')
print('Count of players born in Colorado (SparkSQL syntax):')
# Using triple-quotes to allow for best-practices SQL line separation and tab spacing
spark.sql("""
        SELECT COUNT(*) AS CO_count
            FROM player_info
                WHERE birthState = 'CO'
        """).show()

# Using DataFrame functions
print('Count of players born in Colorado (DataFrame functions):')
print(df.filter(df.birthState == 'CO').count())  # Using .filter() on the df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Query 2
# MAGIC
# MAGIC Find the average height by birth country of all players, ordered from highest to lowest.
# Do this using SparkSQL syntax and with direct, programmatic DataFrame functions.

# COMMAND ----------

# Using SparkSQL syntax
print('Average height by birth country (SparkSQL syntax):')
# Using triple-quotes to allow for best-practices SQL line separation and tab spacing
spark.sql("""
    SELECT birthCountry, AVG(height) AS avg_height
        FROM player_info
            GROUP BY birthCountry
            ORDER BY avg_height DESC
    """).show(df.count())

# Using DataFrame functions
# Again, I can chain method calls here while preserving order
print('Average height by birth country (DataFrame functions):')
(df.groupBy('birthCountry')  # Create groups first
 .agg(avg('height').alias('avg_height'))  # Then calculate aggregation
 .orderBy(desc('avg_height'))  # Then specify order
 .show(df.count()))
