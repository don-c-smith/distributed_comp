# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Library import
from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ####Load Files and Build Tables

# COMMAND ----------

# Load CSV files into distributed dataframes
# Select only needed columns and infer the schema
# Note that all the .csv files have clear interpretable headers
master_df = spark.read.csv('dbfs:/FileStore/tables/Master.csv', header=True, inferSchema=True).select('playerID',
                                                                                                      'nameFirst',
                                                                                                      'nameLast')

teams_df = spark.read.csv('dbfs:/FileStore/tables/Teams.csv', header=True, inferSchema=True).select('teamID',
                                                                                                    'name')

allstar_df = spark.read.csv('dbfs:/FileStore/tables/AllstarFull.csv', header=True, inferSchema=True).select('playerID',
                                                                                                            'teamID')

# Print debugging
master_df.show(5)
teams_df.show(5)
allstar_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join Dataframes

# COMMAND ----------

# Joining the DataFrames, beginning with allstar_df because it's both smaller than master and it's the linking table
# Inner join, using playerID and teamID fields as the join fields
joined_df = allstar_df.join(master_df, 'playerID').join(teams_df, 'teamID')

# Remove duplicates using .distinct()
joined_df = joined_df.distinct().withColumnRenamed('name', 'teamName')  # Renaming column

# Select only the required columns in order specified
final_df = joined_df.select('playerID', 'teamID', 'nameFirst', 'nameLast', 'teamName')

# Print debugging
final_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build and Save Parquet File

# COMMAND ----------

# Save final dataframe as Parquet file, partition by teamName
final_df.write.partitionBy('teamName').parquet('/FileStore/team_allstars.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read Parquet File and Run Queries

# COMMAND ----------

# Read Parquet file into dataframe and filter for Rockies players only
COL_df = (spark.read.parquet('/FileStore/team_allstars.parquet')
          .filter(col('teamName') == 'Colorado Rockies')
          .select('nameFirst', 'nameLast'))

# Print number of Colorado Rockies allstars (should be 24)
print(f'Count of All-Stars from the Colorado Rockies: {COL_df.count()}')

# Show interactive list of Colorado Rockies allstars
# This method display(df) is a cool DataBricks-specific capability which I read about here
# https://www.databricks.com/spark/getting-started-with-apache-spark/dataframes
display(COL_df)
