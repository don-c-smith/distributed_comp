# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Library imports
from pyspark.sql.functions import col
from graphframes import GraphFrame

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 1: Read & Process Data**

# COMMAND ----------

# Read in routes.csv - I'm only going to explicitly define a schema if the inference fails. The file has a header.
routes_df = spark.read.csv('/FileStore/tables/routes.csv', header=True, inferSchema=True)

# Read in airports.csv. Same comments about schema inference and header apply.
airport_df = spark.read.csv('/FileStore/tables/airports.csv', header=True, inferSchema=True)

# Remove duplicate routes by calling .distinct()
routes_df = routes_df.select('sourceAirport', 'destinationAirport').distinct()

# Filter for US airports and extract relevant columns using .select()
us_airports = airport_df.filter(col('country') == 'United States').select('IATA')

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 2: Create Dataframes**

# COMMAND ----------

# Building the 'edges' DDF - Once again, I'm chaining my method calls. This is in essence just a bunch of inner joins.
edges = (routes_df
         # Limits sources to US airports - aliases are to prevent column confusion
         .join(us_airports.alias('src_airports'), routes_df.sourceAirport == col('src_airports.IATA'), 'inner')
         .join(us_airports.alias('dst_airports'), routes_df.destinationAirport == col('dst_airports.IATA'),
               'inner')  # Limits destinations to US airports - same note re: aliases
         .select(col('sourceAirport').alias('src'), col('destinationAirport').alias('dst'))
         # Select only source and destination airports and alias to be in conformance with GraphFrames
         )

# Building the 'vertices' DDF as follows using method-chaining:
# I select only the source and destination airports by calling .select()
# I call the .union() method on the results of the .select() calls to combine the results of the calls
# Note that this *will* create duplicates, which is desired
# I call .distinct() to remove those duplicates and generate a unique list
# I rename the source column to 'id' to be in conformance with GraphFrames
vertices = edges.select('src').union(edges.select('dst')).distinct().withColumnRenamed('src', 'id')

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 3: Create GraphFrame**

# COMMAND ----------

gf = GraphFrame(vertices, edges)  # Build graphframe

# COMMAND ----------

# MAGIC %md
# MAGIC **Question 1: Count all US Airports and Domestic Routes**

# COMMAND ----------

# This can be done directly with f-strings
print(f'Total # of US Airports: {vertices.count()}')
print(f'Total # of US Domestic Routes: {edges.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC **Question 2: Find US Airports with Only One-Way Flights to/from DEN**

# COMMAND ----------

'''
Implementing this motif requires some explanation. Here's what's happening:
1. The call to the graphframe's .find() method looks for a specified pattern in the graph.
2. This pattern: (a)-[e]->(b); !(b)-[]->(a) finds one-way paths because these are all paths which flow 
from a to b but *not* from b to a.
3. The call to .filter() ensures that exactly one of the airport pairs is DEN - i.e. that there's no out-and-back path, 
which wouldn't be a one-way flight
'''
oneway_motif = gf.find("(a)-[e]->(b); !(b)-[]->(a)").filter(
    "(b.id = 'DEN' or a.id = 'DEN') and (b.id != 'DEN' or a.id != 'DEN')")

'''
And here's how we get the one-way airport list:
1. The when/otherwise pairing just means that if 'b' in the pattern is DEN, select 'a', and if 'b' in the pattern 
is *not* DEN, select 'b'.
2. I then alias the results of the query (because this actually a SparkSQL implementation) as 'IATA'.
3. Credit to https://sparkingscala.com/examples/when-otherwise/ for supplying the neat when/otherwise idea.
'''
oneway_results = oneway_motif.select(when(col('b.id') == 'DEN', col('a.id')).otherwise(col('b.id')).alias('IATA'))

print('Airports with no direct roundtrip to or from DEN:')  # Header as per the prompt
oneway_results.show()  # Show the query results

# COMMAND ----------

# MAGIC %md
# MAGIC **Question 3: Find US Airports Requiring Four or More Flights to Reach DEN**

# COMMAND ----------

'''
To calculate the shortest paths from DEN to all other airports:
1. I set 'DEN' as the landmark from which to compute the paths.
2. The call to .shortestPaths() creates a new column called 'distances' which is just a mapping of distances 
to the specified landmark.
'''
paths = gf.shortestPaths(landmarks=['DEN'])

# Now, I chain method calls to find only those airports which are four or more 'hops' away from DEN
airports_gte4 = (paths
                 .filter(col('distances.DEN') >= 4)  # Filter for 'hops' counts gte 4
                 .select('id',
                         col('distances.DEN').alias('Hops'))  # Select airport ID and # of hops, alias per the prompt
                 .orderBy('Hops', 'id')  # Specify order of query return to match prompt
                 )

print('Airports that require four or more flights to get to DEN:')  # Create header
airports_gte4.show(truncate=False)  # Display the query results and do not truncate
