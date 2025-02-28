# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Construct hardcoded list
link_list = ["a b c", "b a a", "c b", "d a"]

# Construct links RDD
# Note that we can avoid the creation of unnecessary RDDs by chaining methods.
links = (sc.parallelize(link_list)  # Parallelize the link list
         .map(lambda x: x.split())  # Split each string into single elements
         .map(lambda x: (x[0], set(x[1:])))  # Create k-v pairs where the key is the element at index 0
         .mapValues(list))  # Map the pairs back into a list

# Because we'll be iteratively accessing this RDD and it won't change, I'll persist it
links.persist()  # Persist the RDD

print(f'Links RDD: {links.collect()}')  # Print debugging

# Construct rankings RDD
source_pages = links.keys().distinct()  # Extract only the distinct keys from the links RDD. These are our source pages
page_count = source_pages.count()  # Count the number of distinct source pages

# Distribute total rank space (1.0) by dividing by the distinct count of source pages
rankings = source_pages.map(lambda x: (x, 1.0 / page_count))
print(f'Initial ranking values: {rankings.collect()}')  # Print debugging

# Implement iterative step
print('Begin iterative step:')
for iteration in range(10):  # Run 10 iterations
    print()
    print(f'Iteration: {iteration}')  # Print iteration count

    join_RDD = links.join(rankings)  # Join the links and rankings RDDs
    print(f'Joined RDD: {join_RDD.collect()}')  # Print joined RDD

    # Compute neighbor contribution values for each iteration
    '''
    This flatMap implementation is a little bit complicated.
    I got the idea here:
    https://sparktpoint.com/spark-flatmap-function-usage-examples/
    What I'm doing is basically this:
    1. Within the lambda function, for each element 'x' in join_RDD
    2. Such that x[0] is the source page, x[1][0] is the list of 'neighbor' pages, and x[1][1] is the iterated rank 
        of the iterated source page
    3. Create a list of tuples configured as (neighbor page, contribution value)
    4. Where contribution is given by iterated rank/number of 'neighbor' pages
    5. NOTE: The division 'distributes' the ranking values across the total rank space (1.0)
    6. Then, apply flatMap so that each resulting tuple is deduplicated and expressed as a single element in the 
        neighbor contributions RDD
    '''
    contribs_RDD = join_RDD.flatMap(lambda x: [(neighbor, x[1][1] / len(x[1][0])) for neighbor in x[1][0]])
    print(f'Neighbor contributions: {contribs_RDD.collect()}')
    
    # Sum newly-computed rank values, combine them at the source page (i.e. key) level, and reassign to the rankings RDD
    rankings = contribs_RDD.reduceByKey(lambda x, y: x + y)
    print(f'New rankings: {rankings.collect()}')

# Sort final results
# Sort by rankings value in descending order, return to driver node as a list via .collect()
final_ranks = rankings.sortBy(lambda x: x[1], ascending=False).collect()

# Print final results
print()
print('Final sorted rankings:')
for source_page, rank in final_ranks:
    print(f'{source_page} has rank: {rank}')

links.unpersist()  # Unpersist links RDD
