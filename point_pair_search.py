# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Library imports
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 1 - Create Directory and Move files**

# COMMAND ----------

# Creating directory using dbutils
dbutils.fs.mkdirs('FileStore/tables/geodata')  # Directory for GeoPoint data

# Moving files using closed iteration
geopoint_files = ['geoPoints0.csv', 'geoPoints1.csv']

# Copy files to specified directory
for filepath in geopoint_files:
    dbutils.fs.cp(f'FileStore/tables/{filepath}', f'FileStore/tables/geodata/{filepath}')

# Print debugging
print('Contents of geodata directory')
for file_info in dbutils.fs.ls('/FileStore/tables/geodata'):
    print(f'Name: {file_info.name} - Filepath: {file_info.path}')

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 2 - Parse Data and Define Initial Cells**

# COMMAND ----------

def parse_data_make_cells(line: str, cell_size: float):
    """
    This function reads each line from the csv files, parses the data, and converts each row to a point-referent tuple.
    Then, it takes each created point tuple and assigns it to a cell in the grid.
    Args:
        line (str): A comma-separated string containing ID and coordinate information for each point.
        cell_size (float): A float-point value defining the cell size for each cell on the grid.
    Returns:
        Unnamed tuple: A tuple containing the (x, y) coordinates of each grid cell.
        Unnamed tuple: A tuple containing the point data of the form (point_id, x, y).
    """
    elements = line.split(',')  # Split each row into its component elements using commas as the delimiter
    point_id, x, y = elements[0], float(elements[1]), float(elements[2])  # Access each index position after splitting and use multiple assignment
    
    # Per the prompt, calculate and assign appropriate grid cell coordinates using floored division (//)
    x_cell_coord, y_cell_coord = int(x // cell_size), int(y // cell_size)
    
    # Return the cell coordinate and point data tuples - this function will be called directly during assignment
    return ((x_cell_coord, y_cell_coord), (point_id, x, y))

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 3 - "Pre-Push" Point Data to Neighboring Cells**

# COMMAND ----------

def pre_push_point_data(cell_tuple: tuple):
    """
    This function "pre-pushes" point data from initial cells into neighboring cells. 
    This makes the search for "close pairs" much more efficient.
    Args:
        cell_tuple (tuple): A tuple containing cell coordinates and point data.
    Returns:
        list: A list of tuples, each containing cell coordinates and point data.
    """
    cell_coord, point_data = cell_tuple  # Multiple assignment using tuple unpacking
    x_coord, y_coord = cell_coord  # Repeat multiple assignment using tuple unpacking
    
    # Instantiate the pair list using the original cell information
    pair_list = [(cell_coord, point_data)]
    
    # To avoid creating duplicates, we push the data to each of the neighboring cells in the grid
    # Recall that per RJ in live session, we only want to push in one "direction"
    for neighbor_cell in [(x_coord, y_coord + 1),  # Push directly upward
                          (x_coord + 1, y_coord + 1),  # Push up and to the right
                          (x_coord + 1, y_coord),  # Push directly to the right
                          (x_coord + 1, y_coord - 1)]:  # Push down and to the right
        
        pair_list.append((neighbor_cell, point_data))
    
    return pair_list  # Return the list of newly-associated neighboring cells with each point

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 4 - Find "Close Pairs" of Points**

# COMMAND ----------

def find_close_pairs(cell_tuple: tuple, threshold_distance: float):
    """
    This function finds "close pairs" of points within a cell and its pre-pushed neighboring cells.
    I've built it with a vectorized NumPy implementation to handle the distance calculations.
    It should be faster than using "ordinary" iteration.
    Args:
        cell_tuple (tuple): A tuple containing a sub-tuple with the coordinates of the cell and a list of points within
        the cell of the form (point_id, x, y).
        threshold_distance (float): The distance threshold defining what constitutes a "close pair" of points.
    Returns:
        close_pairs (list): A list of tuples containing the ID values of the "close pairs" of points.
    """
    cell_coord, points = cell_tuple  # Multiple assignment using tuple unpacking
    close_pairs = []  # Instantiate an empty list to store the pairs of close points

    # Here's my vectorization - I  convert points to a numpy array for more efficient distance calculations
    points_array = np.array([(float(point[1]), float(point[2])) for point in points])

    # Now I compare each point with all other points within the cell
    for i, current_point in enumerate(points):
        # Calculate distances from current point to all other points in the vectorized array
        # Using Euclidean distance and computing along the point-coordinate axis
        distances = np.sqrt(np.sum((points_array[i + 1:] - points_array[i]) ** 2, axis=1))
        
        # Find all idx values where the computed distance is lte the threshold value
        # More advantages of vectorization - the np.where operation obviates need to check all values individually
        close_point_indices = np.where(distances <= threshold_distance)[0]

        # Fetch "close point" data, create pairs, add close pairs to the list
        for close_point_idx in close_point_indices:  # For each iterated index in the array of "close point" indices...
            '''
            To avoid order-based duplicates, because "we want to treat the point pairs as unordered", we need to ensure 
            that (p3,p5) is understood to be the same as (p5,p3).
            My solution is to ensure that when the pairs are created the numerically smaller ID value always is placed 
            first in the pairing.
            I'm doing this by creating the pairs explicitly and calling the .min() and .max() methods on the ID values 
            to ensure consistency.
            So, for example, all pairs involving points p3 and p5 should be formatted as (p3,p5).
            Then, after I create each pairing, I can check if said pairing already exists in the pair list, 
            and if it does, I won't append it.
            This should prevent order-based duplicates from being added to the list.
            '''
            other_point = points[i + 1 + close_point_idx]
            # Create pair with sorted point IDs to avoid order-based duplicates
            pair = (min(current_point[0], other_point[0]),  # Lower ID value
                    max(current_point[0], other_point[0]))  # Higher ID value
            
            if pair not in close_pairs:  # If the iterated pair is not already in the pair list
                close_pairs.append(pair)  # Append it to the list of close pairs

    return close_pairs  # Return the list of IDs of close points

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 5 - Call Functions, Make RDDs, and Return Results**

# COMMAND ----------

# Define values for cell size and threshold distance per lecture
cell_size = 0.75
threshold_distance = 0.75

# Read the input files from the specified directory
raw_rdd = sc.textFile('dbfs:/FileStore/tables/geodata/*.csv')

# Parse the raw data and assign points to cells
# This transforms each line of data into a (cell_coordinates, point_data) tuple
cell_rdd = raw_rdd.map(lambda line: parse_data_make_cells(line, cell_size))

# Pre-push points to neighboring cells
pushed_rdd = cell_rdd.flatMap(pre_push_point_data)  # Here I create copies of the points in neighboring cells

# Group points at the cell level
# This aggregates all points (including pre-pushed ones) that belong to each cell
# This is a complex method-chained process that makes more sense if I explain it step by step
grouped_rdd = (pushed_rdd  # Begin with the pre-pushed RDD, which currently has points in multiple cells
                
            # Group all points by their keys (i.e. their cell coordinates).
            # All points that belong to the same cell are now grouped together.
            .groupByKey()
            
            .mapValues(list)  # Convert iterator for each cell into a list
            
            .persist())  # Persist this RDD because it's used more than once in my subsequent code

# Find the close pairs by calling the pair-locating function
close_pairs_rdd = grouped_rdd.flatMap(lambda x: find_close_pairs(x, threshold_distance))

# Remove home/neighbord duplicates using standard RDD operations
unique_pairs_rdd = (close_pairs_rdd  # Begin with close-pair RDD  
                    .map(lambda pair: (pair, None))  # Convert contents to key-value pairs
                    .reduceByKey(lambda x, y: None)  # Combine all keys
                    .map(lambda x: x[0]))  # Extract only valid pairs

# Collect results
pair_list = unique_pairs_rdd.collect()

# Print results in the format specified
print(f'Dist: {threshold_distance}')
print(f'{len(pair_list)} {pair_list}')

# Finally, unpersist the grouped_RDD
grouped_rdd.unpersist()
