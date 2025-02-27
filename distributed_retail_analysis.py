# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Library imports
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder  # For data preparation
from pyspark.ml.classification import RandomForestClassifier  # This is my actual classifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # This is to evaluate model performance
from pyspark.ml import Pipeline  # To create the ML pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 1: Data Handling, Estimator Construction, Performance Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC **Read Datafile and Handle Missing Values in Target Feature**

# COMMAND ----------

# Read in the .csv file - the file has a header and inferring the schema should be straightforward
df = spark.read.csv('/FileStore/tables/assignment2/retail_spark.csv', header=True, inferSchema=True)

# Print debugging
print('Inferred schema for DDF:')
print(df.printSchema())  # This looks exactly correct
print()
init_row_count = df.count()
print(f'Original Row Count: {init_row_count}')  # About 300,000 rows
print()

# Drop any rows with missing values
df = df.dropna()
new_row_count = df.count()
print(f'Row Count After Dropping Missing Target Values: {new_row_count}')
print()

# Calculate row loss rate after call to .dropna()
row_loss = ((init_row_count - new_row_count) / init_row_count) * 100
print(f'Proportional Row Loss After Drop: {row_loss:.2f}%')
# Row loss rate is tiny - I can proceed without potentially bias-introducing imputation

# COMMAND ----------

# MAGIC %md
# MAGIC **Print Resulting Dataframe**

# COMMAND ----------

df.show(5)  # Print first few rows of DDF as a sanity check

# COMMAND ----------

# MAGIC %md
# MAGIC **Prepare Input Features and Classifier, Build Step List**

# COMMAND ----------

# Define lists of features by datatype
cat_cols = ['sex', 'income', 'segment', 'category']  # Categorical features to be encoded
num_cols = ['age', 'amount']  # Numerical features to be scaled

steps = []  # Instantiate an empty list to hold the data preparation steps

# Encode categorical features
for column in cat_cols:  # For each categorical feature
    # Append 'idx' suffix to string-indexed columns, keep and code any unseen labels encountered
    index = StringIndexer(inputCol=column, outputCol=column + 'idx', handleInvalid='keep')

    # Take indexed columns and create actual encoded vectors
    encode = OneHotEncoder(inputCols=[index.getOutputCol()], outputCols=[column + 'classVec'])

    steps += [index, encode]  # Add the indexing and encoding steps to the data preparation steps list

# Scale numerical features
num_assembler = VectorAssembler(inputCols=num_cols, outputCol='numFeatures')  # Build vectors of numerical features
num_scaler = StandardScaler(inputCol='numFeatures', outputCol='scaledNumFeatures')  # Apply the standard scaler
steps += [num_assembler, num_scaler]  # Add the assembly and scaling steps to the data preparation steps list

# Assemble the final set of encoded and scaled features
final_inputs = [feature + 'classVec' for feature in cat_cols] + ['scaledNumFeatures']  # Build list of input features
final_assembler = VectorAssembler(inputCols=final_inputs, outputCol='features')  # Run assembler on that list
steps += [final_assembler]  # Add the final assembly step to the data preparation steps list

# Instantiate a Random Forest Classifier
rf_class = RandomForestClassifier(labelCol='rating',  # Target feature
                                  featuresCol='features',  # Final 'assembled' features
                                  numTrees=100,  # 100 is a pretty 'ordinary' number of trees for a base estimator
                                  maxDepth=10)  # 10 balances sufficient complexity and avoiding overfitting

steps += [rf_class]  # Add the classifier to the steps list

# Finally, build the pipeline
model_pipeline = Pipeline(stages=steps)

# COMMAND ----------

# MAGIC %md
# MAGIC **Split Data into Training and Test Sets**

# COMMAND ----------

# Split dataset at random using 70/30 proportions, set seed for reproducibility
(train_set, test_set) = df.randomSplit([0.7, 0.3], seed=4)

# Print debugging
train_count = train_set.count()
test_count = test_set.count()
print(f'Row Count for Training Set: {train_count}')
print()
print(f'Row Count for Test Set: {test_count}')
print()
print('First Five Rows of Training Set:')
print(train_set.show(5))
print()
print('First Five Rows of Test Set:')
print(test_set.show(5))

# COMMAND ----------

# MAGIC %md
# MAGIC **Instantiate the Model, Fit the Pipeline, Evaluate the Classifier**

# COMMAND ----------

# Build the model by calling the pipeline and fitting to the training set
rating_model = model_pipeline.fit(train_set)

# Evaluate model performance on the training set
train_preds = rating_model.transform(train_set)

# Compute and show model performance metrics on the training set
# Weighted metrics are necessary because this is a multi-class classification problem
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure']
print('Model Performance on the Training Set:')
for metric in metrics:  # For each performance metric
    # Instantiate the evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol='rating', predictionCol='prediction', metricName=metric)
    score = evaluator.evaluate(train_preds)  # Compute the given metric
    print(f'{metric}: {score:.4f}')  # Print the metric

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 2: Streaming

# COMMAND ----------

# Create 20 files for streaming simulation
test_set.repartition(20).write.mode('overwrite').parquet('/FileStore/tables/assignment2/streaminput')

# Fetch the schema from the original, pre-split dataframe
init_schema = df.schema

# Set up your streaming context using the original schema
stream_df = spark.readStream.schema(init_schema).parquet('/FileStore/tables/assignment2/streaminput')

# Now, I define a function to process and evaluate model performance on each batch
def process_batch(batch_df, batch_id):
    """
    This function processes a given "batch" of simulated streaming data and evaluates my classifier's performance
    on that batch.
    It applies the pre-trained rating_model classifier and calculates appropriate performance metrics at batch level.
    Args:
        batch_df (pyspark.sql.DataFrame): A DataFrame containing a batch of simulated streaming data.
        batch_id (int): A numerical identifier for the currently-processed batch.
    Returns:
        None. This is a void function. It just prints batch-level performance metrics.
    """
    preds = rating_model.transform(batch_df)  # Apply model to the streaming data
    
    # Evaluate model performance
    metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure']  # Same measures as above
    print(f'Batch ID: {batch_id}')
    for metric in metrics:  # For each metric
        # Instantiate an evaluator
        evaluator = MulticlassClassificationEvaluator(labelCol='rating', predictionCol='prediction', metricName=metric)
        score = evaluator.evaluate(preds)  # Calculate the score for the given metric
        print(f'{metric}: {score:.4f}')  # Display the score
    print()

# Start streaming query
query = stream_df.writeStream.foreachBatch(process_batch).start()  # Call the process_batch function for each batch

# Wait for the stream to terminate
query.awaitTermination()
