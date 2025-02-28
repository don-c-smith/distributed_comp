# Databricks notebook source
# MAGIC %md
# MAGIC ### Don Smith
# MAGIC #### Parallel and Distributed Computing

# COMMAND ----------

# Library imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

# Load training and test data from supplied .csv files
# Both files have headers, automatic schema inference should be easy/sufficient
train_data = spark.read.csv('/FileStore/tables/heartTrain.csv', header=True, inferSchema=True)
test_data = spark.read.csv('/FileStore/tables/heartTest.csv', header=True, inferSchema=True)

age_buckets = [-float('inf'), 40, 50, 60, 70, float('inf')]  # Demarcate age 'buckets'

# I can construct a list of pipeline steps and use it later for better code concision and modularity
pipe_steps = [
    # Apply bucketizer to 'age' using defined buckets
    Bucketizer(splits=age_buckets, inputCol='age', outputCol='age_bucket'),
    StringIndexer(inputCol='sex', outputCol='sex_num'),  # Convert 'sex' to numeric format
    StringIndexer(inputCol='pred', outputCol='known_label'),  # Convert 'prediction' to numeric format
    # Assemble recast features into a vector
    VectorAssembler(inputCols=['age_bucket', 'sex_num', 'chol'], outputCol='features'),
    LogisticRegression(featuresCol='features', labelCol='known_label')  # Instantiate a logistic regressor
]

pipeline = Pipeline(stages=pipe_steps)  # Instantiate the pipeline
model = pipeline.fit(train_data)  # Fit the pipeline to the training data

train_preds = model.transform(train_data)  # Make predictions on the training set
test_preds = model.transform(test_data)  # Make predictions on the test set

# Display the first 20 predictions on test set
print('Results of Predictions on Test Set:')
test_preds.select('id', 'probability', 'prediction').show(20)
print()

# Evaluate model performance on the training data - computing both ROC and Accuracy
# Compute area Under ROC using the binary classification evaluator
roc_evaluator = BinaryClassificationEvaluator(labelCol='known_label',
                                              rawPredictionCol='rawPrediction',
                                              metricName='areaUnderROC')
roc_auc = roc_evaluator.evaluate(train_preds)

# Compute accuracy using the multiclass classification evaluator
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='known_label',
                                                       predictionCol='prediction',
                                                       metricName='accuracy')
accuracy = accuracy_evaluator.evaluate(train_preds)

print(f'Area Under ROC on Training Set: {roc_auc}')
print(f'Accuracy on Training Set: {accuracy}')
