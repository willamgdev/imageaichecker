from imageai.Prediction import ImagePrediction
import tensorflow as tf

import os
execution_path = os.getcwd()


prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(
    execution_path, "DenseNet-BC-121-32.h5"))
prediction.loadModel()
print ('testing')
predictions, probabilities = prediction.classifyImage(
    os.path.join(execution_path, "c63.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
