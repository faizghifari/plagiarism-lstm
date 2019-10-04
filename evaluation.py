import numpy
import tensorflow as tf

from keras import backend as K
from sklearn.metrics import confusion_matrix

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def confusion_matrix_m(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()

def eval_summary(results, metric_names):
    summary = []
    for i in range(len(results)):
        temp = dict()
        temp[metric_names[i]] = results[i]
        summary.append(temp)
    
    return summary

def rescale_predictions(y_pred):
    pred = []
    for _, predicted in enumerate(y_pred):
        if predicted[0] > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    
    return pred