import matplotlib.pyplot as plt
from featureExtractor import featureExtractor
from ann_visualizer.visualize import ann_viz
from keras.optimizers import Adam,SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout,GlobalMaxPool1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import SpatialDropout1D,TimeDistributed
from keras.layers.recurrent import LSTM
import statistics
from keras.models import load_model

from sklearn.model_selection import train_test_split
import numpy as np

f = featureExtractor("hypnogram2.csv","eeg2.csv")
sig = f.getSignalsToPredict()

model = load_model("powerModel2")
predicted = model.predict(sig[0])
f.updateFileWithNewValues(predicted,"updatedHypn2.csv")
