# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier


train_data = "F:\ml\project\pendigits.tra"
test_data = "F:\ml\project\pendigits.tes"


#train_2d = np.loadtxt(train_data, dtype="uint8", delimiter=",")
y_train = np.array(pd.read_csv(train_data, usecols=[16],header=None))
traind = pd.read_csv(train_data,header=None)
x_train = np.array(traind.drop([16], axis=1))
x_train = x_train.reshape(-1,4,4,1)
#print(y_train)
x_train = x_train / 255.0

y_test = np.array(pd.read_csv(test_data, usecols=[16], header=None))
testd = pd.read_csv(test_data, header=None)
x_test = np.array(testd.drop([16], axis=1))
x_test = x_test.reshape(-1,4,4,1)
#print(y_test)
x_test = x_test / 255.0

#print(np.array(y_test))

digits_test = pd.read_csv(test_data,header=None)
#y_label = digits_test.iloc[:,-1]

y_one_hot = label_binarize(y_test,np.arange(10))
#print(y_one_hot.size)

model = tf.keras.models.Sequential([
    #tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(4, 4, 1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    #tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation=tf.nn.relu),
    #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("train")
res = model.fit(x_train, y_train, epochs=10)


print("evaluate")
model.evaluate(x_test, y_test)
y = model.predict(x_test)
#print(y)

np.random.seed(0)
#digits_class = y_train.unique()
#n_class = digits_class.size
#y = pd.Categorical(y_train).codes
#y_one_hot = label_binarize(y_test, np.arange(n_class))
#alpha = np.logspace(-2, 2, 20)
#y_score = model.predict_classes(x_test)
metrics.roc_auc_score(y_one_hot, y, average='micro')

fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y.ravel())
auc = metrics.auc(fpr, tpr)
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'ROC and AUC', fontsize=17)
plt.show()
