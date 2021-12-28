
from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("data.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:73]
Y = dataset[:,73]

# create model
model = Sequential()
model.add(Dense(73, input_dim=73, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(53, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

history = model.fit(X, Y, validation_split=0.1, epochs=32, batch_size=64, verbose=0)
# list all data in history

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
