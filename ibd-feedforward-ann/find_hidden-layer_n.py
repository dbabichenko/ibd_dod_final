
from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import model_to_dot
import numpy
import csv
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("data.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:73]
Y = dataset[:,73]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

for x in range(1, 75):
    model = Sequential()
    model.add(Dense(74, input_dim=73, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(x, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, validation_split=0.1, epochs=32, batch_size=64, verbose=0)
    scores = model.evaluate(X, Y, verbose=0)
    acc = scores[1]
    los = scores[0]
    with open('res.csv', mode='a') as ffile:
        wwriter = csv.writer(ffile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        wwriter.writerow([str(x) , str(acc) , str(los)])

