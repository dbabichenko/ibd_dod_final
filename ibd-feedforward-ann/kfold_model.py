
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
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

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores_acc = []
cvscores_los = []

for train, test in kfold.split(X, Y):
  # create model
    model = Sequential()
    model.add(Dense(74, input_dim=73, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
    
    history = model.fit(X[train], Y[train] , epochs=32, batch_size=128, verbose=0)
   
	# evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("acc = %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("los = %s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    cvscores_acc.append(scores[1] * 100)
    cvscores_los.append(scores[0] * 100)
print("final accuracy : %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_acc), numpy.std(cvscores_acc)))
print("final loss : %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_los), numpy.std(cvscores_los)))


