from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import gzip,  cPickle

with gzip.open('/home/aurora/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz','rb') as f:
    train_set, validate_set, test_set = cPickle.load(f)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=20, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(train_set[0], train_set[1], nb_epoch=20, batch_size=16)
score = model.evaluate(test_set[0], test_set[1], batch_size=16)