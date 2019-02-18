import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
# use tensorflow as back-end
import tensorflow as tf
# use keras as front-end
import keras
from keras.layers.merge import concatenate
from keras.models import Model, Sequential,Input # basic class for specifying and training a neural network
from keras.layers import MaxPooling1D,Input, Conv1D,  Dense, Dropout, Activation, Flatten, Concatenate
from keras import regularizers
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from numpy import newaxis
# show tensorflow version
print("Version tensorflow :" + tf.__version__)
# show keras version
print("Version KERAS :" + keras.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from Embedding import Embedding
path = "./csv_pkuseg"
res = []
res_type = []

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='macro')
        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_f1: %.4f - val_precision: %.4f - val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        return

metrics = Metrics()

# main training code

# CONVNET PARAMETERS
# ===================
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 2500 # we iterate 2500 times over the entire training set
pool_size = 100
drop_prob_1 = 0.5  # dropout after pooling with probability ??
drop_prob_2 = 0.5  # dropout in the FC layer with probability ??
hidden_size = 128  # the FC layer will have 128 neurons
weight_penalty = 0.0 # Factor for weights penalty

# DATASET CHARACTERISTICS
height, width, depth = 200, 320, 1 # data format est 200*320 and channel=1 to adapt ConvNet demand
num_classes = 10 # there are 11 classes

# Set the data path and fetch the data
data_path = './data_wem-posem'
X_train, y_train, X_test, y_test = load_data(data_path)

# REFORMAT PROPERLY THE DATA
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

# NOW, BUILD THE MODEL ARCHITECTURE
# ================================

model_k5 = Sequential()
model_k7 = Sequential()
model_k3 = Sequential()

# this convolutional layer compute the sentence with a word window size of 5
model_k5.add( Conv1D(conv_depth_1,5, border_mode='same', activation='relu'))
model_k5.add( MaxPooling1D(pool_size=pool_size,padding='same') )
model_k5.add( Dropout(drop_prob_1) ) # Some Dropout regularization (if necessary)

# this convolutional layer compute the sentence with a word window size of 3
model_k3.add( Conv1D(conv_depth_1,3, border_mode='same', activation='relu'))
model_k3.add( MaxPooling1D(pool_size=pool_size,padding='same') )
model_k3.add( Dropout(drop_prob_1) ) # Some Dropout regularization (if necessary)

# this convolutional layer compute the sentence with a word window size of 7
model_k7.add( Conv1D(conv_depth_1,7, border_mode='same', activation='relu'))
model_k7.add( MaxPooling1D(pool_size=pool_size,padding='same') )
model_k7.add( Dropout(drop_prob_1) ) # Some Dropout regularization (if necessary)


model_in = Input(shape=(200,320))
merged = concatenate([model_k3(model_in),model_k5(model_in),model_k7(model_in)],axis=1)
model_final = Sequential()

# CLASSIFICATION PART: FULLY-CONNECTED LAYER + OUTPUT LAYER
#   Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
model_final.add( Flatten() )
model_final.add( Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(weight_penalty)) )
model_final.add( Dropout(drop_prob_2) ) # Some Dropout regularization (if necessary)
model_final.add( Dense(num_classes, activation='softmax') )
model = Model(model_in,model_final(merged))

# DEFINE THE LOSS FUNCTION AND OPTIMIZER
model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adadelta',  # using the Adadelta optimiser
              metrics=['accuracy'])  # reporting the accuracy

# TRAIN THE MODEL
history = model.fit(X_train, Y_train,  # Train the model using the training set...
                    batch_size=batch_size, nb_epoch=num_epochs, shuffle=True,
                    verbose=1, validation_split=0.25,
                    callbacks=[metrics])  # ...holding out 30% of the data for validation
print(model.summary())
# EVALUATE THE MODEL ON TEST SET
model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
model.save('./models/1108_model_10types_3kernel.h5')

predict_test = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test.astype(np.int64), predict_test))
print('macro f1:\n')
recall = recall_score(y_test.astype(np.int64), predict_test, average='macro')
f1 = f1_score(y_test.astype(np.int64), predict_test, average='macro')
precision = precision_score(y_test.astype(np.int64), predict_test, average='macro')
print('f1_score: ' + str(f1) + '\n')
print('recall_score: ' + str(recall) + '\n')
print('precision_score: ' + str(precision) + '\n')
print('weighted f1:\n')
recall = recall_score(y_test.astype(np.int64), predict_test, average='weighted')
f1 = f1_score(y_test.astype(np.int64), predict_test, average='weighted')
precision = precision_score(y_test.astype(np.int64), predict_test, average='weighted')
print('f1_score: ' + str(f1) + '\n')
print('recall_score: ' + str(recall) + '\n')
print('precision_score: ' + str(precision) + '\n')



