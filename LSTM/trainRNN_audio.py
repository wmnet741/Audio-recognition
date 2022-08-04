import glob
import os
import librosa
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
#from keras.utils import np_utils
from tensorflow.keras.regularizers import l2

tf.random.set_seed(0)
np.random.seed(0)

frames = 41
bands = 60
feature_size = bands * frames  # 60x41
num_labels = 10
num_channels = 2

# this will aggregate all the training data


def load_all_folds():
    subsequent_fold = False
    for k in range(1, 9):
        fold_name = 'fold' + str(k)
        print("\nAdding " + fold_name)
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print("New Features: ", loaded_features.shape)

        if subsequent_fold:
            train_x = np.concatenate((train_x, loaded_features))
            train_y = np.concatenate((train_y, loaded_labels))
        else:
            train_x = loaded_features
            train_y = loaded_labels
            subsequent_fold = True

    # use the penultimate fold for validation
    valid_fold_name = 'fold9'
    feature_file = os.path.join(data_dir, valid_fold_name + '_x.npy')
    labels_file = os.path.join(data_dir, valid_fold_name + '_y.npy')
    valid_x = np.load(feature_file)
    valid_y = np.load(labels_file)

    # and use the last fold for testing
    test_fold_name = 'fold10'
    feature_file = os.path.join(data_dir, test_fold_name + '_x.npy')
    labels_file = os.path.join(data_dir, test_fold_name + '_y.npy')
    test_x = np.load(feature_file)
    test_y = np.load(labels_file)
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    #model.fit(train_x, train_y, validation_data=(valid_x, valid_y),callbacks=[earlystop], batch_size=32, epochs=1)

# this is used to load the folds incrementally
def load_folds(folds):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print(fold_name, "features: ", loaded_features.shape)

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True

    return features, labels


def evaluate(model):
    y_prob = model.predict(test_x, verbose=0)
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(test_y, 1)

    roc = roc_auc_score(test_y, y_prob)
    print("ROC:",  round(roc, 3))

    # evaluate the model
    score, accuracy = model.evaluate(test_x, test_y, batch_size=32)
    print("\nAccuracy = {:.2f}".format(accuracy))

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average='micro')
    print("F-Score:", round(f, 2))

    return roc, accuracy


def build_model():
    ####RNN######
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(bands, frames))) 
    model.add(Dropout(0.5))
    
    model.add(LSTM(16)) 
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    
    return model

def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)

if __name__ == "__main__":
    data_dir = "data/us8k-np-cnn"
    all_folds = True
    av_acc = 0.
    av_roc = 0.
    num_folds = 0

    # as we use two folds for training, there are 9 possible trails rather than 10
    max_trials = 5

    # earlystopping ends training when the validation loss stops improving
    earlystop = EarlyStopping(
        monitor='val_loss', patience=0, verbose=0, mode='auto')

    if all_folds:
        #load_all_folds()
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_all_folds()
        # new_train_x = train_x[:,:,:,0]
        # new_valid_x = valid_x[:,:,:,0]

        print("Building model...")
        model = build_model()
        model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=32, epochs=50)
        # model.fit(new_train_x, train_y, validation_data=(new_valid_x, valid_y), batch_size=32, epochs=2)
        print("Evaluating model...")
        roc, acc = evaluate(model)
        av_roc += roc
        av_acc += acc
    else:
        # use folds incrementally
        for f in range(1, max_trials+1):
            num_folds += 1
            v = f + 2
            if v > 10:
                v = 1
            t = v + 1
            if t > 10:
                t = 1

            print("\n*** Train on", f, "&", (f+1),
                  "Validate on", v, "Test on", t, "***")

            # load two folds for training data
            train_x, train_y = load_folds([f, f+1])

            # load one fold for validation
            valid_x, valid_y = load_folds([v])

            # load one fold for testing
            test_x, test_y = load_folds([t])

            print("Building model...")
            model = build_model()

            # now fit the model to the training data, evaluating loss against the validation data
            print("Training model...")
            model.fit(train_x, train_y, validation_data=(
                valid_x, valid_y), callbacks=[earlystop], batch_size=64, epochs=2)

            # now evaluate the trained model against the unseen test data
            print("Evaluating model...")
            roc, acc = evaluate(model)
            av_roc += roc
            av_acc += acc

    print('\nAverage R.O.C:', round(av_roc/max_trials, 3))
    print('Average Accuracy:', round(av_acc/max_trials, 3))
    # save model
    assure_path_exists("models")
    filepath = "models/salamon-RNNFinal-model.h5"
    model.save(filepath)