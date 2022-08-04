from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os
import time
import librosa
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


'''
def load_folds(folds):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print fold_name, "features: ", loaded_features.shape

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True
        
    return features, labels
def evaluate(model):
    y_prob = model.predict_proba(test_x, verbose=0)
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(test_y, 1)

    roc = roc_auc_score(test_y, y_prob)
    print "ROC:",  round(roc,3)

    # evaluate the model
    score, accuracy = model.evaluate(test_x, test_y, batch_size=32)
    print("\nAccuracy = {:.2f}".format(accuracy))

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print "F-Score:", round(f,2)
    
    return roc, accuracy
data_dir = "data/us8k-np-cnn"
# model has never seen fold 10, so use that for testing
test_x, test_y = load_folds([10])
roc, acc = evaluate(model)
print '\nModel R.O.C:', round(roc, 3)
print 'Model Accuracy:', round(acc, 3)
'''
############# test sample audio ###############
sound_file_paths = ["aircon.wav", "carhorn.wav", "play.wav", "dogbark.wav", "drill.wav",
                    "engine.wav","gunshots.wav","jackhammer.wav","siren.wav","music.wav"]
sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]
parent_dir = '../samples/us8k/'

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size // 2)

def extract_features_array(filename, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_clip,s = librosa.load(filename)        
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames)
    features = log_specgrams

    
    return np.array(features)

if __name__ == "__main__":
    # create predictions for each of the sound classes
    model_filepath = "models/salamon-RNNFinal-model.h5"
    model = load_model(model_filepath)

    print('model name:', model_filepath)
    #plot_model(model, show_shapes=True, to_file='models/salamon-cnn-model.svg')
    print(model.summary())
    sum_time = 0
    #Predict_times = 2000
    for s in range(len(sound_names)):
        predict_file = parent_dir + sound_file_paths[s]
        predict_x = extract_features_array(predict_file)
        predictions = model.predict(predict_x)
        
        print("\n----- ", sound_names[s], "-----")
        
        if len(predictions) == 0: 
            print("No prediction")
            continue

        ind = np.argpartition(predictions[0], -2)[-2:]
        ind[np.argsort(predictions[0][ind])]
        ind = ind[::-1]

        print('predictions:',predictions[0])
        
        print("Top guess: ", sound_names[ind[0]], " (",round(predictions[0,ind[0]],3),")")
        print("2nd guess: ", sound_names[ind[1]], " (",round(predictions[0,ind[1]],3),")")
        # print ("3rd guess: ", sound_names[ind[2]], " (",round(predictions[0,ind[2]],3),")")
        # print ("4rd guess: ", sound_names[ind[3]], " (",round(predictions[0,ind[3]],3),")")