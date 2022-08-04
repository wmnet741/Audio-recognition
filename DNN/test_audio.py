from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os
import time
import librosa
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


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
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features)

if __name__ == "__main__":
    # create predictions for each of the sound classes
    model_filepath = "models/salamon-dnnFinal50-model.h5"
    model = load_model(model_filepath)
    print('model_name:', model_filepath)
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

        ind = np.argpartition(predictions[0], -3)[-3:]
        ind[np.argsort(predictions[0][ind])]
        ind = ind[::-1]
        
        print('Predictions:', predictions[0])
        print("Top guess: ", sound_names[ind[0]], " (",round(predictions[0,ind[0]],3),")")
        print("2nd guess: ", sound_names[ind[1]], " (",round(predictions[0,ind[1]],3),")")
        print ("3rd guess: ", sound_names[ind[2]], " (",round(predictions[0,ind[2]],3),")")
        #print "4rd guess: ", sound_names[ind[3]], " (",round(predictions[0,ind[3]],3),")"