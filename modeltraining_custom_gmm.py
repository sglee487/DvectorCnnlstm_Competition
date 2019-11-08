import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
# from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from featureextraction_gmm import extract_features
from featureextraction_gmm import get_stft
#from speakerfeatures import extract_features
import warnings
import librosa
warnings.filterwarnings("ignore")

#path to training data
# source   = "development_set/"
# source   = "trainingData/"
# source   = "data/"

#path where training speakers will be saved

# dest = "speaker_models/"
# train_file = "development_set_enroll.txt"

dest = "Speakers_models/"
# train_file = "trainingDataPath.txt"
# file_paths = open(train_file,'r')

import feature_extraction_scripts.organize_speech_data as orgdata
data_path = "./train_data"
paths, labels_wavefile = orgdata.collect_audio_and_labels(data_path)
import os

count = 1
# print(paths[0])
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())

# for path in paths:
#     print(str(path))
#     print(str(path).split("/")[1].split("-")[0])

# for path in file_paths:
for path in paths:
    path = str(path).strip()
    print(path)
    y, sr = librosa.load(path)
    vector = extract_features(y,sr)
    # vector = get_stft(y,sr=16000)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    # print features
    if count == 10:
        # gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
        # gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm = BayesianGaussianMixture(n_components = 32, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # dumping the trained gaussian model
        # picklefile = path.split("-")[0]+".gmm"
        picklefile = str(path).split("/")[1].split("-")[0] + ".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
