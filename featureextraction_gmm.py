import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import librosa

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)
    mfcc_feature_before_preprocessing = mfcc_feature;
    mfcc_feature = preprocessing.scale(mfcc_feature) # from sklearn import preprocessing
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def get_stft(y, sr, window_size=None, window_shift=None):
    if window_size is None:
        n_fft = int(0.025 * sr)
    else:
        n_fft = int(window_size * 0.001 * sr)
    if window_shift is None:
        hop_length = int(0.010 * sr)
    else:
        hop_length = int(window_shift * 0.001 * sr)
    stft = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length))  # comes in complex numbers.. have to take absolute value
    stft = np.transpose(stft)
    stft -= (np.mean(stft, axis=0) + 1e-8)

    return stft