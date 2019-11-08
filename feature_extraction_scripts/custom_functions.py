import numpy as np
import datetime
import os
from feature_extraction_scripts.feature_extraction_functions import get_stft, get_cqt,get_mfcc



def normalize_frames(m,Scale=True):

    if Scale:

        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)

    else:

        return (m - np.mean(m, axis=0))


def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day, time.hour, time.minute, time.second)
    return (time_str)


def train_record_converter(person_name):
    from python_speech_features import logfbank, fbank
    import scipy.io.wavfile as wav
    from python_speech_features import mfcc
    import pickle
    import numpy as np
    import librosa
    import os

    from feature_extraction_scripts.feature_extraction_functions import get_stft
    from feature_extraction_scripts.custom_functions import normalize_frames

    import feature_extraction_scripts.organize_speech_data as orgdata
    data_path = "./train_data"
    paths, labels_wavefile = orgdata.collect_audio_and_labels(data_path + "/" + person_name)

    p_file_path = "./feat_logfbank_nfilt40"

    if not os.path.exists(p_file_path):
        os.makedirs(p_file_path)

    count = 1
    print(paths)

    print("특징을 추출할 방법을 선택하세요.")
    print("1. fbank")
    print("2. stft")
    print("3. mfcc")
    print("4. cqt")

    print("번호 입력")
    train_way = int(input())

    for path in paths:
        path = str(path).strip()
        print(path)

        person_name = path.split("/")[1]
        print(person_name)

        audio, sr = librosa.load(path, sr=16000, mono=True)

        if train_way == 1:
            filter_banks, energies = fbank(audio, samplerate=16000, nfilt=40, winlen=0.025)
            filter_banks = 20 * np.log10(np.maximum(filter_banks, 1e-5))
            feature = normalize_frames(filter_banks, Scale=False)
        elif train_way == 2:
            feature = stft = get_stft(audio, sr)
        elif train_way == 3:
            feature = np.asarray(())
            vector = get_mfcc(audio, sr)

            if feature.size == 0:
                feature = vector
            else:
                feature = np.vstack((feature, vector))
        elif train_way == 4:
            feature = np.asarray(())
            vector = get_cqt(audio, sr)

            if feature.size == 0:
                feature = vector
            else:
                feature = np.vstack((feature, vector))

        info = {"feat": feature, "label": person_name}

        if not os.path.exists(p_file_path + "/train/" + person_name):
            os.makedirs(p_file_path + "/train/" + person_name)

        pickle.dump(info, open(p_file_path + "/train/" + person_name + "/" + person_name +  "_" + str(count) + ".p", "wb"))
        print(p_file_path + "/train/" + person_name + "/" + person_name +  "_" + str(count) + ".p" + " 로 저장됨.")

        if (count == 15):

            count = 1
        else:
            count = count + 1