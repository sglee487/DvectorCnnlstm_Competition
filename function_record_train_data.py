# NOTE: this example requires PyAudio because it uses the Microphone class
# import speech_recognition as sr
# obtain audio from the microphone
from time import sleep

import pickle
import numpy as np
import librosa
import os

from python_speech_features import logfbank, fbank
from feature_extraction_scripts.feature_extraction_functions import get_stft, get_cqt,get_mfcc
from feature_extraction_scripts.custom_functions import normalize_frames

import function_getSentences


def main():
    import speech_recognition as sr
    r = sr.Recognizer()

    print("훈련 데이터 녹음하기")
    file_name = str(input("이름을 입력하세요 : "))

    function_getSentences.main()
    sentences20 = function_getSentences.get20SentencesList()

    for i in range(1,11):

        with sr.Microphone() as source:
            print("잠시 후 20초 동안 녹음을 시작합니다.")
            print("녹음이 시작되면 다음 문장을 읽어주세요.")
            print(sentences20[i-1])
            sleep(4)
            r.adjust_for_ambient_noise(source)
            print("녹음 시작")
            audio = r.listen(source,phrase_time_limit=20)


        result_name = file_name + "_" + str(i)
        print(result_name)

        save_path = "./train_data/"
        if not os.path.exists(save_path + file_name):
            os.makedirs(save_path + file_name)
        # write audio to a WAV file
        with open(save_path + file_name + "/" + result_name+".wav", "wb") as f:
            f.write(audio.get_wav_data())
            print(result_name + " : Finish !")

        if (i == 10):
            save_path = "./enroll_data/"
            if not os.path.exists(save_path + file_name):
                os.makedirs(save_path + file_name)
            # write audio to a WAV file
            with open(save_path + file_name + "/" + result_name+".wav", "wb") as f:
                f.write(audio.get_wav_data())
                print(result_name + " : Finish !")

            print("이제 파일을 feat_logfbank_nfilt40/test/" + file_name + "/enroll.p 로 변환하여 저장합니다.")

            p_file_path = "./feat_logfbank_nfilt40"

            if not os.path.exists(p_file_path):
                os.makedirs(p_file_path)
            if not os.path.exists(p_file_path + "/test"):
                os.makedirs(p_file_path + "/test")

            f = os.listdir(save_path + file_name)[0]
            audio, sr = librosa.load(save_path + file_name + "/" + f, sr=16000, mono=True)

            filter_banks, energies = fbank(audio, samplerate=16000, nfilt=40, winlen=0.025)
            filter_banks = 20 * np.log10(np.maximum(filter_banks, 1e-5))
            feature = normalize_frames(filter_banks, Scale=False)

            info = {"feat": feature, "label": file_name}

            if not os.path.exists(p_file_path + "/test/" + file_name):
                os.makedirs(p_file_path + "/test/" + file_name)

            pickle.dump(info,
                        open(p_file_path + "/test/" + file_name + "/enroll.p", "wb"))

            print("enroll.p 저장 완료.\n")



if __name__ == '__main__':
    main()