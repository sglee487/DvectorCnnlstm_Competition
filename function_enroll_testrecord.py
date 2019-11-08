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


def main():
    import speech_recognition as sr
    r = sr.Recognizer()

    print("등록테스트 음성")
    file_name = str(input("이름을 입력하세요 : "))

    with sr.Microphone() as source:
        #     print("잠시 후 20초 동안 녹음을 시작합니다.")
        #     print("녹음이 시작되면 다음 문장을 읽어주세요.")
        # print('''앞으로 로봇 수요가 증가하면서 로봇 시장의 우위를 선점하기
        # 위한 로봇 기술 개발의 경쟁이 더욱 뜨거워질 것이다. 로봇 기술
        # 중 상당수가 특허권이 인정되는 고부가 가치 기술이기 때문이다.
        # 이러한 상황에서 전문가들은 로봇세를 도입하면 기술 개발에
        # 악영향을 끼칠 수 있다고 말한다. 로봇세를 도입하면 세금에
        # 대한 부담이 늘어나 로봇에 대한 수요가 감소한다. 그렇게 되면
        # 로봇을 생산하는 기업은 기술 개발 의지가 약화되어 로봇 기술
        # 의 특허권으로 이익을 창출할 수 있는 기회가 줄어들게 된다.
        # 그래서 로봇 사용이 필요한 기업이나 개인은 선진 로봇 기술이
        # 적용된 로봇을 외국에서 수입해야 하므로 막대한 금액이 외부로
        # 유출되어 국가적으로 손해이다.''')
        #     sleep(6)
        # r.adjust_for_ambient_noise(source)
        #     print("녹음 시작")
        audio = r.listen(source,phrase_time_limit=20)

    result_name = file_name
    print(result_name)

    save_path = "./test_data/"
    if not os.path.exists(save_path + file_name):
        os.makedirs(save_path + file_name)
    # write audio to a WAV file
    with open(save_path + file_name + "/" + result_name+".wav", "wb") as f:
        f.write(audio.get_wav_data())
        print(result_name + " : Finish !")

    print("이제 파일을 feat_logfbank_nfilt40/test/" + file_name + "/test.p 로 변환하여 저장합니다.")

    p_file_path = "./feat_logfbank_nfilt40"

    if not os.path.exists(p_file_path):
        os.makedirs(p_file_path)
    if not os.path.exists(p_file_path+"/test"):
        os.makedirs(p_file_path+"/test")

    f = os.listdir(save_path + file_name)[0]
    audio, sr = librosa.load(save_path + file_name + "/" + f, sr=16000, mono=True)
    # audio, sr = librosa.load(save_path + file_name + "/" + result_name+".wav", sr=16000, mono=True)

    # print("등록시킬 방법을 선택하세요.")
    # print("1. fbank")
    # print("2. stft")
    # print("3. mfcc")
    # print("4. cqt")
    #
    # print("번호 입력")
    # train_way = int(input())
    train_way = 1

    if train_way == 1 :
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

    info = {"feat": feature, "label": file_name}

    if not os.path.exists(p_file_path + "/test/" + file_name):
        os.makedirs(p_file_path + "/test/" + file_name)

    pickle.dump(info,
                open(p_file_path + "/test/" + file_name + "/test.p", "wb"))

    print("test.p 저장 완료.\n")


if __name__ == '__main__':
    main()