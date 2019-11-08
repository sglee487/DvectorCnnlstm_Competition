import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import datetime
import numpy as np
from keras.models import load_model

import feature_extraction_scripts.feature_extraction_functions as featfun
import speech_recognition as sr


def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day, time.hour, time.minute, time.second)
    return (time_str)


def str2bool(bool_string):
    bool_string = bool_string == "True"
    return bool_string


def record_sound(file_name, path):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say Bed-Bird-Cat-Dog-Down!")
        audio = r.listen(source, phrase_time_limit=5)

    result_name = file_name
    print(result_name)
    # write audio to a WAV file
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "/" + result_name + ".wav", "wb") as f:
        f.write(audio.get_wav_data())
        print(result_name + " : Finish !")


def main(project_head_folder, model_name):
    log_dir = 'model_saved'  # Where the checkpoints are saved
    embedding_dir = 'enroll_embeddings'  # Where embeddings are saved
    test_dir = 'feat_logfbank_nfilt40/test/'  # Where test features are saved

    spk_list = []
    people_list = os.listdir(embedding_dir)

    for file in people_list:
        spk_list.append(file.split('.')[0])
    print("등록 되어있는 사용자들")
    print(spk_list)

    head_folder_beg = "./ml_speech_projects/"
    head_folder_curr_project = head_folder_beg + project_head_folder

    # load the information related to features and model of interest
    features_info_path = head_folder_curr_project + "/features_log.csv"
    encoded_label_path = head_folder_curr_project + "/labels_encoded.csv"
    model_path = head_folder_curr_project + "/models/{}.h5".format(model_name)
    model_log_path = head_folder_curr_project + "/model_logs/{}.csv".format(model_name)

    # find out the settings for feature extraction
    with open(features_info_path, mode='r') as infile:
        reader = csv.reader(infile)
        feats_dict = {rows[0]: rows[1] for rows in reader}
    feature_type = feats_dict['features']
    num_filters = int(feats_dict['num original features'])
    num_feature_columns = int(feats_dict['num total features'])
    delta = str2bool(feats_dict["delta"])
    dom_freq = str2bool(feats_dict["dominant frequency"])
    noise = str2bool(feats_dict["noise"])
    vad = str2bool(feats_dict["beginning silence removal"])
    timesteps = int(feats_dict['timesteps'])
    context_window = int(feats_dict['context window'])
    frame_width = context_window * 2 + 1

    # prepare the dictionary to find out the assigned label
    with open(encoded_label_path, mode='r') as infile:
        reader = csv.reader(infile)
        dict_labels_encoded = {rows[0]: rows[1] for rows in reader}

    print(dict_labels_encoded)
    print("\nAvailable labels:")
    for key, value in dict_labels_encoded.items():
        print(value)

    recording_folder = "./recording_mic"


    identification_try_success = 0
    identification_try_fail = 0
    identification_try_total = 0

    verification_try_success = 0
    verification_try_fail = 0
    verification_try_total = 0


    for speech_filename in os.listdir("recording_mic"):
        # collect new speech
        # file_name = str(input("What is your Name?"))

        # record_sound(file_name, recording_folder)

        # save sound
        # if not os.path.exists(recording_folder):
        #     os.makedirs(recording_folder)
        # speech_filename = recording_folder + "/" + file_name + ".wav"
        speech_filename = recording_folder + "/" + speech_filename
        person_name = speech_filename.split("/")[2].split("_")[0]

        print("")
        print(person_name)
        is_he_on_list = False

        print("is he in LIST?")
        if (person_name in spk_list):
            is_he_on_list = True
            print("YES")
        else:
            is_he_on_list = False
            print("NO")


        features = featfun.coll_feats_manage_timestep(timesteps, frame_width, speech_filename, feature_type,
                                                      num_filters,
                                                      num_feature_columns, recording_folder, delta=delta,
                                                      dom_freq=dom_freq,
                                                      noise_wavefile=None, vad=vad)

        # find out which models:
        with open(model_log_path, mode='r') as infile:
            reader = csv.reader(infile)
            dict_model_settings = {rows[0]: rows[1] for rows in reader}

        model_type = dict_model_settings["model type"]
        activation_output = dict_model_settings["activation output"]

        X = features
        if model_type == "lstm":
            X = X.reshape((timesteps, frame_width, X.shape[1]))
        elif model_type == "cnn":
            X = X.reshape((X.shape[0], X.shape[1], 1))
            X = X.reshape((1,) + X.shape)
        elif model_type == "cnnlstm":
            X = X.reshape((timesteps, frame_width, X.shape[1], 1))
            X = X.reshape((1,) + X.shape)

        # load model
        model = load_model(model_path)

        prediction = model.predict(X)

        identification_try_total += 1
        print(prediction)
        print(max(prediction[0]))
        if (max(prediction[0]) < 0.5):
            print("This person is not permitted maxscore 0.5")
            print("end with FILTER")

            if is_he_on_list:
                identification_try_fail += 1
            else:
                identification_try_success += 1

            continue
            
        if is_he_on_list:
            identification_try_success += 1
        else:
            identification_try_fail += 1

        pred = str(np.argmax(prediction[0]))  # argmax함수로 가장 큰 값 추출
        print(speech_filename)
        label = dict_labels_encoded[pred]


        verification_try_total += 1

        print("Label without noise reduction: {}".format(label))
        if label == person_name:
            verification_try_success += 1
        else:
            verification_try_fail += 1

        print("identification 성공횟수: {}, 실패횟수: {}, 총 시도 횟수: {}, 성공률: {}%".format(identification_try_success,
                                                                                identification_try_fail,
                                                                                identification_try_total, (
                                                                                        identification_try_success / identification_try_total) * 100))
        print("verification 성공횟수: {}, 실패횟수: {}, 총 시도 횟수: {}, 성공률: {}%".format(verification_try_success,
                                                                              verification_try_fail,
                                                                              verification_try_total, (
                                                                                      verification_try_success / verification_try_total) * 100))
        print("=================")
        print("=================")
        print("")

    print("총 합계 계산..")
    print("identification 성공횟수: {}, 실패횟수: {}, 총 시도 횟수: {}, 성공률: {}%".format(identification_try_success,
                                                                            identification_try_fail,
                                                                            identification_try_total, (
                                                                                        identification_try_success / identification_try_total) * 100))
    print("verification 성공횟수: {}, 실패횟수: {}, 총 시도 횟수: {}, 성공률: {}%".format(verification_try_success,
                                                                          verification_try_fail,
                                                                          verification_try_total, (
                                                                                  verification_try_success / verification_try_total) * 100))

    return None

if __name__ == "__main__":
    project_head_folder = "stft201_models_2019y11m5d16h20m48s"
    model_name = "CNNLSTM_speech_commands001"

    main(project_head_folder, model_name)