import os
import csv
import numpy as np
import sounddevice as sd
import librosa
from keras.models import load_model as load_model_CNNLSTM

import feature_extraction_scripts.feature_extraction_functions as featfun
import feature_extraction_scripts.prep_noise as pn

import torch
import torch.nn.functional as F

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet

import speech_recognition as sr
import pickle
import numpy as np

from python_speech_features import logfbank, fbank
from feature_extraction_scripts.feature_extraction_functions import get_stft,get_mfcc,get_cqt

from feature_extraction_scripts.custom_functions import normalize_frames

import datetime

def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day,time.hour,time.minute,time.second)
    return(time_str)

def record_sound(sec,message):
    sr = 16000
    print(message+" for {} seconds..".format(sec))
    sound = sd.rec(int(sec*sr),samplerate=sr,channels=1)
    sd.wait()
    return sound, sr

def str2bool(bool_string):
    bool_string = bool_string=="True"
    return bool_string

def load_model_dvector(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num) + '.pth')
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()

    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]

    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB


def load_enroll_embeddings(embedding_dir):
    embeddings = {}
    for f in os.listdir(embedding_dir):
        spk = f.replace('.pth', '')
        # Select the speakers who are in the 'enroll_spk_list'
        embedding_path = os.path.join(embedding_dir, f)
        tmp_embeddings = torch.load(embedding_path)
        embeddings[spk] = tmp_embeddings

    return embeddings


def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename)  # input size:(n_frames, n_dims)

    tot_segments = math.ceil(len(input) / test_frames)  # total number of segments with 'test_frames'
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i * test_frames:i * test_frames + test_frames]

            TT = ToTensorTestInput()
            temp_input = TT(temp_input)  # size:(1, 1, n_dims, n_frames)

            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation, _ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)

    activation = l2_norm(activation, 1)

    return activation


def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output


def perform_identification(use_cuda, model, embeddings, test_filename, test_frames, spk_list,file_name):
    # print(embeddings)
    print(test_filename)
    # print(model)
    # print(test_frames)
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)
    max_score = -10 ** 8
    best_spk = None
    score_array = []
    for spk in spk_list:
        score = F.cosine_similarity(test_embedding, embeddings[spk])
        score = score.data.cpu().numpy()
        print("speaker : " + str(spk) + " , Score : " + str(score[0]))
        score_array.append(score[0])
        if score > max_score:
            max_score = score
            best_spk = spk
    # print("Speaker identification result : %s" %best_spk)
    true_spk = file_name
    print("\n=== Speaker identification ===")
    print("True speaker : %s" % (true_spk))
    return score_array

def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day, time.hour, time.minute, time.second)
    return (time_str)

def record_sound(file_name, path):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say Bed-Bird-Cat-Dog-Down!")
        audio = r.listen(source, phrase_time_limit=5)
    timestamp = get_date()
    result_name = str(file_name) + "_" + str(timestamp)
    print(result_name)

    # write audio to a WAV file
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "/" + result_name + ".wav", "wb") as f:
        f.write(audio.get_wav_data())
        print(result_name + " : Finish !")
        return result_name

def main(project_head_folder,model_name):
    log_dir = 'model_saved'  # Where the checkpoints are saved
    embedding_dir = 'enroll_embeddings'  # Where embeddings are saved
    test_dir = 'feat_logfbank_nfilt40/test/'  # Where test features are saved

    spk_list = []
    people_list = os.listdir(embedding_dir)

    for file in people_list:
        spk_list.append(file.split('.')[0])
    print("등록 되어있는 사용자들")
    print(spk_list)
    print("훈련 되어있는 사용자들")
    people_train_list = os.listdir("feat_logfbank_nfilt40/train/")
    print(people_train_list)

    recording_folder = "./recording_mic"
    # file_name = str(input("What is your Name?"))
    # if not(any(file_name in spk_list for people_name in spk_list)):
    #     print("존재하지 않는 사용자.")
    #     return None
    # result_file = record_sound(file_name, recording_folder)

    # speech_filename = recording_folder + "/" + result_file + ".wav"

    # === FOR CNNLSTM ===
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

    # recording_folder = "{}/recordings".format(head_folder_curr_project)

    # === FOR CNNLSTM END ===

    # print("훈련되어있는 방법을 선택하세요.")
    # print("1. fbank")
    # print("2. stft")
    # print("3. mfcc")
    # print("4. cqt")
    # print("번호 입력")
    # train_way = int(input())
    train_way = 1

    spk_list.remove('103F3021')
    spk_list.remove('207F2088')
    spk_list.remove('213F5100')
    spk_list.remove('217F3038')
    spk_list.remove('225M4062')
    spk_list.remove('229M2031')
    spk_list.remove('230M4087')
    spk_list.remove('233F4013')
    spk_list.remove('236M3043')
    spk_list.remove('240M3063')

    identification_try_success = 0
    identification_try_fail = 0
    identification_try_total = 0

    verification_try_success = 0
    verification_try_fail = 0
    verification_try_total = 0
    for speech_filename in os.listdir("recording_mic"):
        file_name = speech_filename
        speech_filename = recording_folder + "/" + speech_filename
        print(speech_filename)

        person_name = speech_filename.split("/")[2].split("_")[0]
        print(person_name)

        is_he_on_list = False
        print("is he in LIST?")
        if (person_name in spk_list):
            is_he_on_list = True
            print("YES")
        else:
            is_he_on_list = False
            print("NO")
        y_speech, sr = librosa.load(speech_filename, sr=16000, mono=True)

        print("sound dB : " + str(np.mean(np.sqrt(np.square(y_speech)))))
        if (np.mean(np.sqrt(np.square(y_speech)))) < 0.001:
            print("Too LOW dB.")
            print("GMM check end with FILTER")
        #     return None

        if train_way == 1:
            filter_banks, energies = fbank(y_speech, samplerate=16000, nfilt=40, winlen=0.025)
            filter_banks = 20 * np.log10(np.maximum(filter_banks, 1e-5))
            feature = normalize_frames(filter_banks, Scale=False)
        elif train_way == 2:
            feature = stft = get_stft(y_speech, sr)

        elif train_way == 3:
            feature = np.asarray(())
            vector = get_mfcc(y_speech, sr)

            if feature.size == 0:
                feature = vector
            else:
                feature = np.vstack((feature, vector))

        elif train_way == 4:
            feature = np.asarray(())
            vector = get_cqt(y_speech, sr)

            if feature.size == 0:
                feature = vector
            else:
                feature = np.vstack((feature, vector))

        infor = {"feat": feature, "label": "temp"}
        if not os.path.exists(test_dir + "temp"):
            os.makedirs(test_dir + "temp")
        pickle.dump(infor, open(test_dir + "temp/test.p", "wb"))

        # Settings
        use_cuda = True  # Use cuda or not
        embedding_size = 128  # Dimension of speaker embeddings
        cp_num = 24  # Which checkpoint to use?
        n_classes = 240  # How many speakers in training data?
        test_frames = 100  # Split the test utterance

        # Load model from checkpoint
        model = load_model_dvector(use_cuda, log_dir, cp_num, embedding_size, n_classes)

        # Get the dataframe for test DB
        enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_DIR)

        # Load enroll embeddings
        embeddings = load_enroll_embeddings(embedding_dir)

        """ Test speaker list
        '103F3021', '207F2088', '213F5100', '217F3038', '225M4062', 
        '229M2031', '230M4087', '233F4013', '236M3043', '240M3063'
        """

        # Set the test speaker
        test_speaker = "temp"

        test_path = os.path.join(test_dir, test_speaker, 'test.p')


        # Perform the test
        score_array = perform_identification(use_cuda, model, embeddings, test_path, test_frames, spk_list, file_name)

        features = featfun.coll_feats_manage_timestep(timesteps, frame_width, speech_filename, feature_type,
                                                      num_filters,
                                                      num_feature_columns, recording_folder, delta=delta,
                                                      dom_freq=dom_freq,
                                                      noise_wavefile=None, vad=vad)

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
        model = load_model_CNNLSTM(model_path)

        prediction = model.predict(X)

        print(prediction)
        print(max(prediction[0]))

        identification_try_total += 1
        print("score_array의 표준편차 : " + str(np.std(score_array)))
        if (max(score_array)<0.987):
            print("점수가 너무 낮아 등록되지 않은 사람으로 판정.\n\n")
            if is_he_on_list:
                identification_try_fail += 1
            else:
                identification_try_success+=1
            continue
        if (np.std(score_array) < 0.0225):
            print("점수가 서로 비슷하여 등록되지 않은 사람으로 판정.\n\n")
            if is_he_on_list:
                identification_try_fail += 1
            else:
                identification_try_success += 1
            continue

        if is_he_on_list:
            identification_try_success += 1
        else:
            identification_try_fail += 1

        if (max(prediction[0]) < 0.5):
            print("This person is not permitted maxscore 0.5")
            print("end with FILTER")

            if is_he_on_list:
                identification_try_fail += 1
            else:
                identification_try_success += 1

            continue

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
    print("identification 성공횟수: {}, 실패횟수: {}, 총 시도 횟수: {}, 성공률: {}%".format(identification_try_success,identification_try_fail,identification_try_total,(identification_try_success/identification_try_total)*100))
    print("verification 성공횟수: {}, 실패횟수: {}, 총 시도 횟수: {}, 성공률: {}%".format(verification_try_success,
                                                                            verification_try_fail,
                                                                            verification_try_total, (
                                                                                        verification_try_success / verification_try_total) * 100))

    return None

if __name__=="__main__":
    
    project_head_folder = "stft201_models_2019y11m8d15h3m46s"
    model_name = "CNNLSTM_speech_commands001"
    
    main(project_head_folder,model_name)
        
    
            
