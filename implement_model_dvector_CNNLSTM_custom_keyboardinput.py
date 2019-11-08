import os
import csv
import datetime
import numpy as np
import sounddevice as sd
# import soundfile as sf
import librosa
from keras.models import load_model as load_model_CNNLSTM

import feature_extraction_scripts.feature_extraction_functions as featfun
import feature_extraction_scripts.prep_noise as pn

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet

import speech_recognition as sr
import librosa
import pickle
import numpy as np

from python_speech_features import logfbank, fbank
from feature_extraction_scripts.feature_extraction_functions import get_stft,get_mfcc,get_cqt

from feature_extraction_scripts.custom_functions import normalize_frames

import datetime

import function_getSentences
from time import sleep

# def record_sound(sec,message):
#     sr = 16000
#     print(message+" for {} seconds..".format(sec))
#     sound = sd.rec(int(sec*sr),samplerate=sr,channels=1)
#     sd.wait()
#     return sound, sr

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

    function_getSentences.main()
    sentence = function_getSentences.getRandomSentence()

    with sr.Microphone() as source:
        print("잠시 후 20초 동안 녹음을 시작합니다.")
        print("녹음이 시작되면 다음 문장을 읽어주세요.")
        print(sentence)
        sleep(4)
        r.adjust_for_ambient_noise(source)
        print("녹음 시작")
        audio = r.listen(source, phrase_time_limit=20)

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
    print("등록 되어있는 사용자들")
    print(spk_list)

    recording_folder = "./recording_mic"
    result_file = str(input("input file name : "))

    file_name = speech_filename = recording_folder + "/" + result_file

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

    y_speech, sr = librosa.load(speech_filename, sr=16000, mono=True)

    print("sound dB : " + str(np.mean(np.sqrt(np.square(y_speech)))))
    if (np.mean(np.sqrt(np.square(y_speech)))) < 0.001:
        print("Too LOW dB.")
        print("GMM check end with FILTER")
        return None


    filter_banks, energies = fbank(y_speech, samplerate=16000, nfilt=40, winlen=0.025)
    filter_banks = 20 * np.log10(np.maximum(filter_banks, 1e-5))
    feature = normalize_frames(filter_banks, Scale=False)


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

    print("score_array의 표준편차 : " + str(np.std(score_array)))
    if (max(score_array)<0.948):
        print("점수가 너무 낮아 등록되지 않은 사람으로 판정.\n\n")
        return None

    if (np.std(score_array) < 0.0225):
        print("점수가 서로 비슷하여 등록되지 않은 사람으로 판정.\n\n")
        return None

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


    if (max(prediction[0]) < 0.5):
        print("cnnlstm 점수가 너무 낮아 등록되지 않은 사람으로 판정.")
        print("end with FILTER")
        return None


    pred = str(np.argmax(prediction[0]))  # argmax함수로 가장 큰 값 추출
    print(speech_filename)
    label = dict_labels_encoded[pred]

    print("Label without noise reduction: {}".format(label))




    return None

if __name__=="__main__":
    
    project_head_folder = "stft201_models_2019y11m8d15h3m46s"
    model_name = "CNNLSTM_speech_commands001"
    
    main(project_head_folder,model_name)
        
    
            
