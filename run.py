import function_record_train_data
import train_models_CNN_LSTM_CNNLSTM_custom
import train
import function_enroll
import enroll
import implement_model_dvector_CNNLSTM_custom
import function_ShowEnrollments
import function_enroll_testrecord
import extract_features


def main():
    print("프로그램 시작.")

    while True:
        print("기능을 선택하세요.")
        print("1. 훈련데이터 음성 제작(10번 녹음) 2. 훈련 3. 테스트 4. 훈련/등록자확인 5. 종료")
        choose = int(input())
        if choose == 1:
            print("훈련용 음성을 저장합니다. 음성은 'train_data/(이름) 에 저장됩니다.")
            function_record_train_data.main()
            # train.main()
        elif choose == 2:
            print("모델 훈련")
            enroll.main()
            # variables to set:

            data_path = "./train_data"
            limit = False  # Options: False or fraction of data to be extracted
            feature_type = "stft"  # "mfcc" "fbank" "stft"
            num_filters = 40  # Options: 40, 20, 13, None
            delta = False  # Calculate the 1st and 2nd derivatives of features?
            dom_freq = False  # Kinda sorta... Pitch (dominant frequency)
            noise = False  # Add noise to speech data?
            vad = True  # Apply voice activity detection (removes the beginning and ending 'silence'/background noise of recordings)
            timesteps = 5
            context_window = 5
            noise_path = ""
            date_folder = extract_features.main(data_path, feature_type,
                                                num_filters=num_filters, delta=delta, noise=noise, vad=vad,
                                                dom_freq=dom_freq,
                                                timesteps=timesteps, context_window=context_window,
                                                noise_path=noise_path, limit=limit
                                                )

            f = open("model_date_name.txt", 'a')
            f.write(date_folder)
            f.write("\n")
            f.close()

            model_type = "cnnlstm"  # cnn, lstm, cnnlstm
            epochs = 100
            optimizer = 'sgd'  # 'adam' 'sgd'
            sparse_targets = True
            patience = 5
            train_models_CNN_LSTM_CNNLSTM_custom.main(model_type, epochs, optimizer, sparse_targets, patience,
                                                      date_folder)
        elif choose == 3:
            print()

            f = open("model_date_name.txt",'r')
            lines = f.readlines()
            f.close()

            date_folder = lines[lines.__len__()-1]
            date_folder = date_folder[0:-1]
            print(date_folder)

            project_head_folder = date_folder
            model_name = "CNNLSTM_speech_commands001"
            implement_model_dvector_CNNLSTM_custom.main(project_head_folder, model_name)
        elif choose == 4:
            print()
            function_ShowEnrollments.main()
        elif choose == 5:
            return None
        else:
            print("번호를 다시 입력해주세요.")
    return None

if __name__ == '__main__':
    main()