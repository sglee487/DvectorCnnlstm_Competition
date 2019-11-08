import os

def main():
    print("현재 등록되어있는 사람 목록")
    people_list = os.listdir("feat_logfbank_nfilt40/test/")
    print(people_list)
    for people in people_list:
        print(people)

if __name__ == '__main__':
    main()