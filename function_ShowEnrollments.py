import os

def main():
    print("현재 등록되어있는 사람 목록")
    people_list = os.listdir("feat_logfbank_nfilt40/test/")
    people_list.remove('103F3021')
    people_list.remove('207F2088')
    people_list.remove('213F5100')
    people_list.remove('217F3038')
    people_list.remove('225M4062')
    people_list.remove('229M2031')
    people_list.remove('230M4087')
    people_list.remove('233F4013')
    people_list.remove('236M3043')
    people_list.remove('240M3063')
    print(people_list)
    for people in people_list:
        print(people)

if __name__ == '__main__':
    main()