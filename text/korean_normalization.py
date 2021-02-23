#-*- coding: utf-8 -*-

import re
import pickle

# say-as 기본 규칙: 참고 논문 <기술문에서 우리말 숫자 쓰기, 권성규>
_mandarin_num = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠",
                    "8": "팔", "9": "구", "10": "십", "100": "백", "1000": "천", "10000": "만", "100000000": "억",
                    "1000000000000": "조"}
_korean_num = {"1": "한", "2": "두", "3": "세", "4": "네", "5": "다섯", "6": "여섯", "7": "일곱",
                  '8': "여덟", "9": "아홉", "10": "열", "20": "스물", "30": "서른", "40": "마흔", "50": "쉰",
                  "60": "예순", "70": "일흔", "80": "여든", "90": "아흔"}
_korean_end_word = ['개', '대', '돈', '마리', '벌', '살', '손', '자루', '죽', '채', '켤레', '쾌', '시', '과'
                                                                                           '포기', '번째', '살', '척']
_english_word = {'a': '에이', 'b': '비', 'c': '씨', 'd': '디', 'e': '이', 'f': '에프', 'g': '쥐', 'h': '에이치',
               'i': '아이', 'j': '제이', 'k': '케이', 'l': '엘', 'n': '엔', 'm': '엠', 'o': '오', 'p': '피',
               'q': '큐', 'r':'얼', 's': '에스', 't': '티', 'u':'유', 'v':'브이', 'w':'더블유', 'x': '엑스',
               'y': '와이', 'z': '지'}

# lexicon 기본 규칙: 참고 사전 <국립국어원 표준국어대사전>
with open('./text/dictionary/lexicon.pickle', 'rb') as handle:
    _lexicon = pickle.load(handle)
# sub 기본 규칙
with open('./text/dictionary/sub.pickle', 'rb') as handle:
    _sub = pickle.load(handle)
with open('./text/dictionary/num_sub.pickle', 'rb') as handle:
    _num_sub = pickle.load(handle)

def read1to999(n):
    units = [''] + list('십백천')
    nums = '일이삼사오육칠팔구'
    result = []
    i = 0
    while n > 0:
        n, r = divmod(n, 10)
        if r > 0:
            if units[i] == '':
                result.append(nums[r - 1] + units[i])
            else:
                if r == 1:
                    result.append(units[i])
                else:
                    result.append(nums[r - 1] + units[i])
        i += 1
    return ''.join(result[::-1])


def readNumM(n):
    """
    한자로 숫자 읽기
    """
    result = ''
    if n >= 1000000000000:
        r, n = divmod(n, 10000000000000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '조'
        else:
            result += tmp + "조"
    if n >= 100000000:
        r, n = divmod(n, 100000000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '억'
        else:
            result += tmp + "억"
    if n >= 10000:
        r, n = divmod(n, 10000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '만'
        else:
            result += tmp + "만"
    result += read1to999(n)
    return result


def readNumK(intNum):
    """
    한글로 숫자 읽기
    """
    tmp_list = list(_korean_num.keys())
    num_list = list()
    for num in tmp_list:
        num_list.append(int(num))
    num_list.sort(reverse=True)
    result = ""
    for num in num_list:
        if intNum >= num:
            intNum -= num
            result += _korean_num[str(num)]
    return result


def txt_preprocessing(txt):
    word_list = txt.split(' ') # for tts

    for k, word in enumerate(word_list):
        # lexicon & sub 발음 교체
        english = re.sub('[^a-zA-Z]', '', word)
        not_checked = 1
        if english != '' and not re.findall('\d', word):
            # lexicon 처리
            for key, value in _lexicon.items():
                if key.lower() == english.lower():
                    word_list[k] = word_list[k].replace(english, value)
                    not_checked = 0
            # sub 처리
            for key, value in _sub.items():
                if key.lower() == english.lower():
                    word_list[k] = word_list[k].replace(english, value)
                    not_checked = 0
        elif re.findall('\d+', word):
            # num_sub 처리
            for key, value in _num_sub.items():
                if key in word:
                    word_list[k] = word_list[k].replace(key, value)
                    not_checked = 0
            # say-as 발음 교체
            seperated_num = 0
            if '-' in word:
                seperated_num = 1
            if '.' in word:
                if word[-1] != '.':
                    word_list[k].replace('.', '점')
            if ',' in word:
                if word[-1] != ',':
                    word_list[k].replace(',', '')
                    word.replace(',', '')
            strNum_list = re.findall('\d+', word)  # 값 중복 시 제거해 나가면서 처리 필요
            prev = -1
            for strNum in strNum_list:
                pos = word.index(strNum)
                if prev == pos:  # 약식 값 중복 처리
                    continue
                wList = [word[0:pos], word[pos: pos + len(strNum)], word[pos + len(strNum):]]
                wList = [w for w in wList if not w == '']
                check = ""
                # 처음이 0으로 시작하면 한문-분리
                if strNum[0] == '0' or seperated_num:
                    check = "한문-분리"
                    if word_list[k-1] == '카드번호는':
                        word_list[k]= word_list[k].replace('-', '다시')
                    else:
                        word_list[k]=  word_list[k].replace('-', '에')
                else:
                    for i, w in enumerate(wList):
                        # 숫자 뒤에 붙는 것이 없을 때, 한문
                        if len(wList) == (i + 1):
                            if k > 1:
                                if word_list[k - 1][0] == '-':
                                    check = "한문-분리"
                                    break
                            if k + 1 < len(word_list):
                                if word_list[k + 1] == '':
                                    check = "한문"
                                elif word_list[k + 1][0] == '-':
                                    check = "한문-분리"
                                elif word_list[k + 1][0] in _korean_end_word:
                                    check = "한글"
                                else:
                                    check = "한문"
                            else:
                                check = "한문"
                            break
                        elif w == strNum:
                            # 숫자 뒤에 붙는 것에 따라 한글, 한문 선택
                            if wList[i + 1][0] in _korean_end_word:
                                check = "한글"
                            else:
                                check = "한문"
                            break

                tmpNum = ""
                intNum = int(strNum)
                if check == "한문-분리":
                    for s in strNum:
                        # 한글자씩 읽기 (0 == 공)
                        tmpNum += _mandarin_num[s]
                elif check == "한문":
                    # 숫자 한문 읽기
                    tmpNum = readNumM(intNum)
                else:  # check == "한글"
                    # 100이상 한문 읽기 + 이하 한글 읽기
                    tmpNum = readNumM(intNum // 100 * 100) + readNumK(intNum % 100)

                word_list[k] = word_list[k].replace(strNum, tmpNum)
        elif '-' in word:
            word_list[k] = word_list[k].replace('-', '에')
        if not_checked:
            tmp = ''
            for char in word_list[k]:
                if char.lower() in _english_word.keys():
                    not_checked = 0
                    tmp += _english_word[char.lower()]
                else:
                    tmp += char
            word_list[k] = tmp
    tts_sentence = word_list[0]
    for word in word_list[1:]: # 길이 1 예외처리 필요
        tts_sentence += ' ' + word

    return tts_sentence
