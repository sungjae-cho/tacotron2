""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict
import string

_pad        = '_'
_punctuation = '!\'(),.:;?'
_blank = ' '
_special = '-'

# Korean Symbols
_Start_Code, _ChoSung, _JungSung = 44032, 588, 28
_ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
_JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ',
                    'ㅢ', 'ㅣ']
_JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
_phone_LIST = ['ㅓ', 'ㅝ', 'ㅃ', 'ㅛ', 'ㅢ', 'ㄶ', 'ㅇ', 'ㅎ', 'ㅖ', 'ㅗ', 'ㅠ', 'ㅆ', 'ㅜ', 'ㅌ', 'ㄿ', 'ㅔ', 'ㅋ', 'ㄲ', 'ㅑ', 'ㄸ','ㅙ', 'ㅞ', 'ㅅ',
              'ㅘ', 'ㄻ', 'ㅍ', 'ㄳ', 'ㄼ', 'ㄹ', 'ㅄ', 'ㅡ', 'ㅈ', 'ㅂ', 'ㅣ', 'ㅟ', 'ㄽ', 'ㅐ', 'ㅀ', 'ㅕ', 'ㅒ', 'ㄷ', 'ㅏ', 'ㅊ', 'ㄺ', 'ㄴ', 'ㄱ',
              'ㅉ', 'ㄵ', 'ㅁ', 'ㄾ', 'ㅚ']

#_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters = string.ascii_uppercase + string.ascii_lowercase

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Hangul symbols
#_hangul = sorted(list(set(_ChoSung_LIST + _JungSung_LIST + _JongSung_LIST)))
_hangul = sorted(list(set(_ChoSung_LIST + _JungSung_LIST + _JongSung_LIST[1:])))

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_blank) + list(_letters) + _arpabet  + _hangul
