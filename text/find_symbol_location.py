from text.symbols import _pad, _punctuation, _blank, _special, _letters, _arpabet
from text import _id_to_symbol

"""
_pad        = '_'
_punctuation = '!\'(),.:;?'
_blank = ' '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
"""

def find_letter_locations(text_sequence):
    '''
    PARAMS
    -----
    text_sequence: list. A sequence of integer text IDs.

    RETURNS
    -----
    locations: int list. A sorted list of letter indices.
    '''
    locations = list()
    letter_list = list(_letters) + _arpabet
    for i in range(len(text_sequence)):
        symbol_id = text_sequence[i]
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s in letter_list:
                locations.append(i)

    return locations


def find_punctuation_locations(text_sequence):
    '''
    PARAMS
    -----
    text_sequence: list. A sequence of integer text IDs.

    RETURNS
    -----
    locations: int list. A sorted list of punctuation indices.
    '''
    punctuation = _punctuation + _special
    locations = list()
    for i in range(len(text_sequence)):
        symbol_id = text_sequence[i]
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s in punctuation:
                locations.append(i)

    return locations


def find_blank_locations(text_sequence):
    '''
    PARAMS
    -----
    text_sequence: list. A sequence of integer text IDs.

    RETURNS
    -----
    locations: int list. A sorted list of blank indices.
    '''
    locations = list()
    for i in range(len(text_sequence)):
        symbol_id = text_sequence[i]
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s == _blank:
                locations.append(i)

    return locations
