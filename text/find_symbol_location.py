from text.symbols import _pad, _punctuation, _special, _letters

"""
_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
"""

_blank = ' '

def find_letter_locations(text):
    '''
    PARAMS
    -----
    text: str. A sentence. A sequence of characters.

    RETURNS
    -----
    locations: int list. A sorted list of letter indices.
    '''
    locations = list()
    for i in range(len(text)):
        char = text[i]
        if char in _letters:
            locations.append(i)

    return locations


def find_punctuation_locations(text):
    '''
    PARAMS
    -----
    text: str. A sentence. A sequence of characters.

    RETURNS
    -----
    locations: int list. A sorted list of punctuation indices.
    '''
    punctuation = _punctuation.replace(_blank, '') + _special
    locations = list()
    for i in range(len(text)):
        char = text[i]
        if char in punctuation:
            locations.append(i)

    return locations


def find_blank_locations(text):
    '''
    PARAMS
    -----
    text: str. A sentence. A sequence of characters.

    RETURNS
    -----
    locations: int list. A sorted list of blank indices.
    '''
    locations = list()
    for i in range(len(text)):
        char = text[i]
        if char == _blank:
            locations.append(i)

    return locations
