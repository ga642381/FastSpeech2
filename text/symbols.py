""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

"""
{
'_': 0, '-': 1, '!': 2, "'": 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, ';': 9, '?': 10,
' ': 11, 'A': 12, 'B': 13, 'C': 14, 'D': 15, 'E': 16, 'F': 17, 'G': 18, 'H': 19, 'I': 20,
'J': 21, 'K': 22, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30,
'T': 31, 'U': 32, 'V': 33, 'W': 34, 'X': 35, 'Y': 36, 'Z': 37, 'a': 38, 'b': 39, 'c': 40,
'd': 41, 'e': 42, 'f': 43, 'g': 44, 'h': 45, 'i': 46, 'j': 47, 'k': 48, 'l': 49, 'm': 50,
'n': 51, 'o': 52, 'p': 53, 'q': 54, 'r': 55, 's': 56, 't': 57, 'u': 58, 'v': 59, 'w': 60,
'x': 61, 'y': 62, 'z': 63, '@AA': 64, '@AA0': 65, '@AA1': 66, '@AA2': 67, '@AE': 68, '@AE0': 69, '@AE1': 70,
'@AE2': 71, '@AH': 72, '@AH0': 73, '@AH1': 74, '@AH2': 75, '@AO': 76, '@AO0': 77, '@AO1': 78, '@AO2': 79, '@AW': 80,
'@AW0': 81, '@AW1': 82, '@AW2': 83, '@AY': 84, '@AY0': 85, '@AY1': 86, '@AY2': 87, '@B': 88, '@CH': 89, '@D': 90,
'@DH': 91, '@EH': 92, '@EH0': 93, '@EH1': 94, '@EH2': 95, '@ER': 96, '@ER0': 97, '@ER1': 98, '@ER2': 99, '@EY': 100,
'@EY0': 101, '@EY1': 102, '@EY2': 103, '@F': 104, '@G': 105, '@HH': 106, '@IH': 107, '@IH0': 108, '@IH1': 109, '@IH2': 110,
'@IY': 111, '@IY0': 112, '@IY1': 113, '@IY2': 114, '@JH': 115, '@K': 116, '@L': 117, '@M': 118, '@N': 119, '@NG': 120,
'@OW': 121, '@OW0': 122, '@OW1': 123, '@OW2': 124, '@OY': 125, '@OY0': 126, '@OY1': 127, '@OY2': 128, '@P': 129, '@R': 130,
'@S': 131, '@SH': 132, '@T': 133, '@TH': 134, '@UH': 135, '@UH0': 136, '@UH1': 137, '@UH2': 138, '@UW': 139, '@UW0': 140,
'@UW1': 141, '@UW2': 142, '@V': 143, '@W': 144, '@Y': 145, '@Z': 146, '@ZH': 147, '@sp': 148, '@spn': 149, '@sil': 150
}
"""

from text import cmudict

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + _silences
)

