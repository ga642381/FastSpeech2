from text.symbols import symbols, common_symbols, en_symbols, zh_symbols


def get_phoneme_set(path, encoding='utf-8'):
    phns = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line == '\n':
                continue
            phns.append('@' + line.strip())
    return phns


LANG_ID2SYMBOLS = {
    "en": en_symbols,
    "zh": zh_symbols,
    "twn": get_phoneme_set("lexicon/taiwanese_phones.txt"),
}
