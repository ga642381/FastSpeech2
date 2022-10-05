from typing import Type

from .interface import BasePreprocessor, BaseRawParser
from .TAT import TATPreprocessor, TATRawParser
from .TAT_TTS import TATTTSPreprocessor, TATTTSRawParser
from .libritts import LibriTTSPreprocessor, LibriTTSRawParser


PREPROCESSORS = {
    "LJSpeech": None,
    "LibriTTS": LibriTTSPreprocessor,
    "AISHELL-3": None,
    "TAT": TATPreprocessor,
    "TATTTS": TATTTSPreprocessor,
}

RAWPARSERS = {
    "LJSpeech": None,
    "LibriTTS": LibriTTSRawParser,
    "AISHELL-3": None,
    "TAT": TATRawParser,
    "TATTTS": TATTTSRawParser,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]


def get_raw_parser(tag: str) -> Type[BaseRawParser]:
    return RAWPARSERS[tag]
