from typing import Type

from .interface import BasePreprocessor, BaseRawParser
from .TAT import TATPreprocessor, TATRawParser
from .TAT_TTS import TATTTSPreprocessor, TATTTSRawParser


PREPROCESSORS = {
    "LJSpeech": None,
    "LibriTTS": None,
    "TAT": TATPreprocessor,
    "TATTTS": TATTTSPreprocessor,
}

RAWPARSERS = {
    "LJSpeech": None,
    "LibriTTS": None,
    "TAT": TATRawParser,
    "TATTTS": TATTTSRawParser,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]


def get_raw_parser(tag: str) -> Type[BaseRawParser]:
    return RAWPARSERS[tag]
