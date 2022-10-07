from typing import Type

from .interface import BasePreprocessor, BaseRawParser
from .TAT import TATPreprocessor, TATRawParser
from .TAT_TTS import TATTTSPreprocessor, TATTTSRawParser
from .libritts import LibriTTSPreprocessor, LibriTTSRawParser
from .ljspeech import LJSpeechPreprocessor, LJSpeechRawParser
from .aishell3 import AISHELL3Preprocessor, AISHELL3RawParser 


PREPROCESSORS = {
    "LJSpeech": LJSpeechPreprocessor,
    "LibriTTS": LibriTTSPreprocessor,
    "AISHELL-3": AISHELL3Preprocessor,
    "TAT": TATPreprocessor,
    "TATTTS": TATTTSPreprocessor,
}

RAWPARSERS = {
    "LJSpeech": LJSpeechRawParser,
    "LibriTTS": LibriTTSRawParser,
    "AISHELL-3": AISHELL3RawParser,
    "TAT": TATRawParser,
    "TATTTS": TATTTSRawParser,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]


def get_raw_parser(tag: str) -> Type[BaseRawParser]:
    return RAWPARSERS[tag]
