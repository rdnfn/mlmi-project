import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset

from transcribe.constants.general import *


class SingleAudioDataset(Dataset):
    """ Pytorch Dataset that repeatedly returns the same audio file.
    """
    
    def __init__(self, audio_path, sequence_length=None):
        
        self.sequence_length = sequence_length
        
        audio, sr = soundfile.read(audio_path, dtype='int16')
        audio = torch.FloatTensor([audio]).div_(32768.0)
        
        self.data = [audio]

    def __getitem__(self, index):
        
        result = {}
        result['audio'] = self.data[0]

        return result

    def __len__(self):
        return 1