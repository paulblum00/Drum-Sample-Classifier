import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import os


"""
The preprocessor expects an audio file with a samplerate of 44.1khz. It is cut to 12736 samples. With a 
The network expects an image of size 128 x 200 by default.
"""
class Drums_Dataset(Dataset):

    def prepare_audiofile(self, filename):
        # Load file
        y, sr = librosa.load(filename)

        # Trim to desired size
        y = y[:self.insize]
        if len(y) < self.insize:
            y = np.pad(y,(0,self.insize-len(y)))
        
        # Get spectrogram
        return librosa.feature.melspectrogram(y=y, sr=sr, win_length=256, hop_length=64)

    def __init__(self, root, insize_samples=3316):
        self.categories = os.listdir(root)
        self.filenames = [os.listdir( os.path.join(root,category) ) for category in self.categories]
        self.root = root
        self.insize = insize_samples

    def __len__(self):
        res = 0
        for l in self.filenames:
            res += len(l)
        return res
    
    def __getitem__(self, index):
        # Find category and shift index to be an index in the corrsponding category
        cat = 0
        while cat < len(self.categories) and index-len(self.filenames[cat]) >= 0:
            index -= len(self.filenames[cat])
            cat += 1
        return self.prepare_audiofile(os.path.join(self.root, self.categories[cat], self.filenames[cat][index]))
        



ds = Drums_Dataset("./dataset/")
ms = ds[64]
plt.imshow(ms)
plt.show()