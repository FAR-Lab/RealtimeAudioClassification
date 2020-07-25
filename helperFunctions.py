from __future__ import print_function
import csv
import numpy as np
import random
import librosa
import wave
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

from ipywidgets import interactive
import ipywidgets as widgets

from PIL import Image
import IPython.display as displayImg

from ipywidgets import interact, widgets
import glob
import IPython.display as ipd

import sys

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def GenerateSpectrums(MainFile):
    SpectrumVariables={}
    with open('../SpectrumVarialbes.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k in row:
                SpectrumVariables[k]=int(row[k])

    x ,sample_rate_in = librosa.load(MainFile,mono=True)
    audio_data = librosa.resample(x, sample_rate_in, SpectrumVariables['SAMPLE_RATE'])
    mel_spec_power = librosa.feature.melspectrogram(audio_data, sr=SpectrumVariables['SAMPLE_RATE'],
                                                    n_fft=SpectrumVariables['N_FFT'],
                                                    hop_length=SpectrumVariables['HOP_LENGTH'],
                                                    n_mels=SpectrumVariables['N_MELS'],
                                                    power=SpectrumVariables['POWER'],
                                                   fmin=SpectrumVariables['FMIN'],
                                                    fmax=SpectrumVariables['FMAX'])
    mel_spec_db = np.float32(librosa.power_to_db(mel_spec_power, ref=np.max))
    mel_spec_db-=mel_spec_db.min()
    mel_spec_db/=mel_spec_db.max()
    im = np.uint8(cm.gist_earth(mel_spec_db)*255)[:,:,:3]
    ArrayofPictures = []
    RESOLUTION = SpectrumVariables['RESOLUTION']
    for i in range(int(np.floor(im.shape[1]/RESOLUTION))):
        startx=RESOLUTION*i
        stopx=RESOLUTION*(i+1)
        ArrayofPictures.append(im[:,startx:stopx,:])
    return ArrayofPictures

def log_mel_spec_tfm(dataInput):
    src_path=dataInput[0]
    dst_path=dataInput[1]
    #print(src_path, dst_path)
    print('Starting on',os.path.split(src_path)[1])
    pictures = GenerateSpectrums(src_path)
    print(len(pictures))
    fname = os.path.split(src_path)[-1]
    count=0
    for pic in pictures:
        plt.imsave(os.path.join(dst_path,(fname.replace(".flac",'-')\
                                          .replace(".aif",'-').replace(".wav",'-')\
                                          .replace(".m4a",'-').replace(".mp3",'-')\
                                          +str(count)+'.png')), pic)
        count+=1
    if(count==0):
        print(src_path)


try:
    Type
except NameError:
    if(len(sys.argv)>1):
        print("FoundArguments, will start converting")
        source = str(sys.argv[1])
        target = str(sys.argv[2])
        log_mel_spec_tfm((source,target))
else:
    if(Type=="INTERFACE"):
        SOURCE_DATA_ROOT='../AudioData/'

        style = {'description_width': 'initial'}

        ClassSelection = widgets.Dropdown(options=listdir_nohidden(SOURCE_DATA_ROOT), description='Source for Training Data:',style=style)
        FileSelection = widgets.Dropdown(description='Audio file to visualize',style=style)

        def updateLocation(*args):
            FileSelection.options=listdir_nohidden(os.path.join(SOURCE_DATA_ROOT,ClassSelection.value))

        ClassSelection.observe(updateLocation)

        display(ClassSelection)
        display(FileSelection)
        updateLocation();
    elif(Type=="TRAINING"):
        SPECTRUM_IMAGES_ROOT="../GeneratedData/"
        class SpectrumDataset(torch.utils.data.Dataset):
            """Face Landmarks dataset."""
            def __init__(self,ClassName,root_dir,transform=None):
                """
                Args:
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
                """
                self.root_dir = root_dir
                self.ClassName=ClassName
                self.fileList= [f for f in os.listdir(root_dir) if f.endswith('.png')]
                print(root_dir,len(self.fileList))
                self.transform = transform
            def ReduceSize(self,ItemCount):
                self.fileList = random.choices(self.fileList, k=ItemCount)
            def __len__(self):
                return len(self.fileList)
            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()
                img_path = os.path.join(self.root_dir,
                                        self.fileList[idx])
                image = Image.open(img_path)
                image=image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image,self.ClassName
        classes = [os.path.split(c)[1] for c in listdir_nohidden(SPECTRUM_IMAGES_ROOT)]
        widgetDict={}
        print("Select classes to use for training:");
        for c in classes:
            widgetDict[c]=widgets.Checkbox(
            value=False,
            description=c,
            disabled=False,
            indent=False)
            display(widgetDict[c])
