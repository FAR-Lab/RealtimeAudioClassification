#!/usr/bin/env python
# coding: utf-8

## Load library
import queue
import matplotlib.pyplot as plt
from threading import Thread
import pyaudio
import librosa
import numpy as np
from PIL import Image
import csv
import librosa
import wave
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
import torchvision
import time
from  numpy_ringbuffer import RingBuffer

from torch.autograd import Variable
SpectrumVariables=None
ringBuffer = RingBuffer(44100*1)
buffer = queue.Queue()
pa = None
stream = None
timer = 0


def LoadModel(ModelPath="../models/CatDogResNet.pth"):
    global model
    global classes
    global Input_Resolution
    ModelData = torch.load(ModelPath,map_location='cpu')
    Input_Resolution = ModelData['resolution']
    classes = ModelData['classes']
    foundAModel=False
    if ModelData['modelType']=="resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(512, len(classes))
        print(classes)
        foundAModel=True
    if not foundAModel:
        print("Could not find requested Model. Please provide a network structure for model:",ModelData['modelType'])
        exit()
    model.load_state_dict (ModelData['model'])
    model.cpu()
    model.eval()



def StartAudio():
    global stream
    print("Opening Audio Channel")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=SpectrumVariables['SAMPLE_RATE'],
                     output=False,
                     input=True,
                     frames_per_buffer=1024,
                     stream_callback=callback)
    stream.start_stream()

def callback(in_data, frame_count, time_info, flag):
    buffer.put(in_data)
    return None, pyaudio.paContinue

def infere_Class_Type(CallBack):
    if(not ringBuffer.is_full):
        return

    mel_spec_power = librosa.feature.melspectrogram(np.array(ringBuffer), sr=SpectrumVariables['SAMPLE_RATE'],
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
    if(im.shape[1]<Input_Resolution):
        return 0
    #CallBack(im)
    #return 0
    imagesTensor = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(im[:,-224:,:])
    outputs = model(Variable(imagesTensor, requires_grad=False).unsqueeze(0))
    outputs = F.softmax(outputs)
    prob, predicted = torch.topk(outputs,len(classes))
    CallBack(predicted[0].detach().numpy(),prob[0].detach().numpy(),classes)


def StopAudio():
    global pa
    global stream
    time.sleep(1)
    stream.close()

def RunTheSystem(CallBackFunction,TargetTime=30,ModelPath = "../models/UrbanResNet.pth",):
    global SpectrumVariables
    print("Loading all relevant data.")
    LoadModel(ModelPath=ModelPath)
    SpectrumVariables={}
    with open('../SpectrumVarialbes.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k in row:
                SpectrumVariables[k]=int(row[k])

    StartAudio()
    print("Starting Running")
    t0 = time.time()
    while stream.is_active():
        while True:
                try:
                    chunk = buffer.get(block=False)
                    if chunk is None:
                        break
                    audio_data = np.frombuffer(chunk, dtype=np.float32)
                    ringBuffer.extend(audio_data)
                except queue.Empty:
                    break
                except KeyboardInterrupt:
                    break
        infere_Class_Type(CallBackFunction)
        if (TargetTime>0 )and ((time.time()-t0)>=TargetTime):
            break
    print("Stopping!")
    StopAudio()
    print("Stopped and Done!")

def EvaluateOneFile(FilePath,ModelPath = "../models/UrbanResNet.pth"):
    LoadModel(ModelPath)
    return infere_Class_For_File(FilePath)

def infere_Class_For_File(FilePath):
    OutPredictArray=[]
    OutProbArray=[]
    N_FFT=SpectrumVariables["N_FFT"]
    HOP_LENGTH= SpectrumVariables["HOP_LENGTH"]
    FMIN=SpectrumVariables["FMIN"]
    FMAX=SpectrumVariables["FMAX"]
    N_MELS=SpectrumVariables["N_MELS"]
    POWER=SpectrumVariables["POWER"]
    RESOLUTION =SpectrumVariables["RESOLUTION"]
    Audio_data ,sample_rate_in = librosa.load(FilePath,mono=True)

    mel_spec_power = librosa.feature.melspectrogram(Audio_data, sr=SamplingRate, n_fft=N_FFT,
                                                hop_length=HOP_LENGTH,
                                                n_mels=N_MELS, power=POWER,
                                                fmin=FMIN,fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    image=mel_spec_db[0:Input_Resolution,0:Input_Resolution]
    image = mel_spec_db; # convert to float
    image -= image.min() # ensure the minimal value is 0.0
    image /= image.max() # maximum value in image is now 1.0
    image*=256
    img = image.astype(np.uint8)
    colerPic = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    for i in range(int(np.floor(colerPic.shape[1]/RESOLUTION))):
        startx=RESOLUTION*i
        stopx=RESOLUTION*(i+1)
        OutputImage = cv2.resize(colerPic[:,startx:stopx,:],(RESOLUTION,RESOLUTION))
        imagesTensor = transforms.Compose([transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(OutputImage)
        imagesTensor = Variable(imagesTensor, requires_grad=False)
        testImages = imagesTensor.unsqueeze(0)
        outputs = model(testImages)
        outputs = F.softmax(outputs)
        prob, predicted = torch.topk(outputs,len(classes),sorted=False)
        OutPredictArray.append(predicted.numpy()[0])
        OutProbArray.append(prob.detach().numpy()[0])
    return OutPredictArray, OutProbArray, classes
