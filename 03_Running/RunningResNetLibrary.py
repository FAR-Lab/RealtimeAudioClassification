#!/usr/bin/env python
# coding: utf-8

## Load library
import pyaudio
import librosa
import numpy as np
from numpy_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torchvision
import time
from  numpy_ringbuffer import RingBuffer
from torch.autograd import Variable
from threading import Thread
from time import sleep
import cv2
import time
import pickle
from IPython.display import clear_output, display
#import rtmidi
from IPython.display import Image 
import os


model=None
classes=None
ringBuffer=None
Input_Resolution=None
SamplingRate =48000
SpectrumVariables=None
ringBuffer = RingBuffer(28672*2)
pa = None
stream = None
RunningAverageSlow={}
RunningAverageFast={}
timer = 0


def SmoothingFunction(PredictedClassName, Probablity,CallBackFunction):
    global RunningAverageSlow
    global RunningAverageFast
    if(not PredictedClassName in RunningAverageSlow):
        RunningAverageSlow[PredictedClassName]=0
    if(not PredictedClassName in RunningAverageFast):
        RunningAverageFast[PredictedClassName]=0
    
    RunningAverageFast[PredictedClassName]=RunningAverageFast[PredictedClassName]*0.7+0.3*Probablity
    RunningAverageSlow[PredictedClassName]=RunningAverageSlow[PredictedClassName]*0.9+0.1*Probablity
    if(RunningAverageFast[PredictedClassName]-RunningAverageSlow[PredictedClassName])>0.25 and Probablity>2.0:
        CallBackFunction(PredictedClassName,Probablity)
        return
    #print(RunningAverageSlow)
    CallBackFunction('None',Probablity)

def LoadModel(ModelPath="../models/CatDogResNet.pth"):
    global model
    global SpectrumVariables
    global classes
    global Input_Resolution
    ModelData = torch.load(ModelPath,map_location='cpu')
    Input_Resolution = ModelData['resolution']
    SpectrumVariables = ModelData['SpectrumVariables']
    classes = ModelData['classes']
    foundAModel=False
    if ModelData['modelType']=="resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(512, len(classes))
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
    cv2.startWindowThread() 
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=SamplingRate,
                     output=False,
                     input=True,
                     stream_callback=callback)
    stream.start_stream()

def callback(in_data, frame_count, time_info, flag):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    ringBuffer.extend(audio_data)
    return None, pyaudio.paContinue

def infere_Class_Type(CallBack,CallBack2):
    if(not ringBuffer.is_full):
        return
    N_FFT=SpectrumVariables["N_FFT"]
    HOP_LENGTH= SpectrumVariables["HOP_LENGTH"]
    FMIN=SpectrumVariables["FMIN"]
    FMAX=SpectrumVariables["FMAX"]
    N_MELS=SpectrumVariables["N_MELS"]
    POWER=SpectrumVariables["POWER"]      
    mel_spec_power = librosa.feature.melspectrogram(np.array(ringBuffer), sr=SamplingRate, n_fft=N_FFT,
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
    if(int(np.floor(colerPic.shape[1]/Input_Resolution))<0):
        return 0
    OutputImage = cv2.resize(colerPic[:,-Input_Resolution:,:],(Input_Resolution,Input_Resolution))
    if(OutputImage.shape[1]<Input_Resolution):
        return 0
    imagesTensor = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(OutputImage)
    imagesTensor = Variable(imagesTensor, requires_grad=False)
    testImages = imagesTensor.unsqueeze(0)
    outputs = model(testImages)
    outputs = F.softmax(outputs)
    prob, predicted = torch.topk(outputs,len(classes))
    #print(predicted[:2],prob[:2])
    if(not CallBack2==None):
        CallBack2(predicted,prob,classes)
    else:
        predicted=predicted[0].numpy()
        prob=prob[0].detach().numpy()
        SmoothingFunction(classes[predicted[0]],prob[0],CallBack)
def infere_Class_Type(CallBack,CallBack2):
    if(not ringBuffer.is_full):
        return
    N_FFT=SpectrumVariables["N_FFT"]
    HOP_LENGTH= SpectrumVariables["HOP_LENGTH"]
    FMIN=SpectrumVariables["FMIN"]
    FMAX=SpectrumVariables["FMAX"]
    N_MELS=SpectrumVariables["N_MELS"]
    POWER=SpectrumVariables["POWER"]      
    mel_spec_power = librosa.feature.melspectrogram(np.array(ringBuffer), sr=SamplingRate, n_fft=N_FFT,
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
    if(int(np.floor(colerPic.shape[1]/Input_Resolution))<0):
        return 0
    OutputImage = cv2.resize(colerPic[:,-Input_Resolution:,:],(Input_Resolution,Input_Resolution))
    if(OutputImage.shape[1]<Input_Resolution):
        return 0
    imagesTensor = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(OutputImage)
    imagesTensor = Variable(imagesTensor, requires_grad=False)
    testImages = imagesTensor.unsqueeze(0)
    outputs = model(testImages)
    outputs = F.softmax(outputs)
    prob, predicted = torch.topk(outputs,len(classes))
    #print(predicted[:2],prob[:2])
    if(not CallBack2==None):
        CallBack2(predicted,prob,classes)
    else:
        predicted=predicted[0].numpy()
        prob=prob[0].detach().numpy()
        SmoothingFunction(classes[predicted[0]],prob[0],CallBack)


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

def StopAudio():
    global pa
    global stream
    time.sleep(1)
    stream.close()
    cv2.destroyAllWindows()

def DoStuff(Input,Probablity):
    #print("I heard a "+str(Input)+'with'+str(Probablity))
    global timer
    clear_output(wait=True)
    print(Input,timer)
    if(timer>0):
        timer-=1
    if(timer<=1):
        os.system("say I think I heard a "+str(Input))
        print("I heard a "+str(Input))
        timer=50

def RunTheSystem(TargetTime=30,ModelPath = "../models/UrbanResNet.pth",CallBackFunction=DoStuff,CallBack2=None):
    print("Loading all relevant data.")
    LoadModel(ModelPath=ModelPath)
    StartAudio()
    print("Starting Running")
    t0 = time.time()
    while stream.is_active():
        infere_Class_Type(CallBackFunction,CallBack2)
        if (TargetTime>0 )and ((time.time()-t0)>=TargetTime):
            break
    print("Stopping!")
    StopAudio()
    print("Stopped and Done!")

def EvaluateOneFile(FilePath,ModelPath = "../models/UrbanResNet.pth"):
    LoadModel(ModelPath)
    return infere_Class_For_File(FilePath)


