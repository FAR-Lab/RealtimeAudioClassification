#!/usr/bin/env python
# coding: utf-8
import pyaudio
import librosa
import numpy as np
from numpy_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
import pyaudio
import torch
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


print("ExecutingCode");
ModelPath="../models/UrbanSoundsWithCats.pth"

ModelData = torch.load(ModelPath,map_location='cpu')
print(ModelData.keys())
Input_Resolution = ModelData['resolution']
### Spectograph resolution
SpectrumVariables = pickle.load(open( '../01_Spectrum Generation/test7.SpectrumVariables', "rb" ) )

SAMPLE_RATE=SpectrumVariables["SAMPLE_RATE"]
N_FFT=SpectrumVariables["N_FFT"]
HOP_LENGTH= SpectrumVariables["HOP_LENGTH"]
FMIN=SpectrumVariables["FMIN"]
FMAX=SpectrumVariables["FMAX"]
N_MELS=SpectrumVariables["N_MELS"]
POWER=SpectrumVariables["POWER"]
FMAX=SAMPLE_RATE/2
### Debug Info
K=2



model=None


foundAModel=False;
if ModelData['modelType']=="resnet18":
    model = models.resnet18()
    foundAModel=True;

if not foundAModel:
    print("Could not find requested Model. Please provide a network structure for model:",ModelData['modelType'])
    exit();

model.load_state_dict (ModelData['model'])
classes = ModelData['classes']


model.cpu()
model.eval()
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def audio_interfaces():
        """
        Extracts audio interfaces data

        :return list[AudioInterface]: Audio interfaces data
        """
        p = pyaudio.PyAudio()
        interfaces = []
        for i in range(p.get_device_count()):
            data = p.get_device_info_by_index(i)
            if 'hw' not in data['name']:
                interfaces.append(data)
        p.terminate()
        return interfaces
#print(audio_interfaces())



ringBuffer = RingBuffer(28672*2)
pa = pyaudio.PyAudio()
running = True




def callback(in_data, frame_count, time_info, flag):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_data = librosa.resample(audio_data, 44100, SAMPLE_RATE)
    ringBuffer.extend(audio_data)
    return None, pyaudio.paContinue




def infere_Class_Type():
    if(not ringBuffer.is_full):
        return;

    audio_data = np.array(ringBuffer)
    mel_spec_power = librosa.feature.melspectrogram(audio_data, sr=SAMPLE_RATE, n_fft=N_FFT,
                                                hop_length=HOP_LENGTH,
                                                n_mels=N_MELS, power=POWER,
                                               fmin=FMIN,fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)

    image=mel_spec_db[0:Input_Resolution,0:Input_Resolution]
    image = mel_spec_db; # convert to float
    image -= image.min() # ensure the minimal value is 0.0
    image /= image.max() # maximum value in image is now 1.0
    image*=256;
    img = image.astype(np.uint8)
    colerPic = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    if( int(np.floor(colerPic.shape[1]/Input_Resolution))<0):
        return;
    #print(colerPic[:,-Input_Resolution:,:].shape)
    OutputImage = cv2.resize(colerPic[:,-Input_Resolution:,:],(Input_Resolution,Input_Resolution))
    cv2.imshow('dst_rt', OutputImage)
    cv2.waitKey(1)

    if(colerPic.shape[1]<Input_Resolution):
        return;
    imagesTensor = transform(colerPic)
    imagesTensor = Variable(imagesTensor, requires_grad=True)
    testImages = imagesTensor.unsqueeze(0)
    outputs = model(testImages)

    prob, predicted = torch.topk(outputs,K)
    predicted=predicted[0].numpy()
    prob=prob[0].detach().numpy()
    print('---')
    for  j in range(len(predicted)):
        #if prob[j] > 10.0 and len(predicted)>j+1 and len(prob)>j+1 :
        if(predicted[j]>len(classes)):
            print(predicted,len(classes))
            continue
        print('Predicted:\t{} \t| Probablilty: \t{:f}'.format(classes[predicted[j]],prob[j]))



def startProgram(targetLength=20):
    stream.start_stream()
    t0 = time.time()
    while stream.is_active():
        tStart=time.time()
        infere_Class_Type()
        #print("Inference time in ms:\t{:f}".format((time.time()-tStart)*1000) )
        #print("Running time: {:f}".format(time.time()-t0))
        if ( targetLength>0 )and ( (time.time()-t0)>=targetLength):
            break;

stream = pa.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=44100,
                 output=False,
                 input=True,
                 stream_callback=callback)


if __name__ == '__main__':
    print("Starting Running");
    startProgram(0)
    print("Stopping!");
    time.sleep(1)
    pa.terminate()
    stream.close()
