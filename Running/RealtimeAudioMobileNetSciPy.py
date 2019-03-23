    #!/usr/bin/env python
# coding: utf-8
import pyaudio
import librosa
import numpy as np
from numpy_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal

import pyaudio
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import numpy as np
import time
from  numpy_ringbuffer import RingBuffer
from torch.autograd import Variable
from threading import Thread
from time import sleep
import cv2 as cv
import time


dimension=300 # Required for mobile net
pictureTimeLength = 1
TargetSampleRate = 44100

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

class Net(nn.Module):  # Mobile Net
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
model = Net()



model.load_state_dict(torch.load("../models/MobileNetSciPy.pth", map_location=lambda storage, loc: storage));
model.cpu()
model.eval()

SamplesPerPicture = pictureTimeLength * TargetSampleRate
nTimeBin =dimension+1 # width
nFreqBin =dimension+1 # height
eps=1e-10

nperseg =int(2*(nFreqBin-1))
noverlap = int(np.floor((SamplesPerPicture - nTimeBin*nperseg)/(1-nTimeBin)))



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
print(audio_interfaces())


# In[ ]:
#TARGETSampleRate = 22050
#n_fft = 1024
#hop_length = 256
#n_mels = 224
#fmin = 20
#fmax = TARGETSampleRate / 2


TARGETSampleRate = TargetSampleRate
n_fft = 2048
hop_length = 128 #64 #256
n_mels = 224
fmin = 20 #60
fmax = TARGETSampleRate / 2


SampleRate = TargetSampleRate

ringBuffer = RingBuffer(pictureTimeLength * SampleRate,np.float32)
#ringBuffer = RingBuffer(28672)
pa = pyaudio.PyAudio()
running = True
MemoryInUse=False


# In[ ]:


def callback(in_data, frame_count, time_info, flag):
    global MemoryInUse
    global ringBuffer
    audio_data = np.frombuffer(in_data, dtype=np.float32)

    audio_data = librosa.resample(audio_data, 44100, SampleRate)
    MemoryInUse = True
    ringBuffer.extend(audio_data)
    #print(len(ringBuffer)/(3 * SampleRate)*100)
    MemoryInUse=False
    return None, pyaudio.paContinue


# In[ ]:


def infere_Class_Type():
    #n_fft = 1024
    #hop_length = 256
    #n_mels = 224
    #fmin = 20
    #fmax = SampleRate / 2
    if(not ringBuffer.is_full):
    #    print(len(ringBuffer))
        return;

    audio_data = np.array(ringBuffer)

    #if(len( ringBuffer)<=1):
#              return;
    #audio_data = np.load("test.npy")
    #print(audio_data.shape)
    buffLength = 1*TARGETSampleRate
    targetTimeBins = 300
    timePasedPerImage=1

    #duration =0.25
    #window_size=40
    #step_size=20
    eps=1e-10

    #nperseg = int(round(window_size * TARGETSampleRate / 1e3))
    #nperseg = int(round(duration*TARGETSampleRate))

    #print("blub",nperseg,noverlap);
    #noverlap = int(round(step_size * TARGETSampleRate / 1e3))
    freqs, times, spec = signal.spectrogram(audio_data,fs=TARGETSampleRate,window='hann',nperseg=nperseg,noverlap=noverlap)#,scaling ='spectrum')
    #freqs, times, spec = signal.spectrogram(audio_data,fs=TARGETSampleRate,window='hann')#,scaling ='spectrum')
    #print(len(freqs))
    log_specgram = np.log(spec.astype(np.float32) + eps)
    #log_specgram = spec.T
    #mel_spec_power = librosa.feature.melspectrogram(audio_data, sr=SampleRate, n_fft=n_fft,
                                                #hop_length=hop_length,
                                                #n_mels=n_mels, power=2.0,
                                               #fmin=fmin,fmax=fmax)
    #mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    #print(mel_spec_db.shape[1],"resulting Spectrum and associated ring buffer: ",len(ringBuffer))
    image=log_specgram[0:224,0:224]
    #print(log_specgram.shape)
    im = plt.imshow(image)
    colors = im.cmap(im.norm(image))
    data = colors.astype(np.float64) / np.max(colors) # normalize the data to 0 - 1
    data = 255 * data # Now scale by 255
    img = data.astype(np.uint8)
    img = img[:,:,0:3]
    #plt.imshow(img)
    cv.imshow('dst_rt', img)
    cv.waitKey(1)

    if(log_specgram.shape[1]<dimension):
        return;
    imagesTensor = transform(img)
    imagesTensor = Variable(imagesTensor, requires_grad=True)
    testImages = imagesTensor.unsqueeze(0)
    outputs = model(testImages)
    k=3
    prob, predicted = torch.topk(outputs, k)
    predicted=predicted[0].numpy()
    #print(prob)
    #print(prob.dtype)

    prob=prob[0].detach().numpy()
    print('---')
    for  j in range(k):
        if prob[j] > 00.0:
            print('Predicted:\t{} \t| Probablilty: \t{:f}'.format(classes[predicted[j]],prob[j]))


stream = pa.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=TARGETSampleRate,
                 output=False,
                 input=True,
                 stream_callback=callback)
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


if __name__ == '__main__':
    print("Starting Running");
    startProgram(0)
    print("Stopping!");
    time.sleep(1)
    pa.terminate()
    stream.close()




# In[ ]:
