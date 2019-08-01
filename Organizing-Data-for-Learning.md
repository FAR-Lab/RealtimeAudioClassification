To enable the use of common scripts for different audio detection projects, we have defined a general framework for organizing data for learning.

To lessen the work of this data organization, we use a lot of Finder(Mac)/Explorer(win) file manipulation techniques.

### Defining classes

We define classes of data by folders--that is, data that should be classified similarly are put in the same file folder. If you later want to change the types of thing you want to distinguish, you just change the datafilee in the folders.

The file names in our case are not important.

In general the structure should always look like the following:
``` shell
AudioData
├── audio-cats-and-dogs
│   └── cats_dogs
│       ├── test
│       │   ├── cat
│       │   └── dog
│       └── train
│           ├── cat
│           └── dog
└── urban-sound-classification
    ├── test
    │   └── Test
    └── train
        └── Train
            ├── air_conditioner
            ├── car_horn
            ├── children_playing
            ├── dog_bark
            ├── drilling
            ├── engine_idling
            ├── gun_shot
            ├── jackhammer
            ├── siren
            └── street_music
```
            
## Working with Existing DataSets
A lot of data sets do not come structured in the way we described above. Instead, the data files come with a .csv spreadsheet file that associates each file into a class. 

You will see an example of such a spread sheet when you open the cats-and-dogs audio library.

For our algorithms to understand the data, we need to write small scripts that moves the data into the folder structure mentioned above. 

Have a look at the **transforming data-sets**  notebook. There we provide scripts to sort files into folders; this is used to process the cats-and-dogs data set and the urban-sounds dataset.

## Squishing time  - taking a snapshot

One of the field neural networks in their current form are very good at are detecting images. The convolutional Neural Networks (CNNs) that are being used for image recognition have become almost ubiquitous and are therefore very easy to play with. 
In this project we build on that history and convert our audio into spectrograms, which is essentially an image representation of a little audio snippet. After we generated these audio-images we can retrain any standard image classifier to work with our images and help us classify our audio data. 
Spectrograms 
What do they do for us
Why 224x224
What is Mel spectrum 
Whats is hop-length what is fftCount

The values don't matter as much. What is important is consistency among generating and using the system
We generate data and then we use it to train a normal image net neural nentwork.

