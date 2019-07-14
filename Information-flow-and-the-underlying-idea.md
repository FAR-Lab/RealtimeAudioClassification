## Information flow, and the underlying idea.

The scripts in this project are optimized so that you have to change as little code as possible if you want to play with new audio detection projects. To make this work we use a lot of Finder(Mac)/Explorer(win) file manipulation techniques.

### Defining classes
Instead of doing difficult manipulations with `.csv` files we move our class definition into folders. Everything you want to have classified as a certain type of data should go into one folder. This is very important if you, in the future want to change the types of thing you want to distinguish. Just move things around, until they are in the separation that you want.
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
            
## Working with existing DataSets
One problematic aspect is, that a lot of data sets are not structured in that way, and come with a .csv file (like a spreadsheet) that associates each file into a class. You will see an example of such a spread sheet when you open the cats-and-dogs audio library.
For our algorithms to understand the data we need to write small scripts that moves the data into the folder structure mentioned above. 

Have a look at the section **transforming data-sets** In the Setup Notebook. There we show how the data is being transformed and have scripts ready to do that for both the cats-and-dogs data set and the urban-sounds dataset.
