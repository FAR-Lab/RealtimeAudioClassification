To enable the use of common scripts for different audio detection projects, we have defined a general framework for organizing data for learning.

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

For our algorithms to understand the data, we need to write small scripts that moves the data into the folder structure mentioned above. 

Have a look at the **Transforming Datasets.ipynb** notebook in the `04_ProcessingData` folder. There we provide scripts to sort files into folders; this is used to process datasets like the urban-sounds dataset.

Handy Links
https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python how to move a file in python
