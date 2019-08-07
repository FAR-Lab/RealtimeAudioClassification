## Cats vs Dogs

The Cats-Vs-Dogs dataset is a simple audio at a set that has two classes, sounds of cats meowing and of dogs barking. The data was created to have an easy binary example of how classification can work. The data set is a bout 1.5Gb large and has samples of varying length. The samples are all wav files with at least 16bit and at least 44.1 kHz sampling rate. Most will have a sampling rate of 48 kHz and a bit depth off 24bit.


While the bit depth is not as important (noise is a good thing for neural nets) the sampling rate is very important. The higher the [niquist frequency](wikipedia.org) the more information can be displayed in the Spectrogram.

### Collection

This data set was collected from [freesound.org](freesound.org). Most of the sounds are under public license and have been recorded on very different audio gear, in different contexts etc. 