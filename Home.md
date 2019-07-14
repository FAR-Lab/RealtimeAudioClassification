# Real-time Audio classification for Musicians
(As an homage to [Tensorflow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0).)

## Overview (1+3 steps)

In this workshop, we will use Jupyter notebooks, Python3, pyTorch, and Librosa to play with neural nets that can detect different music and different audio sources. 

In this workshop we will teach you how to go from an idea for classifying audio to collecting and organizing data, generating spectrographs, training a network, and the using that network to detect audio in realtime. 



Before we can get started we first have to install a few libraries to make our life easier.

1. [Install jupyter notebooks](https://jupyter.readthedocs.io/en/latest/install.html).
2. Install all the required libraries, by copy/pasting the following commands into your Terminal or command line. If you use a Conda environment add ``-n envName`` at the end.
	```conda install -c conda-forge numpy ```   
	```conda install -c conda-forge matplotlib ```   
	Install from: https://pypi.org/project/numpy_ringbuffer/```pip install numpy_ringbuffer```   
	Install pyTorch: https://pytorch.org/get-started/locally/ so e.g. ```conda install pytorch torchvision -c pytorch```  
	Install opencv2: ```conda install -c conda-forge opencv ```  
	Install sklearn: ```conda install scikit-learn```  
	Install librosa: ```conda install -c conda-forge librosa ```  
	Install PyAudio: ```conda install -c anaconda pyaudio```  
3. [Fork this git repository.](https://github.com/FAR-Lab/Developing-and-Designing-Interactive-Devices/wiki/Forking-a-GitHub-project) [Fork](images/HowToFork.png). [More help](https://help.github.com/en/articles/fork-a-repo)
4. Clone your forked version of the repository to your local computer.
5. Start jupyter notebooks in the folder that you cloned your the forked GitHub repository i.e. in the folder you just downloaded.
6. Within jupyter navigate to the folder ./0_Setup/ and open the Notebook inside. 
7. Follow the steps inside that notebook and come back to with documents once you are done.

```	