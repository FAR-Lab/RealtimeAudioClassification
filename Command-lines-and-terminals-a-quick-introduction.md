## Command-lines and terminals a quick introduction

The command-line (Windows)  and terminal (MacOs, Ubuntu) is a more historic way to control a computer. While a bit hard to understand at first, it is a very powerful tool.
The main interaction is simple. You type in a command hit return and if you spelled the command correctly the computer does whatever the command told it todo.

Let us try this quickly before we get into the details to install the required packages.

### First steps :
1. (Windows) Open the search and type CMD. Click on the entry that says command prompt. A black window should appear that has a smaller than `>` sign at the top left corner. (Mac) open Spot light (Cmd+Space) and type in "Terminal" hit enter. 
2. First, let us see in which folder we are type `pwd` and hit return. `pwd` stands for *"Print Working Directory"* It should Display something like this `/Users/dg` where `dg` is likely the username you have on that computer. 
3. Now type `ls` and hit return. Which *I think* is short for *"List"*. It should show you all the files and folders that are in your home folder. For me this looks something like this:
		```shell
		dg@Davids-MacBook-Pro:~ % ls
		Applications               EpochAnimation.blend       Pictures                   Virtual Machines.localized generatedData              
		Desktop                    GitRepos 
		...
		```
### Installing content
This should give you a short overview of how to use this interface. One last thing, information for the command to e used goes behind the command. So that if you e.g. what to change directory with ``cd`` *"Change Directory"* you need to put the folder name of the folder you want to switch to behind `cd` like so ``cd Desktop``. (The space after `cd` is important.)

In any case you can now start running the commands to install the required libraries.

Here the list  for your convenience:

Install [NumPy](https://en.wikipedia.org/wiki/NumPy) (Python support for large, multi-dimensional arrays and matrices ): ```conda install -c conda-forge numpy ```  
Install [MatplotLib](https://matplotlib.org) (Python 2D plotting library ): ```conda install -c conda-forge matplotlib ```  
Install [NumPy Ringbuffer](https://pypi.org/project/numpy_ringbuffer/) (Ringbuffer Data structure library ) ```pip install numpy_ringbuffer```  
Install [pyTorch](https://pytorch.org/get-started/locally/) (Python Machine Learning Library ): ```conda install pytorch torchvision -c pytorch```  
Install [opencv2](https://opencv.org) (Computer Vision library) : ```conda install -c conda-forge  opencv=3.4.1```  
Install [sklearn](https://scikit-learn.org/stable/) (SciKit Machine Learning library): ```conda install scikit-learn```  
Install [librosa](https://librosa.github.io/librosa/) (Music and Audio analysis library ): ```conda install -c conda-forge librosa ```  
Install [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (Python bindings for PortAudio, the cross-platform audio I/O library ) : ```conda install -c anaconda pyaudio``` 

Just like before we are just running commands in anaconda to install the required libraries. 
e.g. `conda install -c conda-forge  opencv=3.4.1` can be understood as :

`conda`  = > "Hey Anaconda...
`install`  => "... please install...
`-c conda-forge` => "...from repository conda-forge..."
`opencv=3.4.1` => "... the package openCV with version 3.4.1 ."
