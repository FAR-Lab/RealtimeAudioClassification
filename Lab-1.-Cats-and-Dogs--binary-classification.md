## How to use Jupyter Notebooks

The main ["IDE"](https://en.wikipedia.org/wiki/Integrated_development_environment) that we will be using this week are Jupyter Notebooks. Jupyter Notebooks is great to try out different code snippets to selectively debug the code you are writing.

Everything in these notebooks resolves around cells. The current active cell has a blue or green stripe on the left side. ([Why green or blue? Look here!](https://medium.com/ibm-data-science-experience/back-to-basics-jupyter-notebooks-dfcdc19c54bc)) Normally a cell contains a section of code that is executed together. E.g. We often define each function in their own cell.

![Example Image of how a cell looks like.](images/ExampleCell.png)

When clicking into a cell you can edit the code that is in that cell. To run the code you can click on the button on the top that says ` >| Run` or press `shift + return` at the same time on your keyboard. This will run this cell and select the cell below. This behavior means that if you want to run multiple cells at the same time you can press `shift + return` a few times to quickly run the whole notebook.

If you followed the setup steps to [install the packages](https://github.com/DavidGoedicke/RealtimeAudioClassification/wiki/Lab-0.-Setting-up) and [download the data-set](https://github.com/DavidGoedicke/RealtimeAudioClassification/wiki/Lab-0.-Setting-up#download-datasets) then we can proceed in running the relevant code.

### First Generating Spectrums
The data you downloaded are normal audio clips collected from [freesound.org](http://freesound.org). The type of neural networks we are using, however, work with images. So the first step is to compute images from the audio data.

We prepared the notebook `GeneratingSpectrums` in the `01_Spectrum Generation` folder for this task.

The first cell in this notebook, like in all other notebooks, loads in all the libraries that we previously installed. The second cell defines the folder paths to both the source folder where the audio is stored and the folder where the generated images should go.

Now run the first two cells by making sure that the first one is selected (remember the green or blue line on the left side) and press `shift+return` twice. It might take a while but you should see the number in the top left corner, next to the cell change from empty to a star to a number. Something like this 
```python
In []: # This code block has not been executed.

In [*]: # This code is being executed but has not finished. 

In [1]: #This code block is finished and was the first one to finish. 
```

Now, run the 3rd cell from the top. This sell start with '#Loading in the Spectrogram variables'. After starting the cell to run a text box should appear asking you to: "Please type filename without the file ending here". In the textbox type in ``Standard`` and press return. This will load in a file that tells this script how to compute the spectrogram. What the different variables mean, and how to change it is covered in lab two. 

![How to load a Spec file.](images/SpecFileLoad.png)

After that is done, press run (or `shift+return`) another three times. This should start the 6th cell that has only two lines in it.
```python
ComputeSpectrograms(SOURCE_FOLDER_TRAIN,OUTPUT_FOLDER_TRAIN)
pickle.dump(SpectrumVariables, open(os.path.join(OUTPUT_FOLDER_TRAIN,'Main.SpecVar'), "wb" ))
```
It will start printing out a bit of information and information about files it has a problem with. For now, we do not have to worry about those messages and we can just wait until the system is finished. To check if it's done, look at the `In [*]:` box in the top left corner of the cell. If it turns into a number, its finished. 

## Training the neural network

After the previous step has finished computing the images, we are ready to proceed to the next step. For that, please open the notebook `TrainingResNet` in the folder `02_Training`. Let us just try to run the whole notebook, either by clicking `Run` a couple of times or by clicking on the menu item `Cell` and selecting `Run All`. The complete execution will probably take a while. It can up to 30 minutes depending on your computers speed number of cores. It is a perfect time, however, to go through the notebook and trying to understand what is going on.

Here are a few interesting sections to look at.



Just like in the previous step we define the important parameters for training the network.
By leaving the variable `SPECTRUM_IMAGES_CLASSES_TEST` empty we create automatically a test and training data set with a 80%/20% ratio split. 
```python
SPECTRUM_IMAGES_CLASSES_TRAIN = '../GeneratedData/Cats-Vs-Dogs/'
SPECTRUM_IMAGES_CLASSES_TEST = ''
INPUT_RESOLUTION = 224
```
The following section then loads the data from the specified folder and looks for the classes of the data based on the sub-folders. THe script reports the classes it found and if it used an 80%/20% split or not.

```shell
Using one data set and separating them with an 80%/20% split
['Cats', 'Dogs']
```
Two cells down we display one batch of images with their respective labels written below each image.


The next couple of cells deal with loading the model. The most interesting section here is the following:
```python 
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, len(classes))
```
In these three lines of code, we basically replace the last layer of our network with one that fits our problem. In this ``len(classes)`` gives us the number of classes we loaded in with our data set. In this example, it would be two (``classes =['Cats', 'Dogs']``).
The other two lines before hand basically tell the network, not to change thos values when it tries to tweak the parameters. 


## Running the Neural Net

This is also known as inference. In this step, we basically put the neural net to a real-world test. We let the network infer from the incoming audio which class its thinks is the best fit.
This involves creating an audio buffer that we continually update with information from the microphone, then creating an image and running it through the neural net. This happens as fast as possible over and over again.

The underlying Python code is a bit more complex and so Lab 3 will address the details on that. For now, we have a simple Jupyter Notebook calls the more actual script. So, please open the notebook `ResNetInference` in the folder `ResNetInference`. 

 