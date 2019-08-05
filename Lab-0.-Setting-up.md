
# Getting Set-up
Before we can get started we need to download a few things. First we need to download the project files we will use for this workshop

You can either download them directly from GitHub to your computer just follow [this link](https://github.com/DavidGoedicke/RealtimeAudioClassification) and click the download button on the right side. If you are familiar with using a command-line/terminal or git you can [clone/(fork-clone)](HowToForkHowToClone.md) the repository directly. 

## How to use Jupyter Notebooks
The main ["IDE"](https://en.wikipedia.org/wiki/Integrated_development_environment) that we will be using this week are Jupyter Notebooks. Jupyter Notebooks is great to try out different code snippets to selectively debug the code you are writing.

Everything in these notebooks resolves around cells. The current active cell has a blue or green stripe on the left side. ([Why green or blue? Look here!](https://medium.com/ibm-data-science-experience/back-to-basics-jupyter-notebooks-dfcdc19c54bc)) Normally a cell contains a section of code that is executed together, e.g. We often define each function in its own cell.

![Example Image of how a cell looks like.](images/ExampleCell.png)

You can *edit* the code that is in that cell by clicking into the cell. 

To *run* the code in the cell, you can click on the button on the top that says ` >| Run` or press `shift + return` at the same time on your keyboard. 

This will run this cell and select the cell below. If you want to run multiple cells in succession, you can press `shift + return` a few times to quickly run the whole notebook.

If you followed the setup steps to [install the packages](https://github.com/DavidGoedicke/RealtimeAudioClassification/wiki/Lab-0.-Setting-up) and [download the data-set](https://github.com/DavidGoedicke/RealtimeAudioClassification/wiki/Lab-0.-Setting-up#download-datasets), then we can proceed in running the relevant code.

## Installing software tools and libraries
Before we can use these project files, we have to install a few tools and software libraries.

1. [Install Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html). (Some might already have prepared that)
2. Start Jupyter notebooks in the folder that you cloned to with you downloaded the Git Hub Repository (i.e. in the folder you just downloaded).
3. Within Jupyter, navigate to the folder `00_Setup` and open the Notebook inside. The notebook is called `Setup.ipynb`. 
4. Follow the steps inside that notebook to install and test the software libraries we need for this workshop.

## Download datasets
For the first lab we will use [our very own Cats-Vs-Dogs](Cats-Vs-Dogs.md) dataset. We will be distributing this dataset in person directly, through [Dropbox](https://www.dropbox.com/sh/pgy6tn4ugbfag0j/AADuiHrW-XgbwCDqiKUrMQ6Na?dl=0), USB-stick or AirDrop. 

When these datasets are finished downloading or copying to the computer, unzip and move them to the ``AudioData/`` folder in your workshop repository folder. Any Dataset you might want to train on later should go in there to make the use of the existing scripts easy. 

For later exercise we will also use the UrbanSound dataset. You can download it here:

[Urban Sounds](https://urbansounddataset.weebly.com/download-urbansound.html)

Other data set can be found on [Kaggle](https://www.kaggle.com). Kaggle is a data science community site run by Google, which has, among other things, datasets for machine learning.