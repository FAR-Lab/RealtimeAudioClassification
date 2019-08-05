## How to use Jupyter Notebooks

The main ["IDE"](https://en.wikipedia.org/wiki/Integrated_development_environment) that we will be using this week are Jupyter Notebooks. Jupyter Notebooks is great to try out different code snippets to selectively debug the code you are writing.

Everything in these notebooks resolves around cells. The current active cell has a blue or green stripe on the left side. ([Why green or blue? Look here!](https://medium.com/ibm-data-science-experience/back-to-basics-jupyter-notebooks-dfcdc19c54bc)) Normally a cell contains a section of code that is executed together. E.g. We often define each function in their own cell.
![Example Image of how a cell looks like.](images/ExampleCell.png)

When clicking into a cell you can edit the code that is in that cell. To run the code you can click on the button on the top that says ` >| Run` or press `shift + return` at the same time on your keyboard. This will run this cell and select the cell below. This behavior means that if you want to run multiple cells at the same time you can press `shift + return` a few times to quickly run the whole notebook.

![How to load a Spec file.](images/SpecFileLoad.png)


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