## How to use Jupyter Notebooks

The main ["IDE"](https://en.wikipedia.org/wiki/Integrated_development_environment) that we will be using this week are Jupyter Notebooks. Jupyter Notebooks is great to try out different code snippets to selectively debug the code you are writing.

Everything in these notebooks resolves around cells. The current active cell has a blue or green stripe on the left side. ([Why green or blue? Look here!](https://medium.com/ibm-data-science-experience/back-to-basics-jupyter-notebooks-dfcdc19c54bc)) Normally a cell contains a section of code that is executed together. E.g. We often define each function in their own cell.
![Example Image of how a cell looks like.](images/ExampleCell.png)

When clicking into a cell you can edit the code that is in that cell. To run the code you can click on the button on the top that says ` >| Run` or press `shift + return` at the same time on your keyboard. This will run this cell and select the cell below. This behavior means that if you want to run multiple cells at the same time you can press `shift + return` a few times to quickly run the whole notebook.

If you followed the setup steps to [install the packages](https://github.com/DavidGoedicke/RealtimeAudioClassification/wiki/Lab-0.-Setting-up) and [download the data-set](https://github.com/DavidGoedicke/RealtimeAudioClassification/wiki/Lab-0.-Setting-up#download-datasets) then we can proceed in running the relevant code.


1. Generating Spectrums - as a first step we need to convert the audio-data into image-data.