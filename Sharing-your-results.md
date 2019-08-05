## Saving the model

One of the most important aspects of training a neural network is saving the trained model. This part of the script is already implemented and runs each time you run the `TrainingResNet` python notebook. 

The following code section, does the job.
```python
SpectrumVariables = pickle.load(open(os.path.join(SPECTRUM_IMAGES_CLASSES_TRAIN,'Main.SpecVar'), "rb" ) )
torch.save({
    'model':model.state_dict(),
    'classes':classes,
    'resolution':INPUT_RESOLUTION,
    'SpectrumVariables':SpectrumVariables,
    'modelType':"resnet18" # <= If you try out different models make sure to change this too
},"../models/CatDogResNet.pth") # <=Edit file name here 
```

If you changed your model by e.g. training on different classes then before, you should change the file name at the very end of this block of python code.
So, If e.g. the model can now distinguish birds and cats you might want to change this string, the file name to
```python "../models/CatDogResNet.pth" =>"../models/CatBirdsResNet.pth" ```.
After changing the name run the cell again. 

It will create a file that you can use in the `ResNetInference` notebook or for sharing with other people. 