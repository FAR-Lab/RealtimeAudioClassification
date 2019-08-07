```python 
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, len(classes))
```
In these three lines of code, we basically replace the last layer of our network with one that fits our problem. In this ``len(classes)`` gives us the number of classes we loaded in with our data set. In this example, it would be two (``classes =['Cats', 'Dogs']``).
The other two lines beforehand basically tell the network, not to change any values in the network when it tries to tweak the parameters. Since we replace the last layer **after** this loop `.requires_grad` is automatically set to true again.

```python
for i in range(5):  #loop over the dataset multiple times
```