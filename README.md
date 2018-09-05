# Find the Phone Task with nearly 90% accuracy

## Python Libraries:
```
Sklearn, OpenCV, Numpy, matplotlib, Keras, Tensorflow
```
## Code Execution:

It  Suppose the data directory is in the home directory

### Terminal Command:
The train_phone_finder.py file receives the path to training data directory to generates a model.h5 file, which would be used later by the find_phone.py file to predict the location of phone.
```
> python train_phone_finder.py ~/find_phone
```
The find_phone.py file takes the path to one image to predict the location of phone in that image
```
> python find_phone.py ~/find_phone_test_images/51.jpg
```


